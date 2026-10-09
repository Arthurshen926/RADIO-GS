"""Menu-conditioned native semantic measures owned by a Gaussian scene.

Competition and reliability precede spatial integration. No class names, source
frames, images or masks are stored. This is uncompressed functional state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SemanticMeasure(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        if set(state) != {'descriptors','pointer','index','mass','supported'}:
            raise ValueError('incomplete semantic measure')
        for k,v in state.items(): self.register_buffer(k,v.detach())
        if self.descriptors.ndim!=2 or not len(self.descriptors): raise ValueError('empty descriptors')
        if self.pointer.dtype!=torch.long or self.index.dtype!=torch.long:
            raise ValueError('measure indices must be int64')
        if self.pointer.shape!=(count+1,) or int(self.pointer[0]) or int(self.pointer[-1])!=len(self.index):
            raise ValueError('invalid measure CSR')
        if bool((self.pointer.diff()<0).any()) or self.mass.shape!=self.index.shape:
            raise ValueError('invalid measure rows')
        if bool(((self.index<0)|(self.index>=len(self.descriptors))).any()):raise ValueError('invalid semantic atom')
        if self.supported.dtype!=torch.bool or self.supported.shape!=(count,):raise ValueError('invalid semantic support')
        if not torch.equal(self.pointer.diff()>0,self.supported):raise ValueError('support does not match measure')
        if not torch.isfinite(self.descriptors).all() or not torch.isfinite(self.mass).all() or bool((self.mass<=0).any()):
            raise ValueError('invalid semantic measure values')
        total=self.operator()@torch.ones(len(self.descriptors),1,device=self.mass.device)
        if not torch.allclose(total[self.supported],torch.ones_like(total[self.supported]),atol=2e-5,rtol=0):
            raise ValueError('semantic measure is not normalized')

    def operator(self):
        return torch.sparse_csr_tensor(self.pointer,self.index,self.mass,
            size=(len(self.supported),len(self.descriptors)),device=self.mass.device)

    @torch.no_grad()
    def probabilities(self, queries, fallback, rule='reliable', temperature=100.):
        if rule not in {'mean','measure','reliable'}:raise ValueError('unknown semantic measure rule')
        if queries.ndim!=2 or queries.shape[0]<2 or queries.shape[1]!=self.descriptors.shape[1]:
            raise ValueError('a complete menu in native coordinates is required')
        if fallback.shape!=(len(self.supported),len(queries)) or not torch.isfinite(fallback).all():
            raise ValueError('invalid fallback matrix')
        if not torch.isfinite(queries).all() or not math.isfinite(temperature) or temperature<=0:
            raise ValueError('invalid query or temperature')
        q=F.normalize(queries.to(self.mass),dim=1)
        atoms=F.normalize(self.descriptors.to(self.mass),dim=1)
        identity=atoms@q.T;transfer=self.operator()
        if rule=='mean':result=(temperature*(transfer@identity)).softmax(1)
        else:
            response=(temperature*identity).softmax(1)
            result=transfer@response
            if rule=='reliable':
                entropy=-(response*response.clamp_min(torch.finfo(response.dtype).tiny).log()).sum(1)
                reliability=(1-entropy/math.log(len(queries))).clamp(0,1)
                mass=transfer@reliability[:,None]
                numerator=transfer@(reliability[:,None]*response)
                result=torch.where(mass>0,numerator/mass.clamp_min(torch.finfo(mass.dtype).tiny),result)
        return torch.where(self.supported[:,None],result,fallback.to(result))

    @torch.no_grad()
    def union_probabilities(self, query, negatives, fallback, rule='measure', temperature=10.):
        """Independent foreground union, not competition against other requests.

        Generic background directions define absolute admission before spatial
        integration. No best-mode-relative gate or candidate-count truncation.
        The mean control collapses both foreground and background logits first.
        """
        if rule not in {'mean','measure'}:raise ValueError('unknown semantic union rule')
        if query.numel()!=self.descriptors.shape[1] or negatives.ndim!=2 or not len(negatives) or negatives.shape[1]!=query.numel():
            raise ValueError('union requires aligned foreground and background directions')
        if not torch.isfinite(query).all() or not torch.isfinite(negatives).all():raise ValueError('nonfinite union inputs')
        if fallback.shape!=(len(self.supported),) or not torch.isfinite(fallback).all():raise ValueError('invalid union fallback')
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('invalid union temperature')
        q=F.normalize(query.to(self.mass).flatten(),dim=0)
        neg=F.normalize(negatives.to(self.mass),dim=1)
        directions=F.normalize(self.descriptors.to(self.mass),dim=1)
        positive=directions@q;negative=directions@neg.T;operator=self.operator()
        if rule=='mean':
            response=(temperature*((operator@positive[:,None]).flatten()-(operator@negative).max(1).values)).sigmoid()
        else:
            response=(operator@(temperature*(positive-negative.max(1).values)).sigmoid()[:,None]).flatten()
        return torch.where(self.supported,response,fallback.to(response))
