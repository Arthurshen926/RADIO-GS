"""Scale-conditioned grouping in the existing structural coordinates.

Registered prompts read map elements directly; native text/image semantics are
not rewritten. Scale selection is fitted/validated on source views only.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ScaleAffinityReadout(nn.Module):
    def __init__(self, structural_dim, scales, default_scale=0, native_seed_queries=False, full_prompt_features=False):
        super().__init__()
        if len(scales)<1 or any(s<=0 for s in scales):raise ValueError('positive physical scales required')
        self.full_prompt_features=full_prompt_features
        self.native_seed_queries=native_seed_queries
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
        self.register_buffer('default_scale',torch.tensor(default_scale,dtype=torch.long))
        lo,hi=float(torch.tensor(scales).log().min()),float(torch.tensor(scales).log().max())
        self.cutoffs=nn.Parameter(torch.linspace(lo-2,hi,structural_dim))
        self.register_buffer('xyz',None,persistent=False)
        self.register_buffer('minimum',None,persistent=False)
        self.register_buffer('extent',None,persistent=False)

    def bind_geometry(self,xyz,normalization):
        self.xyz=xyz.detach()
        self.minimum=normalization['minimum'].to(xyz)
        self.extent=normalization['extent'].to(xyz)
        if self.xyz.ndim!=2 or self.xyz.shape[1]!=3 or not torch.isfinite(self.xyz).all() or not (self.extent>0).all():raise ValueError('finite registered geometry required')

    def features(self,structure,scale):
        scale=torch.as_tensor(scale,device=structure.device,dtype=structure.dtype)
        gate=torch.sigmoid(3*(scale.log()-self.cutoffs))
        return F.normalize(structure*gate,dim=-1)

    def seed_rows(self,query):
        if self.xyz is None:raise ValueError('registered geometry must be bound')
        if query.ndim!=2 or query.shape[1]<6:raise ValueError('registered signed query required')
        points=query[:,-6:].reshape(-1,2,3)*self.extent+self.minimum
        # Coordinates are quantized by the existing query protocol; nearest
        # geometry snapping uses all carriers, never labels or target imagery.
        return torch.cdist(points.reshape(-1,3),self.xyz).argmin(1).reshape(-1,2)

    def logits_at_scale(self,structure,positive,negative,scale,rows=None):
        features=F.normalize(structure,dim=-1) if self.full_prompt_features else self.features(structure,scale)
        selected=features if rows is None else features[rows]
        cp=selected@features[positive];cn=selected@features[negative]
        probability=torch.sigmoid(12*(cp-.5))*torch.sigmoid(12*(cp-cn))
        return torch.logit(probability.clamp(1e-6,1-1e-6))

    def native_logits(self,cosine,structure,observed,rows=None):
        """Fixed native matching selects seeds; structure determines their extent.

        Single-group positive-query prototype: it does not establish absence
        detection or complete same-category multi-instance union retrieval.
        """
        positive=cosine.masked_fill(~observed[:,None],-torch.inf).argmax(0)
        negative=cosine.masked_fill(~observed[:,None],torch.inf).argmin(0)
        return torch.stack([self.logits_at_scale(structure,p,n,self.scales[self.default_scale],rows)
                            for p,n in zip(positive,negative)],dim=1)

    def forward(self,query,structure,observed,rows=None):
        result=[]
        for positive,negative in self.seed_rows(query):
            if positive==negative or not observed[positive] or not observed[negative]:
                # No silent relocation to another object's supported element.
                result.append(torch.full((len(structure) if rows is None else len(rows),),-20.,device=structure.device))
            else:
                result.append(self.logits_at_scale(structure,positive,negative,self.scales[self.default_scale],rows))
        return torch.stack(result,1)
