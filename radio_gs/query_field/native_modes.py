"""Gaussian-bound native appearance measures and their query functions.

Modes bind to Gaussian support coordinates, not an external observation map.
The uncompressed local atoms are model state; their cost is explicitly counted.
No source frame layout, patch raster, image or source reader exists here.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import QueryResult, Support
from .identity_evidence import native_rank_fusion


class NativeModes(nn.Module):
    def __init__(self, state, count):
        super().__init__()
        expected={'anchors','semantic','atoms','pointer','index','mass','mean','relation_columns'}
        if set(state)!=expected:raise ValueError('incomplete native mode field')
        for k,v in state.items():self.register_buffer(k,v.detach())
        n=len(self.anchors)
        if not n or self.anchors.dtype!=torch.long or bool(((self.anchors<0)|(self.anchors>=count)).any()):
            raise ValueError('modes require valid Gaussian anchors')
        if self.pointer.shape!=(n+1,) or int(self.pointer[0]) or int(self.pointer[-1])!=len(self.index):
            raise ValueError('invalid appearance measure CSR')
        if bool((self.pointer.diff()<=0).any()) or self.mass.shape!=self.index.shape:
            raise ValueError('each mode needs a nonempty measure')
        if bool(((self.index<0)|(self.index>=len(self.atoms))).any()):raise ValueError('invalid atom index')
        if self.semantic.ndim!=2 or len(self.semantic)!=n or self.mean.shape!=(n,self.atoms.shape[1]):
            raise ValueError('unaligned native modes')
        if self.relation_columns.shape!=(n,) or bool((self.relation_columns<0).any()):
            raise ValueError('mode support binding absent')
        for value in [self.semantic,self.atoms,self.mean,self.mass]:
            if not torch.isfinite(value).all():raise ValueError('nonfinite native modes')
        if bool((self.mass<=0).any()):raise ValueError('appearance mass must be positive')

    @torch.no_grad()
    def local_scores(self, query, weights=None, rule='measure', chunk=16, query_chunk=128):
        query=F.normalize(query.to(self.atoms).reshape(-1,self.atoms.shape[1]),dim=1)
        weights=torch.ones(len(query),device=query.device) if weights is None else weights.to(query)
        if weights.shape!=(len(query),) or not torch.isfinite(weights).all() or bool((weights<0).any()) or not bool(weights.sum()>0):
            raise ValueError('invalid local query mass')
        keep=weights>0;query=query[keep];weights=weights[keep];weights=weights/weights.sum()
        if rule=='mean':
            # Same weighted Chamfer operation, with each mode collapsed to one atom.
            cosine=query@F.normalize(self.mean,dim=1).T
            return .5*((cosine*weights[:,None]).sum(0)+cosine.max(0).values)
        if rule!='measure':raise ValueError('unknown native local function')
        # Every supported atom participates; batching is not a candidate cap.
        atoms=F.normalize(self.atoms,dim=1)
        reverse=query.new_full((len(atoms),),-torch.inf)
        forward_mean=query.new_zeros(len(self.anchors))
        for qstart in range(0,len(query),query_chunk):
            cosine=query[qstart:qstart+query_chunk]@atoms.T
            reverse=torch.maximum(reverse,cosine.max(0).values)
            for start in range(0,len(self.anchors),chunk):
                end=min(len(self.anchors),start+chunk)
                lengths=self.pointer[start+1:end+1]-self.pointer[start:end]
                positions=self.pointer[start:end,None]+torch.arange(int(lengths.max()),device=query.device)
                valid=torch.arange(positions.shape[1],device=query.device)[None,:]<lengths[:,None]
                ids=self.index[positions.clamp_max(len(self.index)-1)]
                forward=cosine[:,ids].masked_fill(~valid[None,:,:],-torch.inf).max(2).values
                forward_mean[start:end]+=(forward*weights[qstart:qstart+query_chunk,None]).sum(0)
        result=[]
        for start in range(0,len(self.anchors),chunk):
            end=min(len(self.anchors),start+chunk)
            lengths=self.pointer[start+1:end+1]-self.pointer[start:end]
            positions=self.pointer[start:start+len(lengths),None]+torch.arange(int(lengths.max()),device=query.device)
            valid=torch.arange(positions.shape[1],device=query.device)[None,:]<lengths[:,None]
            positions=positions.clamp_max(len(self.index)-1)
            ids=self.index[positions];mass=self.mass[positions].masked_fill(~valid,0)
            mass=mass/mass.sum(1,keepdim=True)
            result.append(.5*(forward_mean[start:end]+(reverse[ids]*mass).sum(1)))
        return torch.cat(result)


@torch.no_grad()
def query_modes(field, packet):
    """All supported modes compete; one inferred identity uses the shared member."""
    packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
    modes=field.native_modes
    semantic=F.normalize(modes.semantic,dim=1)@F.normalize(packet.semantic.to(field.xyz).flatten(),dim=0)
    if packet.modality=='image' and packet.local is not None:
        local=modes.local_scores(packet.local,packet.local_weights,field.native_mode_rule)
        identity=native_rank_fusion(semantic,local,torch.ones_like(semantic,dtype=torch.bool))
    else:identity=semantic
    chosen=int(identity.argmax());scale=field.default_scale if packet.scale is None else packet.scale
    # A mode is only a source-derived identity hypothesis, never a pure instance.
    if packet.task=='instance':admitted=torch.tensor([chosen],device=identity.device)
    else:admitted=torch.where(identity>=identity.max()-.02)[0]
    values=[]
    for i in admitted:
        support=Support(modes.anchors[i].reshape(1),torch.ones(1,device=identity.device))
        logits=field.member_logits(support,scale)
        values.append(field.exclude_membership(logits.sigmoid(),packet.negative,scale,logits))
    membership=torch.stack(values).max(0).values.masked_fill(~field.member_available,0)
    # Expose every candidate with its identity weight for independent attribution.
    weights=torch.softmax(10*identity,0)
    return QueryResult(field.semantic.score(packet.semantic).flatten(),membership,membership>field.threshold,
        modes.anchors,weights,float(scale),dict(shared_member_core=True,source_access=False,
        native_mode_rule=field.native_mode_rule,identity_modes=len(identity),chosen_mode=chosen,
        chosen_anchor=int(modes.anchors[chosen]),hard_positive_constraint=False,
        local_function='weighted bidirectional native cosine',source_patch_layout_retained=False,
        uncompressed_native_measure=True,identity_weights_calibrated=False))
