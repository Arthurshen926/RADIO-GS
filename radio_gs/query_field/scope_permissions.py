"""Query-independent regional exclusion permissions, without raster dependencies.

Outputs candidate-region permissions, not Gaussian negatives or membership.
The scene compiler binds these coordinates to the field's accepted-region axis.
"""
import torch
from torch import nn


class ScopePermissions(nn.Module):
    KEYS={'candidate_views','accepted_views','overlap_rows','overlap_columns',
          'comparable_rows','comparable_columns'}

    def __init__(self,state):
        super().__init__()
        if set(state)!=self.KEYS:raise ValueError('incomplete scope graph')
        for k,v in state.items():
            if v.ndim!=1 or v.dtype!=torch.long or bool((v<0).any()):raise ValueError('invalid graph coordinates')
            self.register_buffer(k,v.detach().clone())
        for prefix in ['overlap','comparable']:
            r=getattr(self,prefix+'_rows');c=getattr(self,prefix+'_columns')
            if r.shape!=c.shape or bool((r>=len(self.candidate_views)).any()) or bool((c>=len(self.accepted_views)).any()):
                raise ValueError('graph edge outside region domain')
            if not torch.equal(self.candidate_views[r],self.accepted_views[c]):raise ValueError('cross-view scope edge')

    @torch.no_grad()
    def forward(self,accepted):
        if accepted.dtype!=torch.bool or accepted.shape!=self.accepted_views.shape:
            raise ValueError('acceptance must align with field regions')
        if accepted.device!=self.accepted_views.device:raise ValueError('graph and query devices differ')
        flags=[]
        for prefix in ['overlap','comparable']:
            rows=getattr(self,prefix+'_rows');columns=getattr(self,prefix+'_columns')
            count=torch.zeros_like(self.candidate_views).scatter_add_(0,rows,accepted[columns].long())
            flags.append(count>0)
        return ~flags[0]&flags[1]


def compile_view(masks,scales,accepted_indices,view_index):
    """Construction only; no queries, detectors or benchmark answers are inputs."""
    if masks.ndim!=3 or masks.dtype!=torch.bool or scales.shape!=(len(masks),):raise ValueError('invalid region rasters')
    if accepted_indices.dtype!=torch.long or accepted_indices.ndim!=1 or len(accepted_indices.unique())!=len(accepted_indices):
        raise ValueError('invalid accepted coordinate binding')
    if bool(((accepted_indices<0)|(accepted_indices>=len(masks))).any()):raise ValueError('accepted coordinate outside candidates')
    if not bool(torch.isfinite(scales).all()) or bool((scales<=0).any()):raise ValueError('invalid scale')
    target=masks[accepted_indices].flatten(1).float();flat=masks.flatten(1).float()
    overlap=torch.cat([(x@target.T)>0 for x in flat.split(64)],0)
    ratio=scales[:,None]/scales[accepted_indices][None,:].clamp_min(1e-6)
    comparable=(ratio>=.5)&(ratio<=2)
    a,b=overlap.nonzero(as_tuple=True);c,d=comparable.nonzero(as_tuple=True)
    return dict(candidate_views=torch.full((len(masks),),view_index,dtype=torch.long),
        accepted_views=torch.full((len(accepted_indices),),view_index,dtype=torch.long),
        overlap_rows=a,overlap_columns=b,comparable_rows=c,comparable_columns=d)


def concatenate(states):
    values={k:[] for k in ScopePermissions.KEYS};rows=columns=0
    for state in states:
        for k,v in state.items():
            offset=rows if k.endswith('_rows') else columns if k.endswith('_columns') else 0
            values[k].append(v+offset)
        rows+=len(state['candidate_views']);columns+=len(state['accepted_views'])
    return {k:torch.cat(v) for k,v in values.items()}
