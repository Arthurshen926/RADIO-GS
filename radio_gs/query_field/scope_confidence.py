"""Preserve the existing per-view native confidence in union evidence readout."""
import torch
from .part_conditioning import native_region_probability,read_parts
from .scope_readout import ScopeReadout


@torch.no_grad()
def confident_scope(field,prior,probability):
    available=field.observed|field.semantic_measure.supported
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    preference=torch.where(mass>0,(2*probability-1).clamp(0,1),0.)
    accepted=preference>0
    confidence=prior.new_zeros(field.evidence.view_count).scatter_reduce_(0,field.evidence.views,preference,reduce='amax',include_self=True)
    prepared=field.scope_constraints.paired(accepted,field.scope_permissions)
    positive=prepared[False][0]*confidence[None,:];result={}
    for expanded in [False,True]:
        negative=prepared[expanded][1]*confidence[None,:]
        detail=dict(prepared[expanded][2],native_view_confidence_preserved=True,confidence_sum=float(confidence.sum()))
        result[expanded]=read_parts(prior,available,(positive,negative,detail))
    return result,accepted


@torch.no_grad()
def query_text_scope_confident(field,query,negatives,*,expanded=True):
    available=field.observed|field.semantic_measure.supported
    background=field.semantic.score(negatives).max(1).values
    fallback=(10*(field.semantic.score(query).flatten()-background)).sigmoid()
    prior=field.semantic_measure.union_probabilities(query,negatives,fallback,'measure',10.).masked_fill(~available,0)
    probability=native_region_probability(field.evidence,query,negatives)
    results,accepted=confident_scope(field,prior,probability);member,details=results[expanded]
    return ScopeReadout(prior,member,member>.5,accepted,dict(details,experimental=True,transient_query_fitting=False,cached_query_response=False))
