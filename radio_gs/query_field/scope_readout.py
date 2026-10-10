"""Experimental text-union readout using only an exported field and query vectors.

No query fitting, cached per-query responses, source masks or source cameras.
The finite evidence decoder is explicit and is not the v10 joint member solver.
"""
from dataclasses import dataclass
import torch
from .part_conditioning import native_region_probability,read_parts


@dataclass(frozen=True)
class ScopeReadout:
    semantic_probability: torch.Tensor
    membership_score: torch.Tensor
    selected: torch.Tensor
    accepted_regions: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def query_text_scope(field,query,negatives,*,expanded=True):
    if field.scope_constraints is None or field.scope_permissions is None or field.semantic_measure is None:
        raise ValueError('field lacks compiled scope functions')
    available=field.observed|field.semantic_measure.supported
    background=field.semantic.score(negatives).max(1).values
    fallback=(10*(field.semantic.score(query).flatten()-background)).sigmoid()
    prior=field.semantic_measure.union_probabilities(query,negatives,fallback,'measure',10.).masked_fill(~available,0)
    probability=native_region_probability(field.evidence,query,negatives)
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    accepted=(probability>.5)&(mass>0)
    prepared=field.scope_constraints.paired(accepted,field.scope_permissions)[expanded]
    member,details=read_parts(prior,available,prepared)
    return ScopeReadout(prior,member,member>.5,accepted,
        dict(details,readout='independent semantic probability and actual member set',
             transient_query_fitting=False,cached_query_response=False,experimental=True))
