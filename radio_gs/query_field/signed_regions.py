"""Condition field-owned partial regions jointly on explicitly signed support.

This is a function-only hypothesis, not parity with v10's RGB prompt expansion.
Unknown support abstains; excluded region mass is not renormalized into a vote.
"""
import torch
from .contracts import Support, QueryResult
from .confidence_mixture_evidence import ConfidenceMixtureEvidence


def signed_coefficients(evidence, positive, negative, scale):
    p=evidence.coefficients(positive,scale)
    n=torch.zeros_like(p) if negative is None else evidence.coefficients(negative,scale)
    return (p-n).clamp_min(0),p,n


@torch.no_grad()
def query(field, packet, *, joint=True):
    packet.validate(len(field.xyz),field.semantic.native_dim,field.local.native_dim)
    if packet.positive is None or packet.modality not in {'point_2d','point_3d'}:
        raise ValueError('signed region conditioning needs explicit spatial supports')
    if packet.negative is None:return field.query(packet)
    if not isinstance(field.evidence,ConfidenceMixtureEvidence) or field.support_aggregation!='normalized_prototype' or field.negative_rule!='product':
        raise ValueError('signed region control requires confidence-mixture prototype field')
    def filtered(support):
        ids=support.indices.to(field.xyz.device);weights=support.weights.to(field.xyz)
        valid=field.member_available[ids] & (weights>0)
        return None if not bool(valid.any()) else Support(ids[valid],weights[valid])
    positive=filtered(packet.positive);negative=filtered(packet.negative)
    if positive is None:raise ValueError('positive support lacks observed mass')
    if negative is None:return field.query(packet)
    scale=field.default_scale if packet.scale is None else packet.scale
    pos=field.member_logits(positive,scale,include_evidence=False).sigmoid()
    neg=field.member_logits(negative,scale,include_evidence=False).sigmoid()
    prior=pos*(1-neg)
    coefficient,p,n=signed_coefficients(field.evidence,positive,negative,scale)
    if not joint:coefficient=p
    mass=[torch.sparse.mm(field.evidence.matrix(sign),coefficient[:,None]).flatten() for sign in ['positive','negative']]
    probability=(prior+mass[0])/(1+mass[0]+mass[1])
    # Retain the exact legacy prior if the field has no applicable region.
    probability=probability.masked_fill(~field.member_available,0)
    if packet.exact_positive:
        probability[positive.indices]=1;probability[negative.indices]=0
    return QueryResult(None,probability,probability>field.threshold,positive.indices,
        positive.weights/positive.weights.sum(),float(scale),dict(shared_member_core=True,source_access=False,
        signed_region_conditioning=joint,source_rgb_prompt_expansion_transferred=False,
        prior='unaugmented positive member times one minus negative member',
        positive_region_mass=float(p.sum()),negative_region_mass=float(n.sum()),retained_region_mass=float(coefficient.sum()),
        positive_regions=int((p>0).sum()),negative_regions=int((n>0).sum()),retained_regions=int((coefficient>0).sum()),
        completely_removed_regions=int(((p>0)&(coefficient==0)).sum()),
        target_evidence_rows=int(((mass[0]+mass[1])>0).sum()),scores_calibrated=False,
        unknown_is_negative=False,renormalize_after_exclusion=False))
