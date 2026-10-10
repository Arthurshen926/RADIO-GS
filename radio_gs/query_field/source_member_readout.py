"""Conditional source fraction on known support; semantic fallback on unknown.

This transfers an initialization principle, not v10's full source measure or
joint selected-silhouette solver. Tiny known mass can be overconfident.
"""
import torch
from .part_conditioning import integrate_views


def read_source_members(prior,available,prepared,*,protect=False):
    if available.dtype!=torch.bool or available.shape!=prior.shape:raise ValueError('invalid available domain')
    positive,negative,details=prepared
    _,exclusion=integrate_views(prior,positive,negative,protect=protect)
    p=positive.sum(1);n=exclusion.sum(1);known=p+n
    denominator=torch.where(known>0,known,torch.ones_like(known))
    scores=torch.where(known>0,p/denominator,prior).masked_fill(~available,0)
    return scores,dict(**details,positive_mass=float(p.sum()),negative_mass_before=float(negative.sum()),
        negative_mass_after=float(exclusion.sum()),other_view_protection=protect,source_access=False,
        source_normalized_members=True,prior_only_on_unknown=True,known_rows=int((known>0).sum()),
        tiny_evidence_can_be_overconfident=True,scores_calibrated=False,new_training=False)
