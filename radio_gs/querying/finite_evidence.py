"""Finite source support for a query-transient reconstruction.

The native map contributes one unit of reference strength. Each source frame
contributes either its detector confidence or positive logit once, regardless of
pixel count or optimizer iterations. These are heuristic support weights, not
calibrated identity odds or independent observations. Full-cohort controls reject
both as defaults: retaining a native prior can suppress a correct rare detection.
"""
import torch


def combine_with_native(proposal, native, strength):
    """Convex predictive mixture, permitting strong evidence to replace native."""
    return (native + strength * proposal) / (1 + strength)


def source_strength(confidences, measure='confidence'):
    weights = torch.stack(confidences)
    if measure == 'positive_logit':
        # SAM accepts scores above .5: near-threshold hypotheses should carry
        # near-zero strength, while one clear source can override a wrong map.
        # Use the detector's logit as an uncalibrated margin, not identity odds.
        eps = torch.finfo(weights.dtype).eps
        weights = torch.logit(weights.clamp(eps, 1-eps)).clamp_min(0)
    return weights.sum(0)
