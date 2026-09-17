"""Combine source memberships without giving vanishing observations a full vote."""
import torch


def source_membership_consensus(probability, visible_mass, *, minimum_mass=1e-6):
    """Return contribution-weighted membership and known support over views.

    Inputs are V x N. Missing detections must be omitted or have zero mass;
    this function never infers a negative label from a missing detection.
    Mass measures rendering evidence, not calibrated teacher correctness.
    """
    if probability.ndim != 2 or visible_mass.shape != probability.shape:
        raise ValueError('view/element axes differ')
    if minimum_mass < 0 or not torch.isfinite(visible_mass).all() or (visible_mass < 0).any():
        raise ValueError('invalid visible mass')
    known = visible_mass > minimum_mass
    if not torch.isfinite(probability[known]).all() or ((probability[known] < 0) | (probability[known] > 1)).any():
        raise ValueError('invalid known membership')
    weight = torch.where(known, visible_mass, 0)
    total = weight.sum(0)
    numerator = (torch.where(known, probability, 0) * weight).sum(0)
    return numerator / total.clamp_min(torch.finfo(total.dtype).tiny), total > 0
