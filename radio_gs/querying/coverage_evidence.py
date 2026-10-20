"""Transport teacher probability and confidence mass to an observation grid.

The routine has no scene/query-specific parameter or access to target labels.
It preserves a confident subpixel foreground instead of deleting its positive
permission through low-resolution binary erosion. It does not infer identity.
"""
import torch
from torch.nn import functional as F


def coverage_evidence(logits, size):
    if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError('finite two-dimensional teacher logits required')
    if min(size) <= 0 or size[0] > logits.shape[0] or size[1] > logits.shape[1]:
        raise ValueError('coverage integration requires a coarser positive grid')
    p = logits.float().sigmoid()[None, None]
    confidence = (2*p-1).abs()
    mass = F.interpolate(confidence, size=size, mode='area')
    positive = F.interpolate(p*confidence, size=size, mode='area')
    target = torch.where(mass > 1e-8, positive/mass.clamp_min(1e-8), .5)
    return target.flatten()[:,None], mass.flatten()[:,None]
