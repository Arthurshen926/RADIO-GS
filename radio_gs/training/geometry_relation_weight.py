"""Use frozen geometry to emphasize already labelled, nearby negative pairs.

Geometry never supplies a semantic label: coplanar objects and curved objects
remain governed by the source mask evidence. The weight is scale invariant and
normal-sign invariant; zero/unknown normals contribute no extra evidence.
"""
import torch
from torch.nn import functional as F


def geometry_negative_weights(xyz, normals, rows, positive_row, negative_labels, scale):
    if negative_labels.dtype != torch.bool or negative_labels.shape != rows.shape:
        raise ValueError('one Boolean negative label per row required')
    scale = torch.as_tensor(scale, device=xyz.device, dtype=xyz.dtype)
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError('finite positive object scale required')
    with torch.no_grad():
        n = normals[rows]
        anchor = normals[positive_row]
        known = torch.isfinite(n).all(-1) & (n.norm(dim=-1) > 1e-6)
        known &= torch.isfinite(anchor).all() & (anchor.norm() > 1e-6)
        angle = 1 - (F.normalize(torch.nan_to_num(n), dim=-1) * F.normalize(torch.nan_to_num(anchor), dim=0)).sum(-1).abs().clamp(0, 1)
        distance = (xyz[rows] - xyz[positive_row]).norm(dim=-1)
        emphasis = 2 * torch.exp(-distance / (.1 * scale)) * angle
        return 1 + torch.where(negative_labels & known, emphasis, 0.)
