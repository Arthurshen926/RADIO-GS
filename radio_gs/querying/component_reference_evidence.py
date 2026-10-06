"""Retain disconnected positive components only with measured reference support.

This is an additive query-time correction to an existing seeded-component mask.
Reference SAM support is a teacher hypothesis, never verified physical identity.
Pixels outside that support are not assigned a negative training label.  Missing
or subpixel reference visibility cannot establish a new component's identity.
"""
from __future__ import annotations

import torch


def retain_reference_supported_components(
    labels: torch.Tensor,
    candidate: torch.Tensor,
    retained: torch.Tensor,
    foreground_mass: torch.Tensor,
    visibility_mass: torch.Tensor,
    *,
    minimum_purity: float = 0.9,
    minimum_pixel_mass: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Aggregate actual renderer-adjoint masses before making a scope decision.

    ``labels`` names spatial components in the positive-candidate graph.  Masses
    are measured over the full authorized reference raster, in effective pixels,
    using the same observed-normalized renderer as inference.  They must not be
    unit-normalized per element: a pure but invisible tail is not strong support.
    """
    labels = torch.as_tensor(labels)
    candidate = torch.as_tensor(candidate, device=labels.device)
    retained = torch.as_tensor(retained, device=labels.device)
    foreground_mass = torch.as_tensor(foreground_mass, device=labels.device)
    visibility_mass = torch.as_tensor(visibility_mass, device=labels.device)
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Component labels must be a one-dimensional long tensor")
    if candidate.dtype != torch.bool or retained.dtype != torch.bool:
        raise ValueError("Candidate and retained masks must be boolean")
    if any(x.shape != labels.shape for x in (candidate, retained, foreground_mass, visibility_mass)):
        raise ValueError("Component masks and masses must align")
    if labels.numel() and int(labels.min()) < 0:
        raise ValueError("Component labels cannot be negative")
    if not 0 < float(minimum_purity) <= 1 or not float(minimum_pixel_mass) > 0:
        raise ValueError("Positive purity and mass thresholds required")
    if not torch.isfinite(foreground_mass).all() or not torch.isfinite(visibility_mass).all():
        raise ValueError("Reference masses must be finite")
    if (foreground_mass < 0).any() or (visibility_mass < 0).any():
        raise ValueError("Reference masses must be nonnegative")
    if (foreground_mass > visibility_mass + 1e-4).any():
        raise ValueError("Foreground cannot exceed reference visibility")
    if (retained & ~candidate).any():
        raise ValueError("Existing retention must be within candidate support")
    count = int(labels.max()) + 1 if labels.numel() else 0
    total = torch.zeros(count, device=labels.device, dtype=torch.float64)
    positive = torch.zeros_like(total)
    total.scatter_add_(0, labels[candidate], visibility_mass[candidate].double())
    positive.scatter_add_(0, labels[candidate], foreground_mass[candidate].double())
    purity = positive / total.clamp_min(torch.finfo(total.dtype).tiny)
    admitted = (purity >= minimum_purity) & (positive >= minimum_pixel_mass)
    added = candidate & admitted[labels] & ~retained
    result = retained | added
    return result, dict(
        minimum_purity=float(minimum_purity),
        minimum_pixel_mass=float(minimum_pixel_mass),
        additional_components=int(labels[added].unique().numel()),
        additional_elements=int(added.sum()),
        additional_reference_foreground_mass=float(foreground_mass[added].sum()),
        additional_reference_visibility_mass=float(visibility_mass[added].sum()),
        subpixel_candidate_components=int(((total > 0) & (total < minimum_pixel_mass)).sum()),
        invisible_candidate_components=int((total == 0).sum()),
        semantic_negative_labels_created=False,
        physical_identity_verified=False,
        rule="component aggregate eroded-reference-SAM purity and absolute pixel mass",
    )
