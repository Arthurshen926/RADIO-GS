"""Exact-MPR transport of positive, negative, and unknown mask evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TernaryMaskEvidence:
    positive_mass: torch.Tensor
    negative_mass: torch.Tensor
    unknown_mass: torch.Tensor
    boundary_mass: torch.Tensor
    weighted_scale: torch.Tensor
    weighted_quality: torch.Tensor
    view_count: torch.Tensor
    disagreement: torch.Tensor


def aggregate_mask_evidence(
    gaussian_ids: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    boundary: torch.Tensor,
    scales: torch.Tensor,
    qualities: torch.Tensor,
    view_ids: torch.Tensor,
    *,
    num_gaussians: int,
) -> TernaryMaskEvidence:
    """Aggregate hit evidence; labels are 1 positive, 0 negative, -1 unknown."""

    gids = torch.as_tensor(gaussian_ids).long().reshape(-1)
    weight = torch.as_tensor(weights).float().reshape(-1)
    label = torch.as_tensor(labels).to(torch.int8).reshape(-1)
    edge = torch.as_tensor(boundary).float().reshape(-1)
    scale = torch.as_tensor(scales).float().reshape(-1)
    quality = torch.as_tensor(qualities).float().reshape(-1)
    views = torch.as_tensor(view_ids).long().reshape(-1)
    if not (gids.shape == weight.shape == label.shape == edge.shape == scale.shape == quality.shape == views.shape):
        raise ValueError("ternary mask evidence axes differ")
    if num_gaussians <= 0 or (gids.numel() and (int(gids.min()) < 0 or int(gids.max()) >= num_gaussians)):
        raise ValueError("Gaussian id outside declared domain")
    if bool((~torch.isin(label, torch.tensor([-1, 0, 1], dtype=torch.int8))).any()):
        raise ValueError("mask labels must be positive, negative, or unknown")
    if not bool(torch.isfinite(torch.stack((weight, edge, scale, quality))).all()) or bool((weight < 0).any()):
        raise ValueError("mask evidence weights and metadata must be finite")
    device = weight.device
    def total(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(num_gaussians, device=device)
        result.index_add_(0, gids, value)
        return result
    positive = total(weight * (label == 1))
    negative = total(weight * (label == 0))
    unknown = total(weight * (label == -1))
    observed = positive + negative
    boundary_mass = total(weight * edge.clamp(0, 1))
    weighted_scale = total(weight * scale.clamp(0, 1)) / total(weight).clamp_min(1e-8)
    weighted_quality = total(weight * quality.clamp(0, 1)) / total(weight).clamp_min(1e-8)
    pairs = torch.unique(torch.stack((gids, views), dim=1), dim=0)
    view_count = torch.zeros(num_gaussians, device=device)
    view_count.index_add_(0, pairs[:, 0], torch.ones(pairs.shape[0], device=device))
    disagreement = torch.where(
        observed > 0,
        2 * torch.minimum(positive, negative) / observed.clamp_min(1e-8),
        torch.zeros_like(observed),
    )
    return TernaryMaskEvidence(
        positive, negative, unknown, boundary_mass, weighted_scale,
        weighted_quality, view_count, disagreement,
    )


__all__ = ["TernaryMaskEvidence", "aggregate_mask_evidence"]
