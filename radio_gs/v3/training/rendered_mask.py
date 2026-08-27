"""Differentiable 3D-membership to source-mask training closure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RenderedMaskLoss:
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    brier: torch.Tensor
    boundary: torch.Tensor


def render_membership(
    posterior: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    contribution_weights: torch.Tensor,
    *,
    num_pixels: int,
) -> torch.Tensor:
    """Render the same Gaussian posterior used by 3D evaluation."""

    probability = torch.as_tensor(posterior).float().reshape(-1)
    gids = torch.as_tensor(gaussian_ids, device=probability.device).long().reshape(-1)
    pids = torch.as_tensor(pixel_ids, device=probability.device).long().reshape(-1)
    weights = torch.as_tensor(contribution_weights, device=probability.device).float().reshape(-1)
    if not (gids.shape == pids.shape == weights.shape):
        raise ValueError("render hit axes differ")
    if num_pixels <= 0 or (gids.numel() and (int(gids.min()) < 0 or int(gids.max()) >= probability.numel())):
        raise ValueError("render Gaussian domain differs")
    if pids.numel() and (int(pids.min()) < 0 or int(pids.max()) >= num_pixels):
        raise ValueError("render pixel domain differs")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("contribution weights must be finite and non-negative")
    output = torch.zeros(num_pixels, device=probability.device)
    output.index_add_(0, pids, weights * probability[gids])
    return output.clamp(0, 1)


def rendered_mask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    known: torch.Tensor | None = None,
    boundary_target: torch.Tensor | None = None,
    boundary_prediction: torch.Tensor | None = None,
) -> RenderedMaskLoss:
    score = torch.as_tensor(prediction).float().reshape(-1).clamp(1e-6, 1 - 1e-6)
    truth = torch.as_tensor(target, device=score.device).float().reshape(-1)
    authority = torch.ones_like(truth, dtype=torch.bool) if known is None else torch.as_tensor(known, device=score.device).bool().reshape(-1)
    if score.shape != truth.shape or authority.shape != truth.shape or not bool(authority.any()):
        raise ValueError("rendered mask authority axes differ or contain no known pixels")
    score, truth = score[authority], truth[authority].clamp(0, 1)
    bce = F.binary_cross_entropy(score, truth)
    dice = 1 - (2 * (score * truth).sum() + 1) / (score.sum() + truth.sum() + 1)
    brier = (score - truth).square().mean()
    boundary = score.new_zeros(())
    if boundary_target is not None or boundary_prediction is not None:
        if boundary_target is None or boundary_prediction is None:
            raise ValueError("boundary loss requires target and prediction")
        edge_truth = torch.as_tensor(boundary_target, device=score.device).float().reshape(-1)
        edge_score = torch.as_tensor(boundary_prediction, device=score.device).float().reshape(-1)
        if edge_truth.shape != authority.shape or edge_score.shape != authority.shape:
            raise ValueError("boundary axes differ")
        boundary = F.binary_cross_entropy_with_logits(edge_score[authority], edge_truth[authority].clamp(0, 1))
    return RenderedMaskLoss(bce + dice + brier + boundary, bce, dice, brier, boundary)


__all__ = ["RenderedMaskLoss", "render_membership", "rendered_mask_loss"]
