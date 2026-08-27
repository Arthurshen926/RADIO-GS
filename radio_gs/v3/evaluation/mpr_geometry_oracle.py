"""Source-only exact-MPR and frozen-geometry oracle ladder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from radio_gs.rendering.contribution_compositor import marginal_responsibility_statistics
from radio_gs.v3.evaluation.source_heldout import SourceHeldoutMetrics, evaluate_source_heldout
from radio_gs.v3.training.instance_upper_bound import MaskEpisode
from radio_gs.v3.training.rendered_mask import render_membership
from radio_gs.v3.training.rendered_mask import rendered_mask_loss


@dataclass(frozen=True)
class ExactMembership:
    probability: torch.Tensor
    observed: torch.Tensor
    semantic_mass: torch.Tensor


def exact_mpr_membership(episode: MaskEpisode, num_gaussians: int) -> ExactMembership:
    """Lift one binary source mask without sparse membership thresholding."""

    gids = episode.gaussian_ids.long()
    pids = episode.pixel_ids.long()
    base = episode.contribution_weights.float()
    target = episode.target.flatten().float()
    if num_gaussians <= 0 or target.numel() <= 0:
        raise ValueError("exact-MPR membership domains must be positive")
    statistics = marginal_responsibility_statistics(
        pids, base, num_pixels=target.numel()
    )
    semantic = statistics.target_weight
    denominator = torch.zeros(num_gaussians, dtype=torch.float32)
    numerator = torch.zeros_like(denominator)
    denominator.index_add_(0, gids, semantic)
    numerator.index_add_(0, gids, semantic * target[pids])
    observed = denominator > 0
    probability = torch.where(
        observed, numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(denominator),
    ).clamp(0, 1)
    return ExactMembership(probability, observed, denominator)


def union_memberships(values: list[ExactMembership]) -> ExactMembership:
    """Positive-only union; missing observations remain unknown, not negative."""

    if not values:
        raise ValueError("membership union requires at least one source observation")
    shape = values[0].probability.shape
    if any(item.probability.shape != shape for item in values):
        raise ValueError("membership union Gaussian axes differ")
    probability = torch.stack([item.probability for item in values]).amax(0)
    observed = torch.stack([item.observed for item in values]).any(0)
    mass = torch.stack([item.semantic_mass for item in values]).sum(0)
    return ExactMembership(probability, observed, mass)


def render_exact_membership(
    membership: ExactMembership, episode: MaskEpisode
) -> torch.Tensor:
    return render_membership(
        membership.probability,
        episode.gaussian_ids,
        episode.pixel_ids,
        episode.contribution_weights,
        num_pixels=episode.target.numel(),
    )


def prediction_boundary(prediction: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    image = prediction.reshape(shape)
    dilation = F.max_pool2d(image[None, None], 3, 1, 1)
    erosion = -F.max_pool2d(-image[None, None], 3, 1, 1)
    return torch.sigmoid((dilation - erosion)[0, 0] * 16.0 - 4.0).flatten()


def score_oracle_prediction(
    prediction: torch.Tensor, episode: MaskEpisode
) -> SourceHeldoutMetrics:
    return evaluate_source_heldout(
        prediction,
        episode.target.flatten(),
        episode.known.flatten(),
        episode.unknown.flatten(),
        prediction_boundary(prediction, tuple(episode.target.shape)),
        episode.boundary.flatten(),
    )


def positive_hit_coverage(
    membership: ExactMembership,
    episode: MaskEpisode,
    *,
    membership_threshold: float = 0.5,
) -> float:
    """Fraction of positive-pixel contribution mass owned by selected rows."""

    target_hit = episode.target.flatten()[episode.pixel_ids.long()]
    weight = episode.contribution_weights.float() * target_hit
    selected = membership.probability[episode.gaussian_ids.long()] >= membership_threshold
    return float(weight[selected].sum() / weight.sum().clamp_min(1e-12))


def optimize_same_view_posterior(
    initial: ExactMembership,
    episode: MaskEpisode,
    *,
    device: torch.device,
    steps: int = 100,
    learning_rate: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Target-driven diagnostic ceiling over visible Gaussian probabilities."""

    if steps <= 0 or learning_rate <= 0:
        raise ValueError("posterior oracle optimization budget must be positive")
    gids = episode.gaussian_ids.long()
    unique, inverse = torch.unique(gids, sorted=True, return_inverse=True)
    initial_probability = initial.probability[unique].clamp(1e-4, 1 - 1e-4)
    logits = torch.nn.Parameter(torch.logit(initial_probability).to(device))
    inverse = inverse.to(device)
    pids = episode.pixel_ids.long().to(device)
    weights = episode.contribution_weights.float().to(device)
    target = episode.target.flatten().float().to(device)
    known = episode.known.flatten().to(device)
    optimizer = torch.optim.Adam((logits,), lr=float(learning_rate))
    best_loss = float("inf")
    best = None
    for _step in range(int(steps)):
        prediction = torch.zeros(target.numel(), device=device)
        prediction.index_add_(0, pids, weights * logits.sigmoid()[inverse])
        loss = rendered_mask_loss(prediction, target, known=known).total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best = logits.detach().sigmoid().cpu().clone()
    if best is None:
        raise RuntimeError("same-view posterior oracle produced no state")
    dense = initial.probability.new_zeros(initial.probability.shape)
    dense[unique] = best
    return dense, best_loss


__all__ = [
    "ExactMembership",
    "exact_mpr_membership",
    "positive_hit_coverage",
    "optimize_same_view_posterior",
    "prediction_boundary",
    "render_exact_membership",
    "score_oracle_prediction",
    "union_memberships",
]
