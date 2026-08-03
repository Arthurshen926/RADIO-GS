"""Robust multiview primitive reconstruction (MPR) targets and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrimitiveConsensus:
    targets: torch.Tensor
    valid: torch.Tensor
    observation_count: torch.Tensor
    reliability: torch.Tensor
    per_view_agreement: torch.Tensor


def consensus_target_rows(
    consensus: PrimitiveConsensus | Any,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Read target rows from either a dense consensus or a disk-backed MPR."""

    indices = torch.as_tensor(rows).detach().long().cpu()
    fetch = getattr(consensus, "fetch_rows", None)
    if callable(fetch):
        return torch.as_tensor(fetch(indices))
    return torch.as_tensor(consensus.targets)[indices]


def robust_multiview_consensus(
    observations: torch.Tensor,
    valid_observations: torch.Tensor,
    *,
    observation_weights: torch.Tensor | None = None,
    robust_temperature: float = 0.10,
    iterations: int = 2,
    normalize_observations: bool = False,
) -> PrimitiveConsensus:
    """Fuse ``[V,N,C]`` teacher rows without query or benchmark supervision.

    The estimate is initialized by a weighted mean, then reweighted using
    cosine agreement with the current primitive target.  Reliability exposes
    normalized view count, mean agreement, and agreement stability to the
    pointwise canonical fusion module.
    """

    values = torch.as_tensor(observations).float()
    valid = torch.as_tensor(valid_observations).bool()
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("observations/valid must be [V,N,C] and [V,N]")
    if robust_temperature <= 0 or iterations < 0:
        raise ValueError("robust_temperature must be positive; iterations non-negative")
    if normalize_observations:
        values = F.normalize(values, dim=-1, eps=1e-8)
    if observation_weights is None:
        base_weights = valid.to(values.dtype)
    else:
        base_weights = torch.as_tensor(observation_weights).to(values).float()
        if base_weights.shape != valid.shape or bool((base_weights < 0).any()):
            raise ValueError("observation_weights must be non-negative [V,N]")
        base_weights = base_weights * valid
    denominator = base_weights.sum(dim=0)
    target = (values * base_weights[..., None]).sum(dim=0) / denominator.clamp_min(
        1e-8
    )[..., None]
    agreement = torch.zeros_like(base_weights)
    robust_weights = base_weights
    for _ in range(iterations):
        normalized_target = F.normalize(target, dim=-1, eps=1e-8)
        normalized_values = F.normalize(values, dim=-1, eps=1e-8)
        agreement = (normalized_values * normalized_target[None]).sum(dim=-1)
        robust = torch.exp((agreement - 1.0) / robust_temperature)
        robust_weights = base_weights * robust
        robust_denominator = robust_weights.sum(dim=0)
        fallback = robust_denominator <= 1e-8
        if bool(fallback.any()):
            robust_weights[:, fallback] = base_weights[:, fallback]
            robust_denominator = robust_weights.sum(dim=0)
        target = (values * robust_weights[..., None]).sum(dim=0) / robust_denominator.clamp_min(
            1e-8
        )[..., None]

    observation_count = valid.sum(dim=0)
    valid_primitive = denominator > 0
    weighted_mean_agreement = (
        agreement * robust_weights
    ).sum(dim=0) / robust_weights.sum(dim=0).clamp_min(1e-8)
    variance = (
        robust_weights * (agreement - weighted_mean_agreement[None]).square()
    ).sum(dim=0) / robust_weights.sum(dim=0).clamp_min(1e-8)
    max_views = max(1, values.shape[0])
    reliability = torch.stack(
        [
            observation_count.float() / max_views,
            weighted_mean_agreement.clamp(-1.0, 1.0).add(1.0).mul(0.5),
            torch.exp(-variance),
        ],
        dim=-1,
    )
    target = target.masked_fill(~valid_primitive[:, None], 0.0)
    agreement = agreement.masked_fill(~valid, 0.0)
    return PrimitiveConsensus(
        targets=target,
        valid=valid_primitive,
        observation_count=observation_count,
        reliability=reliability,
        per_view_agreement=agreement,
    )


def primitive_reconstruction_loss(
    predicted: torch.Tensor,
    consensus: PrimitiveConsensus,
    *,
    row_indices: torch.Tensor | None = None,
    cosine_weight: float = 1.0,
    huber_weight: float = 0.25,
    huber_delta: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MPR loss used as a training bridge, never as a query-time cache."""

    if row_indices is None:
        if callable(getattr(consensus, "fetch_rows", None)):
            raise ValueError("disk-backed consensus requires explicit row_indices")
        target = consensus.targets.to(predicted.device)
        valid = consensus.valid.to(predicted.device)
        reliability = consensus.reliability[:, :2].mean(dim=-1).to(predicted.device)
    else:
        rows = torch.as_tensor(row_indices).long().cpu()
        target = consensus_target_rows(consensus, rows).to(predicted.device)
        valid = consensus.valid[rows].to(predicted.device)
        reliability = consensus.reliability[rows, :2].mean(dim=-1).to(predicted.device)
    if predicted.shape != target.shape:
        raise ValueError("predicted primitive rows do not align with consensus targets")
    if not bool(valid.any()):
        zero = predicted.sum() * 0.0
        return zero, {"cosine": zero.detach(), "huber": zero.detach(), "valid_ratio": zero.detach()}
    pred = predicted[valid].float()
    teacher = target[valid].float()
    weights = reliability[valid].clamp_min(1e-4)
    cosine = 1.0 - F.cosine_similarity(pred, teacher, dim=-1, eps=1e-8)
    huber = F.huber_loss(pred, teacher, delta=huber_delta, reduction="none").mean(dim=-1)
    normalizer = weights.sum().clamp_min(1e-8)
    cosine_loss = (cosine * weights).sum() / normalizer
    huber_loss = (huber * weights).sum() / normalizer
    total = cosine_weight * cosine_loss + huber_weight * huber_loss
    return total, {
        "cosine": cosine_loss.detach(),
        "huber": huber_loss.detach(),
        "valid_ratio": valid.float().mean().detach(),
    }
