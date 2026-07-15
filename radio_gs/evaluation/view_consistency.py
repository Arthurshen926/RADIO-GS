"""Pure aggregation helpers for multiview primitive-consistency audits."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


TRAINING_SUM_KEYS = (
    "observation_count",
    "weight_sum",
    "weight_square_sum",
    "cosine_sum",
    "cosine_square_sum",
    "weighted_cosine_sum",
    "weighted_cosine_square_sum",
)


def merge_training_partials(partials: Sequence[Mapping]) -> dict[str, torch.Tensor]:
    """Sum disjoint view shards after validating their row shapes."""

    if not partials:
        raise ValueError("at least one training partial is required")
    merged: dict[str, torch.Tensor] = {}
    row_count: int | None = None
    for partial in partials:
        for key in TRAINING_SUM_KEYS:
            if key not in partial:
                raise ValueError(f"training partial lacks {key}")
            value = torch.as_tensor(partial[key]).cpu()
            if value.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional")
            if row_count is None:
                row_count = int(value.numel())
            elif int(value.numel()) != row_count:
                raise ValueError("training partial row counts differ")
            if key not in merged:
                merged[key] = torch.zeros_like(
                    value,
                    dtype=torch.long if key == "observation_count" else torch.float64,
                )
            merged[key] += value.to(dtype=merged[key].dtype)
    return merged


def consistency_from_sums(sums: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive weighted/unweighted agreement, variance, and effective views."""

    count = torch.as_tensor(sums["observation_count"]).double()
    weight = torch.as_tensor(sums["weight_sum"]).double()
    weight_square = torch.as_tensor(sums["weight_square_sum"]).double()
    cosine_sum = torch.as_tensor(sums["cosine_sum"]).double()
    cosine_square_sum = torch.as_tensor(sums["cosine_square_sum"]).double()
    weighted_cosine_sum = torch.as_tensor(sums["weighted_cosine_sum"]).double()
    weighted_cosine_square_sum = torch.as_tensor(
        sums["weighted_cosine_square_sum"]
    ).double()
    mean = cosine_sum / count.clamp_min(1.0)
    variance = (cosine_square_sum / count.clamp_min(1.0) - mean.square()).clamp_min(0.0)
    weighted_mean = weighted_cosine_sum / weight.clamp_min(1e-12)
    weighted_variance = (
        weighted_cosine_square_sum / weight.clamp_min(1e-12)
        - weighted_mean.square()
    ).clamp_min(0.0)
    effective_views = weight.square() / weight_square.clamp_min(1e-12)
    valid = count > 0
    for value in (mean, variance, weighted_mean, weighted_variance, effective_views):
        value[~valid] = 0.0
    return {
        "valid": valid,
        "mean_cosine": mean.float(),
        "cosine_variance": variance.float(),
        "weighted_mean_cosine": weighted_mean.float(),
        "weighted_cosine_variance": weighted_variance.float(),
        "view_disagreement": (1.0 - mean).masked_fill(~valid, 0.0).float(),
        "weighted_view_disagreement": (
            1.0 - weighted_mean
        ).masked_fill(~valid, 0.0).float(),
        "effective_views": effective_views.float(),
    }


def pearson_spearman(x: torch.Tensor, y: torch.Tensor) -> dict[str, float | None]:
    """Return correlations without making SciPy a hard runtime dependency."""

    left = torch.as_tensor(x).detach().float().cpu().reshape(-1)
    right = torch.as_tensor(y).detach().float().cpu().reshape(-1)
    finite = torch.isfinite(left) & torch.isfinite(right)
    left = left[finite]
    right = right[finite]
    result: dict[str, float | None] = {
        "samples": int(left.numel()),
        "pearson": None,
        "spearman": None,
    }
    if left.numel() <= 1:
        return result
    if float(left.std(unbiased=False)) <= 0 or float(right.std(unbiased=False)) <= 0:
        return result
    result["pearson"] = float(torch.corrcoef(torch.stack([left, right]))[0, 1])
    try:
        from scipy.stats import spearmanr

        result["spearman"] = float(spearmanr(left.numpy(), right.numpy()).statistic)
    except (ImportError, AttributeError, ValueError):
        result["spearman"] = None
    return result

