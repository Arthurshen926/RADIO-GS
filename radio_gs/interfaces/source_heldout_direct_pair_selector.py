"""Monotone direct-edge ranker for source-heldout missing support."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from radio_gs.models.region_comembership_native_v3 import PAIR_FEATURE_NAMES
from radio_gs.querying.source_heldout_missing_support import FEATURE_NAMES
from radio_gs.interfaces.source_heldout_missing_support_selector import (
    scene_query_target_balanced_weights,
)


DIRECT_FEATURE_NAMES = FEATURE_NAMES + tuple(
    f"direct_pair_{name}" for name in PAIR_FEATURE_NAMES
)
BASE_INDICES = (0, 1, 2, 5, 6, 7, 8, 10, 11)
PAIR_INDICES = (
    0, 1, 4, 5, 6, 8, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23,
    25, 26, 27, 28, 29,
)
SELECTED_INDICES = BASE_INDICES + tuple(len(FEATURE_NAMES) + i for i in PAIR_INDICES)
NEGATIVE_ORIENTATION_SOURCE_INDICES = {
    5, 6, 7, 8,
    len(FEATURE_NAMES) + 8,
    len(FEATURE_NAMES) + 25,
    len(FEATURE_NAMES) + 26,
}
SELECTOR_FEATURE_NAMES = tuple(
    ("negative_" if index in NEGATIVE_ORIENTATION_SOURCE_INDICES else "positive_")
    + DIRECT_FEATURE_NAMES[index]
    for index in SELECTED_INDICES
)


@dataclass(frozen=True)
class DirectPairMonotoneRanker:
    location: torch.Tensor
    scale: torch.Tensor
    positive_weights: torch.Tensor
    bias: torch.Tensor


def oriented_direct_features(features: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(features).detach().double().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != len(DIRECT_FEATURE_NAMES)
        or values.shape[0] <= 0
        or not bool(torch.isfinite(values).all())
        or bool((values[:, 1] < 0.0).any())
    ):
        raise ValueError("direct-pair heldout-support features differ")
    selected = values[:, SELECTED_INDICES].clone()
    observation_column = SELECTED_INDICES.index(1)
    selected[:, observation_column] = torch.log1p(selected[:, observation_column])
    for column, source in enumerate(SELECTED_INDICES):
        if source in NEGATIVE_ORIENTATION_SOURCE_INDICES:
            selected[:, column].neg_()
    return selected.contiguous()


def _robust_location_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    location = values.median(dim=0).values
    scale = 1.4826 * (values - location).abs().median(dim=0).values
    varying = values.amax(dim=0) > values.amin(dim=0)
    scale[(scale <= 0.0) & varying] = 1.0
    scale[scale <= 0.0] = 1.0
    return location.contiguous(), scale.contiguous()


def fit_direct_pair_monotone_ranker(
    features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    target_region_indices: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> DirectPairMonotoneRanker:
    values = oriented_direct_features(features)
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    targets = torch.as_tensor(target_region_indices).detach().long().cpu().reshape(-1)
    if (
        target.shape != (values.shape[0],)
        or scenes.shape != target.shape
        or queries.shape != target.shape
        or targets.shape != target.shape
        or int(target.sum()) <= 0
        or int((~target).sum()) <= 0
        or not math.isfinite(float(l2_strength))
        or float(l2_strength) <= 0.0
        or int(maximum_iterations) <= 0
    ):
        raise ValueError("direct-pair heldout-support fit inputs differ")
    location, scale = _robust_location_scale(values)
    normalized = (values - location) / scale
    sample_weight = scene_query_target_balanced_weights(
        scenes, queries, targets
    )
    prevalence = (
        (sample_weight * target.double()).sum() / sample_weight.sum()
    ).clamp(1e-6, 1.0 - 1e-6)
    raw = torch.nn.Parameter(
        torch.full(
            (len(SELECTOR_FEATURE_NAMES),),
            math.log(math.expm1(0.1)),
            dtype=torch.float64,
        )
    )
    bias = torch.nn.Parameter(torch.logit(prevalence).reshape(()).clone())
    optimizer = torch.optim.LBFGS(
        (raw, bias),
        lr=1.0,
        max_iter=int(maximum_iterations),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        positive = F.softplus(raw)
        logits = normalized @ positive + bias
        loss = (
            sample_weight
            * F.binary_cross_entropy_with_logits(
                logits, target.double(), reduction="none"
            )
        ).sum() / sample_weight.sum()
        loss = loss + float(l2_strength) * positive.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    positive = F.softplus(raw.detach())
    if not bool(torch.isfinite(positive).all()) or not bool(
        torch.isfinite(bias.detach())
    ):
        raise RuntimeError("direct-pair heldout-support fit is invalid")
    return DirectPairMonotoneRanker(
        location=location.float().contiguous(),
        scale=scale.float().contiguous(),
        positive_weights=positive.float().contiguous(),
        bias=bias.detach().float().reshape(()).contiguous(),
    )


def direct_pair_ranker_probability(
    model: DirectPairMonotoneRanker, features: torch.Tensor
) -> torch.Tensor:
    values = oriented_direct_features(features).float()
    if (
        model.location.shape != (len(SELECTOR_FEATURE_NAMES),)
        or model.scale.shape != model.location.shape
        or model.positive_weights.shape != model.location.shape
        or model.bias.shape != ()
        or bool((model.scale <= 0.0).any())
        or bool((model.positive_weights < 0.0).any())
    ):
        raise ValueError("direct-pair heldout-support model differs")
    return torch.sigmoid(
        ((values - model.location) / model.scale) @ model.positive_weights
        + model.bias
    ).contiguous()


__all__ = [
    "DIRECT_FEATURE_NAMES",
    "DirectPairMonotoneRanker",
    "SELECTED_INDICES",
    "SELECTOR_FEATURE_NAMES",
    "direct_pair_ranker_probability",
    "fit_direct_pair_monotone_ranker",
    "oriented_direct_features",
]
