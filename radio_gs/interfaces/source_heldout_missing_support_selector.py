"""Low-capacity monotone selector for source-heldout missing support."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from radio_gs.querying.source_heldout_missing_support import FEATURE_NAMES


SELECTOR_FEATURE_NAMES = (
    "edge_comembership_reliability_positive",
    "log1p_source_observation_count_positive",
    "source_observation_agreement_positive",
    "negative_target_selected_scale_mean_positive",
    "negative_target_selected_scale_median_positive",
    "negative_target_selected_scale_covered_fraction_positive",
    "negative_seed_to_target_median_ratio_positive",
    "target_visibility_positive",
    "coverage_deficit_positive",
)
SOURCE_FEATURE_INDICES = (0, 1, 2, 5, 6, 7, 8, 10, 11)


@dataclass(frozen=True)
class MonotoneHeldoutSupportRanker:
    location: torch.Tensor
    scale: torch.Tensor
    positive_weights: torch.Tensor
    bias: torch.Tensor


def oriented_features(features: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(features).detach().double().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != len(FEATURE_NAMES)
        or values.shape[0] <= 0
        or not bool(torch.isfinite(values).all())
        or bool((values[:, 1] < 0.0).any())
    ):
        raise ValueError("heldout-support selector features differ")
    selected = values[:, SOURCE_FEATURE_INDICES].clone()
    selected[:, 1] = torch.log1p(selected[:, 1])
    # The candidate precondition already requires an above-boundary anchor.
    # Conditional on that evidence, lower target core coverage is exactly the
    # missing-support event that needs completion; orient those four channels
    # negatively while leaving relation/observation/visibility and deficit
    # positively oriented as same-object and missing-mass evidence.
    selected[:, 3:7] = -selected[:, 3:7]
    return selected.contiguous()


def _robust_location_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    location = values.median(dim=0).values
    scale = 1.4826 * (values - location).abs().median(dim=0).values
    varying = values.amax(dim=0) > values.amin(dim=0)
    scale[(scale <= 0.0) & varying] = 1.0
    scale[scale <= 0.0] = 1.0
    return location.contiguous(), scale.contiguous()


def scene_query_target_balanced_weights(
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    target_region_indices: torch.Tensor,
) -> torch.Tensor:
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    targets = torch.as_tensor(target_region_indices).detach().long().cpu().reshape(-1)
    if (
        scenes.shape != queries.shape
        or scenes.shape != targets.shape
        or scenes.numel() <= 0
        or bool((scenes < 0).any())
        or bool((queries < 0).any())
        or bool((targets < 0).any())
    ):
        raise ValueError("heldout-support selector balance axes differ")
    # Remap arbitrary scene/query values to compact exact groups without
    # relying on a bounded integer packing convention.
    triples = torch.stack((scenes, queries, targets), dim=1)
    unique_group, inverse, counts = torch.unique(
        triples, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    unique_scene = torch.unique(scenes, sorted=True)
    scene_lookup = {int(value): index for index, value in enumerate(unique_scene.tolist())}
    groups_per_scene = torch.zeros(unique_scene.numel(), dtype=torch.double)
    group_scene = torch.tensor(
        [scene_lookup[int(value)] for value in unique_group[:, 0].tolist()],
        dtype=torch.long,
    )
    groups_per_scene.scatter_add_(
        0, group_scene, torch.ones(group_scene.numel(), dtype=torch.double)
    )
    group_mass = groups_per_scene[group_scene].reciprocal()
    weights = group_mass[inverse] / counts[inverse].double()
    return (weights / weights.mean()).contiguous()


def fit_monotone_ranker(
    features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    target_region_indices: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> MonotoneHeldoutSupportRanker:
    values = oriented_features(features)
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    regions = torch.as_tensor(target_region_indices).detach().long().cpu().reshape(-1)
    if (
        target.shape != (values.shape[0],)
        or scenes.shape != target.shape
        or queries.shape != target.shape
        or regions.shape != target.shape
        or int(target.sum()) <= 0
        or int((~target).sum()) <= 0
        or not math.isfinite(float(l2_strength))
        or float(l2_strength) <= 0.0
        or int(maximum_iterations) <= 0
    ):
        raise ValueError("heldout-support selector fit inputs differ")
    location, scale = _robust_location_scale(values)
    normalized = (values - location) / scale
    sample_weight = scene_query_target_balanced_weights(scenes, queries, regions)
    prevalence = (
        (sample_weight * target.double()).sum() / sample_weight.sum()
    ).clamp(1e-6, 1.0 - 1e-6)
    raw_weight = torch.nn.Parameter(
        torch.full(
            (len(SELECTOR_FEATURE_NAMES),),
            math.log(math.expm1(0.1)),
            dtype=torch.float64,
        )
    )
    bias = torch.nn.Parameter(torch.logit(prevalence).reshape(()).clone())
    optimizer = torch.optim.LBFGS(
        (raw_weight, bias),
        lr=1.0,
        max_iter=int(maximum_iterations),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        positive = F.softplus(raw_weight)
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
    positive = F.softplus(raw_weight.detach())
    if not bool(torch.isfinite(positive).all()) or not bool(
        torch.isfinite(bias.detach())
    ):
        raise RuntimeError("heldout-support selector fit is invalid")
    return MonotoneHeldoutSupportRanker(
        location=location.float().contiguous(),
        scale=scale.float().contiguous(),
        positive_weights=positive.float().contiguous(),
        bias=bias.detach().float().reshape(()).contiguous(),
    )


def ranker_probability(
    model: MonotoneHeldoutSupportRanker, features: torch.Tensor
) -> torch.Tensor:
    values = oriented_features(features).float()
    if (
        model.location.shape != (len(SELECTOR_FEATURE_NAMES),)
        or model.scale.shape != model.location.shape
        or model.positive_weights.shape != model.location.shape
        or model.bias.shape != ()
        or bool((model.scale <= 0.0).any())
        or bool((model.positive_weights < 0.0).any())
    ):
        raise ValueError("heldout-support selector model differs")
    normalized = (values - model.location) / model.scale
    return torch.sigmoid(
        normalized @ model.positive_weights + model.bias
    ).contiguous()


__all__ = [
    "MonotoneHeldoutSupportRanker",
    "SELECTOR_FEATURE_NAMES",
    "SOURCE_FEATURE_INDICES",
    "fit_monotone_ranker",
    "oriented_features",
    "ranker_probability",
    "scene_query_target_balanced_weights",
]
