"""Low-capacity source-only selector for missing-core completion proposals."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from radio_gs.querying.source_conditioned_hierarchy_completion import (
    deterministic_group_folds,
)
from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    one_sided_wilson_lower,
)


SELECTOR_FEATURE_NAMES = (
    "unit_o0_score_positive",
    "appearance_concentration_positive",
    "boundary_concentration_positive",
    "negative_log1p_core_spatial_rms_radius",
    "negative_query_selected_scale_index",
    "negative_log1p_full_scalar_source_robust_ood_linf",
)
SOURCE_UNIT_FEATURE_INDICES = (0, 14, 15, 17, 9, 18)
OOF_FOLDS = 3


@dataclass(frozen=True)
class MonotoneAdditiveLogistic:
    location: torch.Tensor
    scale: torch.Tensor
    positive_weights: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class SourceMonotoneSelectorFit:
    fold_ids: torch.Tensor
    fold_models: tuple[MonotoneAdditiveLogistic, ...]
    oof_probability: torch.Tensor


def oriented_selector_features(unit_features: torch.Tensor) -> torch.Tensor:
    """Extract the six preregistered features and orient larger as safer."""

    values = torch.as_tensor(unit_features).detach().double().cpu()
    if (
        values.ndim != 2
        or values.shape[1] <= max(SOURCE_UNIT_FEATURE_INDICES)
        or values.shape[0] <= 0
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("missing-core selector unit features differ")
    selected = values[:, SOURCE_UNIT_FEATURE_INDICES].clone()
    if bool((selected[:, 3] < 0).any()) or bool((selected[:, 5] < 0).any()):
        raise ValueError("missing-core selector non-negative channels differ")
    selected[:, 3] = -torch.log1p(selected[:, 3])
    selected[:, 4] = -selected[:, 4]
    selected[:, 5] = -torch.log1p(selected[:, 5])
    return selected.contiguous()


def _robust_location_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    location = values.median(dim=0).values
    scale = 1.4826 * (values - location).abs().median(dim=0).values
    varying = values.amax(dim=0) > values.amin(dim=0)
    scale[(scale <= 0.0) & varying] = 1.0
    scale[scale <= 0.0] = 1.0
    return location.contiguous(), scale.contiguous()


def _group_balanced_weights(groups: torch.Tensor) -> torch.Tensor:
    unique, inverse, counts = torch.unique(
        groups, sorted=True, return_inverse=True, return_counts=True
    )
    if unique.numel() <= 0:
        raise ValueError("missing-core selector training groups differ")
    weight = counts[inverse].double().reciprocal()
    return (weight / weight.mean()).contiguous()


def fit_monotone_additive_logistic(
    oriented_features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> MonotoneAdditiveLogistic:
    """Fit a group-balanced logistic with preregistered non-negative weights."""

    values = torch.as_tensor(oriented_features).detach().double().cpu()
    target = torch.as_tensor(labels).detach().bool().cpu()
    group = torch.as_tensor(groups).detach().long().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != len(SELECTOR_FEATURE_NAMES)
        or target.shape != (values.shape[0],)
        or group.shape != target.shape
        or values.shape[0] < 2
        or not bool(torch.isfinite(values).all())
        or bool((group < 0).any())
        or int(target.sum()) <= 0
        or int((~target).sum()) <= 0
        or not math.isfinite(float(l2_strength))
        or float(l2_strength) <= 0.0
        or int(maximum_iterations) <= 0
    ):
        raise ValueError("missing-core monotone-logistic inputs differ")
    location, scale = _robust_location_scale(values)
    normalized = (values - location) / scale
    sample_weight = _group_balanced_weights(group)
    prevalence = (
        (sample_weight * target.double()).sum() / sample_weight.sum()
    ).clamp(1e-6, 1.0 - 1e-6)
    initial_positive_weight = 0.1
    initial_raw_weight = math.log(math.expm1(initial_positive_weight))
    raw_weight = torch.nn.Parameter(
        torch.full(
            (len(SELECTOR_FEATURE_NAMES),),
            initial_raw_weight,
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
        positive_weight = F.softplus(raw_weight)
        logits = normalized @ positive_weight + bias
        per_row = F.binary_cross_entropy_with_logits(
            logits, target.double(), reduction="none"
        )
        loss = (sample_weight * per_row).sum() / sample_weight.sum()
        loss = loss + float(l2_strength) * positive_weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    positive_weight = F.softplus(raw_weight.detach())
    if (
        not bool(torch.isfinite(positive_weight).all())
        or not bool(torch.isfinite(bias.detach()))
        or bool((positive_weight < 0.0).any())
    ):
        raise RuntimeError("missing-core monotone-logistic fit is invalid")
    return MonotoneAdditiveLogistic(
        location=location.float().contiguous(),
        scale=scale.float().contiguous(),
        positive_weights=positive_weight.float().contiguous(),
        bias=bias.detach().float().reshape(()).contiguous(),
    )


def selector_probability(
    model: MonotoneAdditiveLogistic, oriented_features: torch.Tensor
) -> torch.Tensor:
    values = torch.as_tensor(oriented_features).detach().float().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != len(SELECTOR_FEATURE_NAMES)
        or model.location.shape != (len(SELECTOR_FEATURE_NAMES),)
        or model.scale.shape != model.location.shape
        or model.positive_weights.shape != model.location.shape
        or model.bias.shape != ()
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(model.location).all())
        or not bool(torch.isfinite(model.scale).all())
        or not bool(torch.isfinite(model.positive_weights).all())
        or bool((model.scale <= 0.0).any())
        or bool((model.positive_weights < 0.0).any())
    ):
        raise ValueError("missing-core monotone-logistic inference differs")
    normalized = (values - model.location) / model.scale
    return torch.sigmoid(
        normalized @ model.positive_weights + model.bias
    ).contiguous()


def fit_source_monotone_selector_oof(
    unit_features: torch.Tensor,
    labels: torch.Tensor,
    region_groups: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> SourceMonotoneSelectorFit:
    """Fit three region-disjoint models and return leakage-free OOF scores."""

    oriented = oriented_selector_features(unit_features)
    target = torch.as_tensor(labels).detach().bool().cpu()
    groups = torch.as_tensor(region_groups).detach().long().cpu()
    if target.shape != (oriented.shape[0],) or groups.shape != target.shape:
        raise ValueError("missing-core OOF selector axes differ")
    group_axis = torch.unique(groups, sorted=True)
    group_fold = deterministic_group_folds(group_axis, num_folds=OOF_FOLDS)
    fold_lookup = torch.full((int(group_axis.max()) + 1,), -1, dtype=torch.long)
    fold_lookup[group_axis] = group_fold
    fold_ids = fold_lookup[groups]
    if bool((fold_ids < 0).any()):
        raise RuntimeError("missing-core OOF selector fold lookup failed")
    prediction = torch.full((oriented.shape[0],), float("nan"))
    models: list[MonotoneAdditiveLogistic] = []
    for fold in range(OOF_FOLDS):
        heldout = fold_ids == fold
        training = ~heldout
        if (
            int(heldout.sum()) <= 0
            or int(training.sum()) <= 0
            or int(target[training].sum()) <= 0
            or int((~target[training]).sum()) <= 0
        ):
            raise ValueError(f"missing-core OOF selector fold {fold} differs")
        model = fit_monotone_additive_logistic(
            oriented[training],
            target[training],
            groups[training],
            l2_strength=l2_strength,
            maximum_iterations=maximum_iterations,
        )
        prediction[heldout] = selector_probability(model, oriented[heldout])
        models.append(model)
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("missing-core OOF selector left invalid predictions")
    return SourceMonotoneSelectorFit(
        fold_ids=fold_ids.contiguous(),
        fold_models=tuple(models),
        oof_probability=prediction.contiguous(),
    )


def target_consensus_probability(
    fold_models: tuple[MonotoneAdditiveLogistic, ...],
    unit_features: torch.Tensor,
) -> torch.Tensor:
    """Use the minimum fold probability as the frozen conservative target score."""

    if len(fold_models) != OOF_FOLDS:
        raise ValueError("missing-core target selector requires three fold models")
    oriented = oriented_selector_features(unit_features)
    return torch.stack(
        [selector_probability(model, oriented) for model in fold_models], dim=1
    ).amin(dim=1).contiguous()


def tie_invariant_average_precision(
    values: torch.Tensor, labels: torch.Tensor
) -> float:
    score = torch.as_tensor(values).detach().float().cpu()
    truth = torch.as_tensor(labels).detach().bool().cpu()
    if score.shape != truth.shape or score.ndim != 1:
        raise ValueError("missing-core selector AP inputs differ")
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    ranked = truth[order].float()
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = counts.cumsum(0)
    cumulative_true = ranked.cumsum(0)[stops - 1]
    precision = cumulative_true / stops.float()
    recall = cumulative_true / positives
    previous = torch.cat((torch.zeros(1), recall[:-1]))
    return float(((recall - previous) * precision).sum())


def select_largest_safe_oof_threshold(
    probability: torch.Tensor,
    hard_labels: torch.Tensor,
    signed_utility: torch.Tensor,
    *,
    minimum_selected: int = 256,
    minimum_wilson_lower: float = 0.80,
) -> dict[str, float | int]:
    """Select maximum OOF coverage satisfying the preregistered safety gate."""

    score = torch.as_tensor(probability).detach().float().cpu()
    labels = torch.as_tensor(hard_labels).detach().bool().cpu()
    utility = torch.as_tensor(signed_utility).detach().float().cpu()
    if (
        score.ndim != 1
        or labels.shape != score.shape
        or utility.shape != score.shape
        or score.numel() < int(minimum_selected)
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(utility).all())
        or bool((score < 0.0).any())
        or bool((score > 1.0).any())
        or int(minimum_selected) <= 0
        or not 0.5 < float(minimum_wilson_lower) < 1.0
    ):
        raise ValueError("missing-core selector threshold inputs differ")
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    sorted_label = labels[order].long()
    sorted_utility = utility[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = counts.cumsum(0)
    positive = sorted_label.cumsum(0)[stops - 1]
    utility_sum = sorted_utility.cumsum(0)[stops - 1]
    eligible: list[tuple[int, int, float, float, float]] = []
    for index, stop_value in enumerate(stops.tolist()):
        total = int(stop_value)
        positives = int(positive[index])
        negatives = total - positives
        if total < int(minimum_selected) or positives <= 0 or negatives <= 0:
            continue
        wilson = one_sided_wilson_lower(positives, total)
        mean_utility = float(utility_sum[index] / total)
        if wilson >= float(minimum_wilson_lower) and mean_utility > 0.0:
            eligible.append(
                (
                    total,
                    positives,
                    wilson,
                    mean_utility,
                    float(sorted_score[stops[index] - 1]),
                )
            )
    if not eligible:
        raise RuntimeError("no missing-core selector threshold passes the safety gate")
    total, positives, wilson, mean_utility, threshold = max(
        eligible, key=lambda row: row[0]
    )
    return {
        "threshold_inclusive": threshold,
        "selected": total,
        "hard_positive": positives,
        "hard_negative": total - positives,
        "hard_precision": positives / total,
        "hard_precision_wilson95_lower": wilson,
        "signed_utility_mean": mean_utility,
        "coverage_fraction": total / score.numel(),
    }


__all__ = [
    "MonotoneAdditiveLogistic",
    "OOF_FOLDS",
    "SELECTOR_FEATURE_NAMES",
    "SOURCE_UNIT_FEATURE_INDICES",
    "SourceMonotoneSelectorFit",
    "fit_monotone_additive_logistic",
    "fit_source_monotone_selector_oof",
    "oriented_selector_features",
    "select_largest_safe_oof_threshold",
    "selector_probability",
    "target_consensus_probability",
    "tie_invariant_average_precision",
]
