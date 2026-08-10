"""Source-only multi-scene monotone selector for missing-core proposals.

The scene index is used only to construct leakage-free folds and balanced
training weights.  It is deliberately absent from the six-feature inference
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
    OOF_FOLDS,
    SELECTOR_FEATURE_NAMES,
    oriented_selector_features,
    selector_probability,
)
from radio_gs.querying.source_conditioned_hierarchy_completion import (
    deterministic_group_folds,
)
from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    one_sided_wilson_lower,
)


MODEL_SCHEMA = "radio_gs.source_multiscene_monotone_missing_core_selector.v2"
MAX_REGION_ID_EXCLUSIVE = 1 << 32


@dataclass(frozen=True)
class SourceMultisceneMonotoneSelectorFit:
    fold_ids: torch.Tensor
    fold_models: tuple[MonotoneAdditiveLogistic, ...]
    oof_probability: torch.Tensor


def _validated_scene_and_region(
    scene_indices: torch.Tensor,
    region_groups: torch.Tensor,
    *,
    rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    regions = torch.as_tensor(region_groups).detach().long().cpu().reshape(-1)
    if (
        scenes.shape != regions.shape
        or (rows is not None and scenes.shape != (int(rows),))
        or scenes.numel() <= 0
        or bool((scenes < 0).any())
        or bool((regions < 0).any())
        or bool((regions >= MAX_REGION_ID_EXCLUSIVE).any())
        or int(torch.unique(scenes).numel()) < 2
    ):
        raise ValueError("multi-scene selector routing axes differ")
    scene_axis = torch.unique(scenes, sorted=True)
    if not torch.equal(scene_axis, torch.arange(scene_axis.numel())):
        raise ValueError("multi-scene selector scene indices must be contiguous")
    return scenes.contiguous(), regions.contiguous()


def packed_scene_region_groups(
    scene_indices: torch.Tensor, region_groups: torch.Tensor
) -> torch.Tensor:
    """Pack complete scene/region groups without cross-scene collisions."""

    scenes, regions = _validated_scene_and_region(scene_indices, region_groups)
    if int(scenes.max()) >= (1 << 31):
        raise ValueError("multi-scene selector scene axis is too large")
    return ((scenes << 32) | regions).contiguous()


def multiscene_region_fold_ids(
    scene_indices: torch.Tensor, region_groups: torch.Tensor
) -> torch.Tensor:
    """Assign every complete scene/region group to one stable OOF fold."""

    packed = packed_scene_region_groups(scene_indices, region_groups)
    unique, inverse = torch.unique(packed, sorted=True, return_inverse=True)
    group_folds = deterministic_group_folds(unique, num_folds=OOF_FOLDS)
    return group_folds[inverse].contiguous()


def scene_region_balanced_weights(
    scene_indices: torch.Tensor, region_groups: torch.Tensor
) -> torch.Tensor:
    """Give every scene equal mass and every region equal mass per scene."""

    scenes, regions = _validated_scene_and_region(scene_indices, region_groups)
    packed = packed_scene_region_groups(scenes, regions)
    unique_groups, inverse, counts = torch.unique(
        packed, sorted=True, return_inverse=True, return_counts=True
    )
    group_scenes = unique_groups >> 32
    scene_count = int(torch.unique(scenes).numel())
    groups_per_scene = torch.bincount(group_scenes, minlength=scene_count).double()
    if bool((groups_per_scene <= 0).any()):
        raise ValueError("multi-scene selector lacks a scene group population")
    per_row = counts[inverse].double().reciprocal()
    per_row = per_row / groups_per_scene[scenes]
    return (per_row / per_row.mean()).contiguous()


def _robust_location_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    location = values.median(dim=0).values
    scale = 1.4826 * (values - location).abs().median(dim=0).values
    varying = values.amax(dim=0) > values.amin(dim=0)
    scale[(scale <= 0.0) & varying] = 1.0
    scale[scale <= 0.0] = 1.0
    return location.contiguous(), scale.contiguous()


def fit_multiscene_monotone_additive_logistic(
    oriented_features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    region_groups: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> MonotoneAdditiveLogistic:
    """Fit the six-feature monotone model with scene/region-balanced BCE."""

    values = torch.as_tensor(oriented_features).detach().double().cpu()
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes, regions = _validated_scene_and_region(
        scene_indices, region_groups, rows=values.shape[0]
    )
    if (
        values.ndim != 2
        or values.shape[1] != len(SELECTOR_FEATURE_NAMES)
        or target.shape != (values.shape[0],)
        or values.shape[0] < 2
        or not bool(torch.isfinite(values).all())
        or int(target.sum()) <= 0
        or int((~target).sum()) <= 0
        or not math.isfinite(float(l2_strength))
        or float(l2_strength) <= 0.0
        or int(maximum_iterations) <= 0
    ):
        raise ValueError("multi-scene monotone-logistic inputs differ")
    location, scale = _robust_location_scale(values)
    normalized = (values - location) / scale
    sample_weight = scene_region_balanced_weights(scenes, regions)
    prevalence = (
        (sample_weight * target.double()).sum() / sample_weight.sum()
    ).clamp(1e-6, 1.0 - 1e-6)
    initial_raw_weight = math.log(math.expm1(0.1))
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
        raise RuntimeError("multi-scene monotone-logistic fit is invalid")
    return MonotoneAdditiveLogistic(
        location=location.float().contiguous(),
        scale=scale.float().contiguous(),
        positive_weights=positive_weight.float().contiguous(),
        bias=bias.detach().float().reshape(()).contiguous(),
    )


def fit_source_multiscene_monotone_selector_oof(
    unit_features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    region_groups: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> SourceMultisceneMonotoneSelectorFit:
    """Fit three scene/region-disjoint models and return OOF probabilities."""

    oriented = oriented_selector_features(unit_features)
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes, regions = _validated_scene_and_region(
        scene_indices, region_groups, rows=oriented.shape[0]
    )
    if target.shape != (oriented.shape[0],):
        raise ValueError("multi-scene OOF selector axes differ")
    fold_ids = multiscene_region_fold_ids(scenes, regions)
    prediction = torch.full((oriented.shape[0],), float("nan"))
    models: list[MonotoneAdditiveLogistic] = []
    for fold in range(OOF_FOLDS):
        heldout = fold_ids == fold
        training = ~heldout
        training_scene_axis = torch.unique(scenes[training], sorted=True)
        heldout_scene_axis = torch.unique(scenes[heldout], sorted=True)
        if (
            int(heldout.sum()) <= 0
            or int(training.sum()) <= 0
            or not torch.equal(training_scene_axis, heldout_scene_axis)
            or int(target[training].sum()) <= 0
            or int((~target[training]).sum()) <= 0
        ):
            raise ValueError(f"multi-scene OOF selector fold {fold} differs")
        model = fit_multiscene_monotone_additive_logistic(
            oriented[training],
            target[training],
            scenes[training],
            regions[training],
            l2_strength=l2_strength,
            maximum_iterations=maximum_iterations,
        )
        prediction[heldout] = selector_probability(model, oriented[heldout])
        models.append(model)
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("multi-scene OOF selector left invalid predictions")
    return SourceMultisceneMonotoneSelectorFit(
        fold_ids=fold_ids,
        fold_models=tuple(models),
        oof_probability=prediction.contiguous(),
    )


def select_largest_multiscene_safe_oof_threshold(
    probability: torch.Tensor,
    hard_labels: torch.Tensor,
    signed_utility: torch.Tensor,
    scene_indices: torch.Tensor,
    *,
    minimum_selected_per_scene: int = 256,
    minimum_overall_wilson_lower: float = 0.80,
    minimum_scene_wilson_lower: float = 0.75,
) -> dict[str, object]:
    """Select maximum global coverage passing overall and every-scene gates."""

    score = torch.as_tensor(probability).detach().float().cpu().reshape(-1)
    labels = torch.as_tensor(hard_labels).detach().bool().cpu().reshape(-1)
    utility = torch.as_tensor(signed_utility).detach().float().cpu().reshape(-1)
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    if (
        labels.shape != score.shape
        or utility.shape != score.shape
        or scenes.shape != score.shape
        or score.numel() <= 0
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(utility).all())
        or bool(((score < 0.0) | (score > 1.0)).any())
        or int(minimum_selected_per_scene) <= 0
        or not 0.5 < float(minimum_scene_wilson_lower) < 1.0
        or not float(minimum_scene_wilson_lower) <= float(
            minimum_overall_wilson_lower
        ) < 1.0
    ):
        raise ValueError("multi-scene selector threshold inputs differ")
    scene_axis = torch.unique(scenes, sorted=True)
    if not torch.equal(scene_axis, torch.arange(scene_axis.numel())):
        raise ValueError("multi-scene selector threshold scene axis differs")
    scene_count = int(scene_axis.numel())
    if scene_count < 2:
        raise ValueError("multi-scene selector threshold requires two scenes")

    one_hot = F.one_hot(scenes, num_classes=scene_count).long()
    unconditional_scene_utility = []
    for scene in range(scene_count):
        mask = scenes == scene
        unconditional_scene_utility.append(float(utility[mask].mean()))
    unconditional_overall_utility = float(utility.mean())

    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    sorted_label = labels[order].long()
    sorted_utility = utility[order]
    sorted_scene = one_hot[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = counts.cumsum(0)
    endpoints = stops - 1
    overall_positive = sorted_label.cumsum(0)[endpoints]
    overall_utility_sum = sorted_utility.cumsum(0)[endpoints]
    scene_selected = sorted_scene.cumsum(0)[endpoints]
    scene_positive = (sorted_scene * sorted_label[:, None]).cumsum(0)[endpoints]
    scene_utility_sum = (
        sorted_scene.double() * sorted_utility.double()[:, None]
    ).cumsum(0)[endpoints]

    eligible: list[dict[str, object]] = []
    for index, total_value in enumerate(stops.tolist()):
        total = int(total_value)
        positive = int(overall_positive[index])
        negative = total - positive
        if positive <= 0 or negative <= 0:
            continue
        overall_wilson = one_sided_wilson_lower(positive, total)
        overall_utility = float(overall_utility_sum[index] / total)
        if (
            overall_wilson < float(minimum_overall_wilson_lower)
            or overall_utility <= 0.0
            or overall_utility <= unconditional_overall_utility
        ):
            continue
        per_scene = []
        scene_gate = True
        for scene in range(scene_count):
            selected = int(scene_selected[index, scene])
            positives = int(scene_positive[index, scene])
            negatives = selected - positives
            if (
                selected < int(minimum_selected_per_scene)
                or positives <= 0
                or negatives <= 0
            ):
                scene_gate = False
                break
            wilson = one_sided_wilson_lower(positives, selected)
            mean_utility = float(scene_utility_sum[index, scene] / selected)
            if (
                wilson < float(minimum_scene_wilson_lower)
                or mean_utility <= 0.0
                or mean_utility <= unconditional_scene_utility[scene]
            ):
                scene_gate = False
                break
            per_scene.append(
                {
                    "scene_index": scene,
                    "selected": selected,
                    "hard_positive": positives,
                    "hard_negative": negatives,
                    "hard_precision": positives / selected,
                    "hard_precision_wilson95_lower": wilson,
                    "signed_utility_mean": mean_utility,
                    "unconditional_signed_utility_mean": unconditional_scene_utility[
                        scene
                    ],
                    "coverage_fraction": selected / int((scenes == scene).sum()),
                }
            )
        if scene_gate:
            eligible.append(
                {
                    "threshold_inclusive": float(sorted_score[endpoints[index]]),
                    "selected": total,
                    "hard_positive": positive,
                    "hard_negative": negative,
                    "hard_precision": positive / total,
                    "hard_precision_wilson95_lower": overall_wilson,
                    "signed_utility_mean": overall_utility,
                    "unconditional_signed_utility_mean": unconditional_overall_utility,
                    "coverage_fraction": total / score.numel(),
                    "per_scene": per_scene,
                }
            )
    if not eligible:
        raise RuntimeError("no multi-scene selector threshold passes every safety gate")
    return max(eligible, key=lambda row: int(row["selected"]))


def validate_multiscene_selector_model_payload(value: object) -> dict[str, object]:
    """Validate the target-facing, query-free v2 model contract."""

    if not isinstance(value, dict):
        raise ValueError("multi-scene selector model must be a mapping")
    required = {
        "schema",
        "schema_version",
        "feature_names",
        "source_unit_feature_indices",
        "fold_models",
        "threshold_inclusive",
        "target_probability",
        "execution_authority",
        "training_provenance",
    }
    if set(value) != required:
        raise ValueError("multi-scene selector model fields differ")
    if (
        value.get("schema") != MODEL_SCHEMA
        or value.get("schema_version") != 2
        or value.get("feature_names") != list(SELECTOR_FEATURE_NAMES)
        or value.get("target_probability")
        != "minimum_probability_across_three_fold_models"
        or not math.isfinite(float(value.get("threshold_inclusive", float("nan"))))
    ):
        raise ValueError("multi-scene selector model header differs")
    rows = value.get("fold_models")
    if not isinstance(rows, list) or len(rows) != OOF_FOLDS:
        raise ValueError("multi-scene selector fold-model axis differs")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "location",
            "scale",
            "positive_weights",
            "bias",
        }:
            raise ValueError("multi-scene selector fold-model fields differ")
        model = MonotoneAdditiveLogistic(
            location=torch.as_tensor(row["location"]).detach().float().cpu(),
            scale=torch.as_tensor(row["scale"]).detach().float().cpu(),
            positive_weights=torch.as_tensor(row["positive_weights"])
            .detach()
            .float()
            .cpu(),
            bias=torch.as_tensor(row["bias"]).detach().float().cpu().reshape(()),
        )
        selector_probability(model, torch.zeros(1, len(SELECTOR_FEATURE_NAMES)))
    provenance = value.get("training_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "source_scene_count",
        "source_scene_ids_sha256",
        "scene_identifier_used_for_balancing_and_folds_only",
        "query_identifier_used_as_feature",
        "scene_identifier_used_as_feature",
    }:
        raise ValueError("multi-scene selector provenance differs")
    if (
        provenance.get("source_scene_count") != 2
        or provenance.get("scene_identifier_used_for_balancing_and_folds_only")
        is not True
        or provenance.get("query_identifier_used_as_feature") is not False
        or provenance.get("scene_identifier_used_as_feature") is not False
    ):
        raise ValueError("multi-scene selector provenance header differs")
    return dict(value)


__all__ = [
    "MODEL_SCHEMA",
    "SourceMultisceneMonotoneSelectorFit",
    "fit_multiscene_monotone_additive_logistic",
    "fit_source_multiscene_monotone_selector_oof",
    "multiscene_region_fold_ids",
    "packed_scene_region_groups",
    "scene_region_balanced_weights",
    "select_largest_multiscene_safe_oof_threshold",
    "validate_multiscene_selector_model_payload",
]
