"""Source-only selector v3 with scene/query/region balance and tail gates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
    OOF_FOLDS,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    oriented_selector_features,
    selector_probability,
)
from radio_gs.querying.source_conditioned_hierarchy_completion import (
    deterministic_group_folds,
)
from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    one_sided_wilson_lower,
)


MODEL_SCHEMA = "radio_gs.source_multiscene_query_balanced_missing_core_selector.v3"
MAX_QUERY_ID_EXCLUSIVE = 1 << 16
MAX_REGION_ID_EXCLUSIVE = 1 << 32


@dataclass(frozen=True)
class SourceMultisceneQueryBalancedSelectorFit:
    fold_ids: torch.Tensor
    fold_models: tuple[MonotoneAdditiveLogistic, ...]
    oof_probability: torch.Tensor


def _validated_axes(
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
    *,
    rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    regions = torch.as_tensor(region_indices).detach().long().cpu().reshape(-1)
    if (
        scenes.shape != queries.shape
        or scenes.shape != regions.shape
        or (rows is not None and scenes.shape != (int(rows),))
        or scenes.numel() <= 0
        or bool((scenes < 0).any())
        or bool((scenes >= (1 << 15)).any())
        or bool((queries < 0).any())
        or bool((queries >= MAX_QUERY_ID_EXCLUSIVE).any())
        or bool((regions < 0).any())
        or bool((regions >= MAX_REGION_ID_EXCLUSIVE).any())
        or int(torch.unique(scenes).numel()) < 2
    ):
        raise ValueError("v3 scene/query/region axes differ")
    scene_axis = torch.unique(scenes, sorted=True)
    if not torch.equal(scene_axis, torch.arange(scene_axis.numel())):
        raise ValueError("v3 scene indices must be contiguous")
    return scenes.contiguous(), queries.contiguous(), regions.contiguous()


def packed_scene_query_region_groups(
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
) -> torch.Tensor:
    scenes, queries, regions = _validated_axes(
        scene_indices, query_indices, region_indices
    )
    return ((scenes << 48) | (queries << 32) | regions).contiguous()


def packed_scene_region_groups(
    scene_indices: torch.Tensor,
    region_indices: torch.Tensor,
) -> torch.Tensor:
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    regions = torch.as_tensor(region_indices).detach().long().cpu().reshape(-1)
    if (
        scenes.shape != regions.shape
        or scenes.numel() <= 0
        or bool((scenes < 0).any())
        or bool((scenes >= (1 << 31)).any())
        or bool((regions < 0).any())
        or bool((regions >= MAX_REGION_ID_EXCLUSIVE).any())
    ):
        raise ValueError("v3 scene/region fold axes differ")
    return ((scenes << 32) | regions).contiguous()


def multiscene_query_region_fold_ids(
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
) -> torch.Tensor:
    scenes, queries, regions = _validated_axes(
        scene_indices, query_indices, region_indices
    )
    del queries
    packed = packed_scene_region_groups(scenes, regions)
    unique, inverse = torch.unique(packed, sorted=True, return_inverse=True)
    group_folds = deterministic_group_folds(unique, num_folds=OOF_FOLDS)
    return group_folds[inverse].contiguous()


def scene_query_region_balanced_weights(
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
) -> torch.Tensor:
    """Equal scene, then query, then region mass; equal rows within region."""

    scenes, queries, regions = _validated_axes(
        scene_indices, query_indices, region_indices
    )
    packed = packed_scene_query_region_groups(scenes, queries, regions)
    unique_groups, inverse, counts = torch.unique(
        packed, sorted=True, return_inverse=True, return_counts=True
    )
    group_scenes = unique_groups >> 48
    group_queries = (unique_groups >> 32) & (MAX_QUERY_ID_EXCLUSIVE - 1)
    scene_count = int(torch.unique(scenes).numel())
    query_key = group_scenes * MAX_QUERY_ID_EXCLUSIVE + group_queries
    unique_queries, query_inverse = torch.unique(
        query_key, sorted=True, return_inverse=True
    )
    query_scenes = unique_queries // MAX_QUERY_ID_EXCLUSIVE
    queries_per_scene = torch.bincount(
        query_scenes, minlength=scene_count
    ).double()
    regions_per_query = torch.bincount(
        query_inverse, minlength=unique_queries.numel()
    ).double()
    if bool((queries_per_scene <= 0).any()) or bool((regions_per_query <= 0).any()):
        raise ValueError("v3 balance population is empty")
    group_mass = (
        queries_per_scene[group_scenes]
        * regions_per_query[query_inverse]
    ).reciprocal()
    per_row = group_mass[inverse] / counts[inverse].double()
    return (per_row / per_row.mean()).contiguous()


def _robust_location_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    location = values.median(dim=0).values
    scale = 1.4826 * (values - location).abs().median(dim=0).values
    varying = values.amax(dim=0) > values.amin(dim=0)
    scale[(scale <= 0.0) & varying] = 1.0
    scale[scale <= 0.0] = 1.0
    return location.contiguous(), scale.contiguous()


def fit_query_balanced_monotone_additive_logistic(
    oriented_features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> MonotoneAdditiveLogistic:
    values = torch.as_tensor(oriented_features).detach().double().cpu()
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes, queries, regions = _validated_axes(
        scene_indices, query_indices, region_indices, rows=values.shape[0]
    )
    if (
        values.ndim != 2
        or values.shape[1] != len(SELECTOR_FEATURE_NAMES)
        or target.shape != (values.shape[0],)
        or not bool(torch.isfinite(values).all())
        or int(target.sum()) <= 0
        or int((~target).sum()) <= 0
        or not math.isfinite(float(l2_strength))
        or float(l2_strength) <= 0.0
        or int(maximum_iterations) <= 0
    ):
        raise ValueError("v3 monotone-logistic inputs differ")
    location, scale = _robust_location_scale(values)
    normalized = (values - location) / scale
    sample_weight = scene_query_region_balanced_weights(scenes, queries, regions)
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
    if not bool(torch.isfinite(positive_weight).all()) or not bool(
        torch.isfinite(bias.detach())
    ):
        raise RuntimeError("v3 monotone-logistic fit is invalid")
    return MonotoneAdditiveLogistic(
        location=location.float().contiguous(),
        scale=scale.float().contiguous(),
        positive_weights=positive_weight.float().contiguous(),
        bias=bias.detach().float().reshape(()).contiguous(),
    )


def fit_source_multiscene_query_balanced_selector_oof(
    unit_features: torch.Tensor,
    labels: torch.Tensor,
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    region_indices: torch.Tensor,
    *,
    l2_strength: float = 0.01,
    maximum_iterations: int = 100,
) -> SourceMultisceneQueryBalancedSelectorFit:
    oriented = oriented_selector_features(unit_features)
    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    scenes, queries, regions = _validated_axes(
        scene_indices, query_indices, region_indices, rows=oriented.shape[0]
    )
    folds = multiscene_query_region_fold_ids(scenes, queries, regions)
    prediction = torch.full((oriented.shape[0],), float("nan"))
    models = []
    scene_axis = torch.unique(scenes, sorted=True)
    for fold in range(OOF_FOLDS):
        heldout = folds == fold
        training = ~heldout
        if (
            int(heldout.sum()) <= 0
            or int(training.sum()) <= 0
            or not torch.equal(torch.unique(scenes[training], sorted=True), scene_axis)
            or not torch.equal(torch.unique(scenes[heldout], sorted=True), scene_axis)
            or int(target[training].sum()) <= 0
            or int((~target[training]).sum()) <= 0
        ):
            raise ValueError(f"v3 OOF fold {fold} differs")
        model = fit_query_balanced_monotone_additive_logistic(
            oriented[training],
            target[training],
            scenes[training],
            queries[training],
            regions[training],
            l2_strength=l2_strength,
            maximum_iterations=maximum_iterations,
        )
        prediction[heldout] = selector_probability(model, oriented[heldout])
        models.append(model)
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("v3 OOF selector left invalid predictions")
    return SourceMultisceneQueryBalancedSelectorFit(
        fold_ids=folds,
        fold_models=tuple(models),
        oof_probability=prediction.contiguous(),
    )


def _lower_tail_cvar(values: torch.Tensor, fraction: float) -> float:
    vector = torch.as_tensor(values).detach().double().cpu().reshape(-1)
    if vector.numel() <= 0 or not 0.0 < float(fraction) <= 1.0:
        raise ValueError("v3 CVaR inputs differ")
    count = max(1, int(math.ceil(float(fraction) * vector.numel())))
    return float(torch.sort(vector).values[:count].mean())


def evaluate_query_utility_gate(
    *,
    selected: torch.Tensor,
    signed_utility: torch.Tensor,
    query_indices: torch.Tensor,
    minimum_candidate_units_per_query: int = 64,
    minimum_selected_units_per_query: int = 16,
    minimum_evaluable_query_fraction: float = 0.80,
    minimum_evaluable_queries: int = 8,
    lower_tail_fraction: float = 0.20,
) -> tuple[dict[str, object], dict[str, bool]]:
    chosen = torch.as_tensor(selected).detach().bool().cpu().reshape(-1)
    utility = torch.as_tensor(signed_utility).detach().float().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    if (
        chosen.shape != utility.shape
        or chosen.shape != queries.shape
        or chosen.numel() <= 0
        or not bool(torch.isfinite(utility).all())
        or bool((queries < 0).any())
    ):
        raise ValueError("v3 query utility axes differ")
    rows = []
    sufficient = 0
    for query in torch.unique(queries, sorted=True).tolist():
        mask = queries == int(query)
        candidates = int(mask.sum())
        if candidates < int(minimum_candidate_units_per_query):
            continue
        sufficient += 1
        selected_count = int((mask & chosen).sum())
        if selected_count < int(minimum_selected_units_per_query):
            continue
        unconditional = float(utility[mask].mean())
        selected_mean = float(utility[mask & chosen].mean())
        rows.append(
            {
                "query_index": int(query),
                "candidate_units": candidates,
                "selected_units": selected_count,
                "selected_utility_mean": selected_mean,
                "unconditional_utility_mean": unconditional,
                "utility_gain": selected_mean - unconditional,
            }
        )
    required_evaluable = max(
        int(minimum_evaluable_queries),
        int(math.ceil(float(minimum_evaluable_query_fraction) * sufficient)),
    )
    selected_utility = torch.tensor(
        [float(row["selected_utility_mean"]) for row in rows], dtype=torch.float64
    )
    gain = torch.tensor(
        [float(row["utility_gain"]) for row in rows], dtype=torch.float64
    )
    enough = len(rows) >= required_evaluable and sufficient >= minimum_evaluable_queries
    outcomes = {
        "candidate_sufficient_queries": sufficient,
        "required_evaluable_queries": required_evaluable,
        "evaluable_queries": len(rows),
        "evaluable_query_fraction": len(rows) / max(sufficient, 1),
        "query_macro_selected_utility_mean": (
            float(selected_utility.mean()) if rows else float("-inf")
        ),
        "query_macro_unconditional_utility_mean": (
            float(
                torch.tensor(
                    [float(row["unconditional_utility_mean"]) for row in rows],
                    dtype=torch.float64,
                ).mean()
            )
            if rows
            else float("inf")
        ),
        "query_macro_utility_gain": float(gain.mean()) if rows else float("-inf"),
        "minimum_selected_query_utility": (
            float(selected_utility.min()) if rows else float("-inf")
        ),
        "lower_tail_20pct_selected_utility_CVaR": (
            _lower_tail_cvar(selected_utility, lower_tail_fraction)
            if rows
            else float("-inf")
        ),
        "lower_tail_20pct_utility_gain_CVaR": (
            _lower_tail_cvar(gain, lower_tail_fraction)
            if rows
            else float("-inf")
        ),
        "per_query": rows,
    }
    checks = {
        "evaluable_query_coverage_at_least_minimum": enough,
        "every_evaluable_query_selected_utility_nonnegative": bool(
            rows and (selected_utility >= 0.0).all()
        ),
        "query_macro_selected_utility_strictly_above_unconditional": bool(
            rows and float(gain.mean()) > 0.0
        ),
        "lower_tail_20pct_selected_utility_CVaR_nonnegative": bool(
            rows and _lower_tail_cvar(selected_utility, lower_tail_fraction) >= 0.0
        ),
        "lower_tail_20pct_utility_gain_CVaR_nonnegative": bool(
            rows and _lower_tail_cvar(gain, lower_tail_fraction) >= 0.0
        ),
    }
    checks["passed"] = all(checks.values())
    return outcomes, checks


def select_largest_query_safe_oof_threshold(
    probability: torch.Tensor,
    hard_labels: torch.Tensor,
    signed_utility: torch.Tensor,
    scene_indices: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    minimum_selected_per_scene: int = 256,
    minimum_rejected_per_scene: int = 256,
    maximum_selected_fraction_per_scene: float = 0.90,
    minimum_overall_wilson_lower: float = 0.80,
    minimum_scene_wilson_lower: float = 0.75,
    minimum_candidate_units_per_query: int = 64,
    minimum_selected_units_per_query: int = 16,
    minimum_evaluable_query_fraction: float = 0.80,
    minimum_evaluable_queries: int = 8,
    lower_tail_fraction: float = 0.20,
    require_lower_tail_utility_gain_cvar: bool = True,
    allow_no_threshold_for_source_diagnostic: bool = False,
) -> dict[str, object]:
    score = torch.as_tensor(probability).detach().float().cpu().reshape(-1)
    labels = torch.as_tensor(hard_labels).detach().bool().cpu().reshape(-1)
    utility = torch.as_tensor(signed_utility).detach().float().cpu().reshape(-1)
    scenes = torch.as_tensor(scene_indices).detach().long().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).detach().long().cpu().reshape(-1)
    if (
        labels.shape != score.shape
        or utility.shape != score.shape
        or scenes.shape != score.shape
        or queries.shape != score.shape
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(utility).all())
        or bool(((score < 0.0) | (score > 1.0)).any())
        or not 0.0 < float(maximum_selected_fraction_per_scene) < 1.0
    ):
        raise ValueError("v3 threshold axes differ")
    scene_axis = torch.unique(scenes, sorted=True)
    if not torch.equal(scene_axis, torch.arange(scene_axis.numel())):
        raise ValueError("v3 threshold scene axis differs")
    scene_count = int(scene_axis.numel())
    query_key = scenes * MAX_QUERY_ID_EXCLUSIVE + queries
    unique_query_key, query_inverse = torch.unique(
        query_key, sorted=True, return_inverse=True
    )
    group_count = int(unique_query_key.numel())
    group_scene = unique_query_key // MAX_QUERY_ID_EXCLUSIVE
    group_query = unique_query_key % MAX_QUERY_ID_EXCLUSIVE
    candidate_count = torch.bincount(query_inverse, minlength=group_count).long()
    unconditional_sum = torch.bincount(
        query_inverse, weights=utility.double(), minlength=group_count
    )
    selected_count = torch.zeros(group_count, dtype=torch.long)
    selected_utility_sum = torch.zeros(group_count, dtype=torch.float64)
    scene_population = torch.bincount(scenes, minlength=scene_count).long()
    scene_selected = torch.zeros(scene_count, dtype=torch.long)
    scene_positive = torch.zeros(scene_count, dtype=torch.long)

    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    _, tie_counts = torch.unique_consecutive(sorted_score, return_counts=True)
    stops = tie_counts.cumsum(0)
    start = 0
    overall_positive = 0
    best: dict[str, object] | None = None
    diagnostics: dict[str, object] = {
        "tie_complete_thresholds": int(stops.numel()),
        "overall_precision_gate_passed_thresholds": 0,
        "all_scene_basic_gates_passed_thresholds": 0,
        "all_query_gates_passed_thresholds": 0,
        "query_gate_failure_threshold_counts": {
            "evaluable_query_coverage_at_least_minimum": 0,
            "every_evaluable_query_selected_utility_nonnegative": 0,
            "query_macro_selected_utility_strictly_above_unconditional": 0,
            "lower_tail_20pct_selected_utility_CVaR_nonnegative": 0,
            "lower_tail_20pct_utility_gain_CVaR_nonnegative": 0,
        },
        "lower_tail_utility_gain_CVaR_required": bool(
            require_lower_tail_utility_gain_cvar
        ),
        "best_minimum_across_scenes": {
            "evaluable_query_fraction": 0.0,
            "minimum_selected_query_utility": float("-inf"),
            "query_macro_utility_gain": float("-inf"),
            "lower_tail_20pct_selected_utility_CVaR": float("-inf"),
            "lower_tail_20pct_utility_gain_CVaR": float("-inf"),
        },
        "minimum_failed_required_query_gates": 5,
        "thresholds_at_minimum_failed_required_query_gates": 0,
        "failure_gate_counts_at_minimum": {
            "evaluable_query_coverage_at_least_minimum": 0,
            "every_evaluable_query_selected_utility_nonnegative": 0,
            "query_macro_selected_utility_strictly_above_unconditional": 0,
            "lower_tail_20pct_selected_utility_CVaR_nonnegative": 0,
        },
        "closest_threshold_example": None,
    }

    def query_gate_from_aggregates(scene: int) -> tuple[dict[str, object], dict[str, bool]]:
        group_mask = group_scene == int(scene)
        sufficient = group_mask & (
            candidate_count >= int(minimum_candidate_units_per_query)
        )
        evaluable = sufficient & (
            selected_count >= int(minimum_selected_units_per_query)
        )
        sufficient_count = int(sufficient.sum())
        evaluable_count = int(evaluable.sum())
        required = max(
            int(minimum_evaluable_queries),
            int(math.ceil(float(minimum_evaluable_query_fraction) * sufficient_count)),
        )
        chosen_utility = (
            selected_utility_sum[evaluable]
            / selected_count[evaluable].double().clamp_min(1)
        )
        unconditional_utility = (
            unconditional_sum[evaluable]
            / candidate_count[evaluable].double().clamp_min(1)
        )
        gain = chosen_utility - unconditional_utility
        enough = (
            sufficient_count >= int(minimum_evaluable_queries)
            and evaluable_count >= required
        )
        selected_cvar = (
            _lower_tail_cvar(chosen_utility, lower_tail_fraction)
            if evaluable_count
            else float("-inf")
        )
        gain_cvar = (
            _lower_tail_cvar(gain, lower_tail_fraction)
            if evaluable_count
            else float("-inf")
        )
        outcomes = {
            "candidate_sufficient_queries": sufficient_count,
            "required_evaluable_queries": required,
            "evaluable_queries": evaluable_count,
            "evaluable_query_fraction": evaluable_count / max(sufficient_count, 1),
            "query_macro_selected_utility_mean": (
                float(chosen_utility.mean()) if evaluable_count else float("-inf")
            ),
            "query_macro_unconditional_utility_mean": (
                float(unconditional_utility.mean()) if evaluable_count else float("inf")
            ),
            "query_macro_utility_gain": (
                float(gain.mean()) if evaluable_count else float("-inf")
            ),
            "minimum_selected_query_utility": (
                float(chosen_utility.min()) if evaluable_count else float("-inf")
            ),
            "lower_tail_20pct_selected_utility_CVaR": selected_cvar,
            "lower_tail_20pct_utility_gain_CVaR": gain_cvar,
            "per_query": [],
        }
        checks = {
            "evaluable_query_coverage_at_least_minimum": enough,
            "every_evaluable_query_selected_utility_nonnegative": bool(
                evaluable_count and (chosen_utility >= 0.0).all()
            ),
            "query_macro_selected_utility_strictly_above_unconditional": bool(
                evaluable_count and float(gain.mean()) > 0.0
            ),
            "lower_tail_20pct_selected_utility_CVaR_nonnegative": bool(
                evaluable_count and selected_cvar >= 0.0
            ),
            "lower_tail_20pct_utility_gain_CVaR_nonnegative": bool(
                evaluable_count and gain_cvar >= 0.0
            ),
        }
        required_checks = {
            key: value
            for key, value in checks.items()
            if require_lower_tail_utility_gain_cvar
            or key != "lower_tail_20pct_utility_gain_CVaR_nonnegative"
        }
        checks["passed"] = all(required_checks.values())
        return outcomes, checks

    for stop in stops.tolist():
        chunk = order[start:int(stop)]
        chunk_group = query_inverse[chunk]
        selected_count += torch.bincount(chunk_group, minlength=group_count)
        selected_utility_sum += torch.bincount(
            chunk_group, weights=utility[chunk].double(), minlength=group_count
        )
        scene_selected += torch.bincount(scenes[chunk], minlength=scene_count)
        scene_positive += torch.bincount(
            scenes[chunk][labels[chunk]], minlength=scene_count
        )
        overall_positive += int(labels[chunk].sum())
        start = int(stop)
        total = int(stop)
        overall_negative = total - overall_positive
        if (
            overall_positive <= 0
            or overall_negative <= 0
            or one_sided_wilson_lower(overall_positive, total)
            < float(minimum_overall_wilson_lower)
        ):
            continue
        diagnostics["overall_precision_gate_passed_thresholds"] = int(
            diagnostics["overall_precision_gate_passed_thresholds"]
        ) + 1
        per_scene = []
        passed = True
        all_scene_basic = True
        threshold_query_failures: set[str] = set()
        for scene in range(scene_count):
            selected_scene = int(scene_selected[scene])
            rejected_scene = int(scene_population[scene]) - selected_scene
            positive_scene = int(scene_positive[scene])
            negative_scene = selected_scene - positive_scene
            basic = (
                selected_scene >= int(minimum_selected_per_scene)
                and rejected_scene >= int(minimum_rejected_per_scene)
                and selected_scene / int(scene_population[scene])
                <= float(maximum_selected_fraction_per_scene)
                and positive_scene > 0
                and negative_scene > 0
                and one_sided_wilson_lower(positive_scene, selected_scene)
                >= float(minimum_scene_wilson_lower)
            )
            if basic:
                query_outcomes, query_checks = query_gate_from_aggregates(scene)
                threshold_query_failures.update(
                    key
                    for key, value in query_checks.items()
                    if key != "passed" and value is not True
                )
            else:
                query_outcomes, query_checks = {}, {"passed": False}
                all_scene_basic = False
            passed = passed and basic and query_checks["passed"]
            per_scene.append(
                {
                    "scene_index": scene,
                    "selected": selected_scene,
                    "rejected": rejected_scene,
                    "coverage_fraction": selected_scene / int(scene_population[scene]),
                    "hard_positive": positive_scene,
                    "hard_negative": negative_scene,
                    "hard_precision": positive_scene / max(selected_scene, 1),
                    "hard_precision_wilson95_lower": (
                        one_sided_wilson_lower(positive_scene, selected_scene)
                        if selected_scene
                        else 0.0
                    ),
                    "query_utility": query_outcomes,
                    "query_gate": query_checks,
                }
            )
        if all_scene_basic:
            diagnostics["all_scene_basic_gates_passed_thresholds"] = int(
                diagnostics["all_scene_basic_gates_passed_thresholds"]
            ) + 1
            failure_counts = diagnostics["query_gate_failure_threshold_counts"]
            assert isinstance(failure_counts, dict)
            for key in threshold_query_failures:
                failure_counts[key] = int(failure_counts[key]) + 1
            best_minimum = diagnostics["best_minimum_across_scenes"]
            assert isinstance(best_minimum, dict)
            outcome_names = tuple(best_minimum)
            for name in outcome_names:
                observed = min(
                    float(row["query_utility"][name]) for row in per_scene
                )
                best_minimum[name] = max(float(best_minimum[name]), observed)
            required_failure_set = {
                key
                for key in threshold_query_failures
                if require_lower_tail_utility_gain_cvar
                or key != "lower_tail_20pct_utility_gain_CVaR_nonnegative"
            }
            failure_count = len(required_failure_set)
            current_minimum = int(
                diagnostics["minimum_failed_required_query_gates"]
            )
            if failure_count < current_minimum:
                diagnostics["minimum_failed_required_query_gates"] = failure_count
                diagnostics[
                    "thresholds_at_minimum_failed_required_query_gates"
                ] = 0
                diagnostics["failure_gate_counts_at_minimum"] = {
                    key: 0
                    for key in diagnostics["failure_gate_counts_at_minimum"]
                }
                diagnostics["closest_threshold_example"] = {
                    "threshold_inclusive": float(sorted_score[int(stop) - 1]),
                    "failed_required_query_gates": sorted(required_failure_set),
                    "minimum_across_scenes_evaluable_query_fraction": min(
                        float(row["query_utility"]["evaluable_query_fraction"])
                        for row in per_scene
                    ),
                    "per_scene": per_scene,
                }
                current_minimum = failure_count
            if failure_count == current_minimum:
                diagnostics[
                    "thresholds_at_minimum_failed_required_query_gates"
                ] = int(
                    diagnostics[
                        "thresholds_at_minimum_failed_required_query_gates"
                    ]
                ) + 1
                minimum_counts = diagnostics["failure_gate_counts_at_minimum"]
                assert isinstance(minimum_counts, dict)
                for key in required_failure_set:
                    minimum_counts[key] = int(minimum_counts[key]) + 1
                closest = diagnostics["closest_threshold_example"]
                minimum_fraction = min(
                    float(row["query_utility"]["evaluable_query_fraction"])
                    for row in per_scene
                )
                if (
                    required_failure_set
                    == {"evaluable_query_coverage_at_least_minimum"}
                    and isinstance(closest, dict)
                    and minimum_fraction
                    > float(
                        closest.get(
                            "minimum_across_scenes_evaluable_query_fraction", -1.0
                        )
                    )
                ):
                    diagnostics["closest_threshold_example"] = {
                        "threshold_inclusive": float(sorted_score[int(stop) - 1]),
                        "failed_required_query_gates": sorted(required_failure_set),
                        "minimum_across_scenes_evaluable_query_fraction": minimum_fraction,
                        "per_scene": per_scene,
                    }
            if not threshold_query_failures or (
                not require_lower_tail_utility_gain_cvar
                and threshold_query_failures
                == {"lower_tail_20pct_utility_gain_CVaR_nonnegative"}
            ):
                diagnostics["all_query_gates_passed_thresholds"] = int(
                    diagnostics["all_query_gates_passed_thresholds"]
                ) + 1
        if passed:
            best = {
                "threshold_inclusive": float(sorted_score[int(stop) - 1]),
                "selected": total,
                "hard_positive": overall_positive,
                "hard_negative": overall_negative,
                "hard_precision": overall_positive / total,
                "hard_precision_wilson95_lower": one_sided_wilson_lower(
                    overall_positive, total
                ),
                "coverage_fraction": total / score.numel(),
                "per_scene": per_scene,
            }
    diagnostics["feasible_threshold_exists"] = best is not None
    if best is None and allow_no_threshold_for_source_diagnostic:
        return {"selected_threshold": None, "scan_diagnostics": diagnostics}
    if best is None:
        raise RuntimeError("no v3 query-safe threshold passes every source gate")
    final_selected = score >= float(best["threshold_inclusive"])
    for scene_row in best["per_scene"]:
        scene = int(scene_row["scene_index"])
        scene_mask = scenes == scene
        outcomes, checks = evaluate_query_utility_gate(
            selected=final_selected[scene_mask],
            signed_utility=utility[scene_mask],
            query_indices=queries[scene_mask],
            minimum_candidate_units_per_query=minimum_candidate_units_per_query,
            minimum_selected_units_per_query=minimum_selected_units_per_query,
            minimum_evaluable_query_fraction=minimum_evaluable_query_fraction,
            minimum_evaluable_queries=minimum_evaluable_queries,
            lower_tail_fraction=lower_tail_fraction,
        )
        if not require_lower_tail_utility_gain_cvar:
            checks["passed"] = all(
                value
                for key, value in checks.items()
                if key not in {"passed", "lower_tail_20pct_utility_gain_CVaR_nonnegative"}
            )
        scene_row["query_utility"] = outcomes
        scene_row["query_gate"] = checks
    best["scan_diagnostics"] = diagnostics
    return best


def validate_query_balanced_selector_model_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("v3 selector model must be a mapping")
    required = {
        "schema", "schema_version", "feature_names", "source_unit_feature_indices",
        "fold_models", "threshold_inclusive", "target_probability",
        "execution_authority", "training_provenance",
    }
    if set(value) != required:
        raise ValueError("v3 selector model fields differ")
    threshold = float(value.get("threshold_inclusive", float("nan")))
    if (
        value.get("schema") != MODEL_SCHEMA
        or value.get("schema_version") != 3
        or value.get("feature_names") != list(SELECTOR_FEATURE_NAMES)
        or value.get("source_unit_feature_indices") != list(SOURCE_UNIT_FEATURE_INDICES)
        or value.get("target_probability")
        != "minimum_probability_across_three_fold_models"
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("v3 selector model header differs")
    rows = value.get("fold_models")
    if not isinstance(rows, list) or len(rows) != OOF_FOLDS:
        raise ValueError("v3 selector fold-model axis differs")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "location", "scale", "positive_weights", "bias"
        }:
            raise ValueError("v3 selector fold-model fields differ")
        model = MonotoneAdditiveLogistic(
            location=torch.as_tensor(row["location"]).detach().float().cpu(),
            scale=torch.as_tensor(row["scale"]).detach().float().cpu(),
            positive_weights=torch.as_tensor(row["positive_weights"]).detach().float().cpu(),
            bias=torch.as_tensor(row["bias"]).detach().float().cpu().reshape(()),
        )
        selector_probability(model, torch.zeros(1, len(SELECTOR_FEATURE_NAMES)))
    provenance = value.get("training_provenance")
    if not isinstance(provenance, dict) or provenance != {
        "source_scene_count": 2,
        "scene_identifier_used_for_balancing_and_groups_only": True,
        "query_identifier_used_for_balancing_and_threshold_gate_only": True,
        "query_identifier_used_as_feature": False,
        "scene_identifier_used_as_feature": False,
    }:
        raise ValueError("v3 selector provenance differs")
    return dict(value)


__all__ = [
    "MODEL_SCHEMA", "SourceMultisceneQueryBalancedSelectorFit",
    "evaluate_query_utility_gate", "fit_query_balanced_monotone_additive_logistic",
    "fit_source_multiscene_query_balanced_selector_oof",
    "multiscene_query_region_fold_ids", "packed_scene_query_region_groups",
    "packed_scene_region_groups",
    "scene_query_region_balanced_weights", "select_largest_query_safe_oof_threshold",
    "validate_query_balanced_selector_model_payload",
]
