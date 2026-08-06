"""Source-only rank-gauge calibration for strict SPIn reference-mask queries.

The v1 cross-fit experiment established that the *absolute* propagated score
gauge changes substantially when a complete source-footprint fold is removed.
This module removes that nuisance degree of freedom with one fixed,
parameter-free map: each fold is evaluated in the weighted empirical-CDF
gauge induced only by its own training-source reference scores.

The implementation is deliberately independent from the stopped v1
visibility-calibration module.  It may consume the three sealed v1 matched-OOF
folds as immutable evidence authorities, but it neither edits nor extends any
v1 artifact.  No target raster, RGB value, feature, mask, metric, or score
distribution is used to fit an empirical CDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    cap_positive_reference_evidence,
    knn_feature_distances,
    normalize_node_features,
    rbf_similarity_from_distances,
    run_query_conditioned_diffusion,
    weighted_logistic_query_compatibility,
)


METHOD = "source_footprint_crossfit_quantile_calibration_v2"
INPUT_MATCHED_OOF_ARTIFACT_TYPE = (
    "spin_source_footprint_crossfit_matched_oof_fold_v1"
)
QUANTILE_OOF_ARTIFACT_TYPE = "spin_source_footprint_quantile_oof_calibration_v2"
FULL_FIT_GAUGE_ARTIFACT_TYPE = "spin_source_footprint_full_fit_quantile_gauge_v2"
NUM_FOLDS = 3
FEATURE_BANDWIDTH = 0.5
REGULARIZER_BANDWIDTH = 1.0
LOGISTIC_C = 0.01
ITERATIONS = 100
EDGE_BINARIZE_THRESHOLD = 1e-5
MAX_POSITIVE_FRACTION = 0.1
MAX_FOLD_QUANTILE_THRESHOLD_SPAN = 0.10


def quantile_threshold_grid() -> tuple[float, ...]:
    """Return the single frozen descending quantile grid 0.99,...,0.03."""

    return tuple(float(value) for value in np.arange(0.99, 0.02, -0.01))


def quantile_method_contract() -> dict[str, object]:
    """Return the machine-readable, non-sweepable v2 contract."""

    return {
        "method": METHOD,
        "input_fold_artifact_type": INPUT_MATCHED_OOF_ARTIFACT_TYPE,
        "num_folds": NUM_FOLDS,
        "score_source": "frozen_matched_k201_propagated_probability",
        "cdf_population": "training_source_rows_with_reference_weight_positive",
        "cdf_weight": "exact_training_source_reference_weight",
        "cdf_definition": "weighted_right_continuous_empirical_cdf",
        "cdf_formula": "sum_i(w_i*1[p_i<=s])/sum_i(w_i)",
        "cdf_tie_semantics": "right_searchsorted_includes_equal_score_mass",
        "cdf_smoothing": False,
        "cdf_temperature": None,
        "cdf_bandwidth": None,
        "official_num_neighbors": 200,
        "effective_knn_columns": 201,
        "feature_bandwidth": FEATURE_BANDWIDTH,
        "regularizer_bandwidth": REGULARIZER_BANDWIDTH,
        "logistic_c": LOGISTIC_C,
        "logistic_fit_population": "all_nodes_positive_only",
        "iterations": ITERATIONS,
        "edge_binarize_threshold": EDGE_BINARIZE_THRESHOLD,
        "max_positive_fraction": MAX_POSITIVE_FRACTION,
        "positive_cap_rule": "released_argsort_keep_largest_int_fraction",
        "threshold_grid": list(quantile_threshold_grid()),
        "threshold_tie_break": "descending_grid_strict_improvement_first_maximizer",
        "maximum_fold_quantile_threshold_span": (
            MAX_FOLD_QUANTILE_THRESHOLD_SPAN
        ),
        "target_distribution_used": False,
        "target_rgb_used": False,
        "target_mask_used": False,
        "target_metric_used": False,
        "parameter_scan": False,
    }


def _finite_probability(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().double().cpu().reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name} must lie in [0,1]")
    return tensor


def _finite_nonnegative(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().double().cpu().reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    return tensor


@dataclass(frozen=True)
class WeightedRightECDF:
    """Exact weighted right-continuous empirical CDF.

    ``support`` contains unique, increasing score values and ``cumulative``
    contains the cumulative source-reference mass at each support value.
    Scores with zero weight are excluded and therefore cannot change the map.
    """

    support: torch.Tensor
    cumulative: torch.Tensor
    total_weight: float
    source_rows: int

    @classmethod
    def fit(cls, score: torch.Tensor, weight: torch.Tensor) -> "WeightedRightECDF":
        probability = _finite_probability(score, name="CDF score")
        mass = _finite_nonnegative(weight, name="CDF weight")
        if probability.shape != mass.shape:
            raise ValueError("CDF score and weight must be aligned")
        positive = mass > 0
        if not bool(positive.any()):
            raise ValueError("CDF requires positive source-reference mass")
        probability = probability[positive]
        mass = mass[positive]
        order = torch.argsort(probability, stable=True)
        sorted_score = probability[order]
        sorted_mass = mass[order]
        support, inverse = torch.unique_consecutive(
            sorted_score, return_inverse=True
        )
        support_mass = torch.zeros_like(support)
        support_mass.scatter_add_(0, inverse, sorted_mass)
        cumulative = torch.cumsum(support_mass, dim=0)
        total = float(cumulative[-1])
        if not np.isfinite(total) or total <= 0:
            raise ValueError("CDF total source-reference mass must be positive")
        return cls(
            support=support.contiguous(),
            cumulative=cumulative.contiguous(),
            total_weight=total,
            source_rows=int(positive.sum()),
        )

    def __post_init__(self) -> None:
        support = torch.as_tensor(self.support).detach().double().cpu().reshape(-1)
        cumulative = (
            torch.as_tensor(self.cumulative).detach().double().cpu().reshape(-1)
        )
        if support.numel() == 0 or support.shape != cumulative.shape:
            raise ValueError("CDF support and cumulative mass must align")
        if not bool(torch.isfinite(support).all()) or bool(
            ((support < 0) | (support > 1)).any()
        ):
            raise ValueError("CDF support must be finite and in [0,1]")
        if support.numel() > 1 and not bool((support[1:] > support[:-1]).all()):
            raise ValueError("CDF support must be strictly increasing")
        if not bool(torch.isfinite(cumulative).all()) or bool(
            (cumulative <= 0).any()
        ):
            raise ValueError("CDF cumulative mass must be finite and positive")
        if cumulative.numel() > 1 and not bool(
            (cumulative[1:] > cumulative[:-1]).all()
        ):
            raise ValueError("CDF cumulative mass must be strictly increasing")
        if not np.isfinite(self.total_weight) or self.total_weight <= 0:
            raise ValueError("CDF total weight must be positive")
        tolerance = max(1e-12, abs(self.total_weight) * 1e-12)
        if abs(float(cumulative[-1]) - float(self.total_weight)) > tolerance:
            raise ValueError("CDF total weight differs from cumulative mass")
        if int(self.source_rows) <= 0:
            raise ValueError("CDF source row count must be positive")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "cumulative", cumulative)

    def map(self, score: torch.Tensor | float) -> torch.Tensor:
        """Map scores to ``F(s)=P_source(score <= s)`` exactly."""

        query = torch.as_tensor(score).detach().double().cpu()
        if not bool(torch.isfinite(query).all()) or bool(
            ((query < 0) | (query > 1)).any()
        ):
            raise ValueError("CDF query score must be finite and in [0,1]")
        flat = query.reshape(-1)
        insertion = torch.searchsorted(self.support, flat, right=True)
        mapped = torch.zeros_like(flat)
        found = insertion > 0
        mapped[found] = self.cumulative[insertion[found] - 1] / self.total_weight
        return mapped.reshape(query.shape)


@dataclass(frozen=True)
class QuantileThresholdSelection:
    threshold: float
    weighted_soft_iou: float
    positive_mass: float
    selected_negative_mass: float
    eligible_rows: int


@dataclass(frozen=True)
class QuantileFoldDiagnostic:
    fold: int
    threshold: float
    weighted_soft_iou: float
    positive_quantile_mean: float
    negative_quantile_mean: float
    training_source_rows: int
    training_source_weight: float
    training_zero_score_weight_fraction: float
    ecdf: WeightedRightECDF


@dataclass(frozen=True)
class QuantileOOFCalibration:
    t_completion_quantile: float
    pooled_weighted_soft_iou: float
    fold_diagnostics: tuple[
        QuantileFoldDiagnostic,
        QuantileFoldDiagnostic,
        QuantileFoldDiagnostic,
    ]
    threshold_span: float
    stable: bool
    source_visible: torch.Tensor
    pooled_oof_quantile: torch.Tensor
    pooled_oof_eligible: torch.Tensor


@dataclass(frozen=True)
class FullFitQuantileGauge:
    probability: torch.Tensor
    query_compatibility: torch.Tensor
    capped_positive_weight: torch.Tensor
    ecdf: WeightedRightECDF
    t_seen_raw: float
    t_seen_quantile: float


@dataclass(frozen=True)
class QuantilePredictionFields:
    """One target raster's source-gauge score, threshold, and margin."""

    score_quantile: torch.Tensor
    spatial_threshold_quantile: torch.Tensor
    continuous_margin: torch.Tensor
    low_resolution_prediction: torch.Tensor


def select_responsibility_weighted_quantile_threshold(
    quantile: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    eligible: torch.Tensor,
    *,
    thresholds: Sequence[float] | None = None,
) -> QuantileThresholdSelection:
    """Select one completion threshold in the common empirical-CDF gauge."""

    score = _finite_probability(quantile, name="completion quantile")
    positive = _finite_nonnegative(positive_weight, name="positive weight")
    negative = _finite_nonnegative(negative_weight, name="negative weight")
    use = torch.as_tensor(eligible).detach().bool().cpu().reshape(-1)
    if any(value.shape != score.shape for value in (positive, negative, use)):
        raise ValueError("quantile threshold inputs must be aligned")
    if not bool(use.any()):
        raise ValueError("completion threshold has no eligible held-out rows")
    positive_mass = float(positive[use].sum())
    negative_mass = float(negative[use].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("completion threshold requires both responsibility classes")
    grid = tuple(quantile_threshold_grid() if thresholds is None else thresholds)
    if not grid or any(
        not np.isfinite(value) or not 0 < float(value) < 1 for value in grid
    ):
        raise ValueError("quantile threshold grid must lie in (0,1)")
    if any(float(left) <= float(right) for left, right in zip(grid, grid[1:])):
        raise ValueError("quantile threshold grid must be strictly descending")

    best_iou = -1.0
    best_threshold: float | None = None
    best_selected_negative = 0.0
    for raw_threshold in grid:
        threshold = float(raw_threshold)
        selected = use & (score >= threshold)
        intersection = float(positive[selected].sum())
        selected_negative = float(negative[selected].sum())
        union = positive_mass + selected_negative
        weighted_iou = intersection / union if union > 0 else 1.0
        if weighted_iou > best_iou:
            best_iou = weighted_iou
            best_threshold = threshold
            best_selected_negative = selected_negative
    if best_threshold is None:
        raise RuntimeError("quantile completion-threshold selection failed")
    return QuantileThresholdSelection(
        threshold=best_threshold,
        weighted_soft_iou=float(best_iou),
        positive_mass=positive_mass,
        selected_negative_mass=best_selected_negative,
        eligible_rows=int(use.sum()),
    )


def _weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    eligible: torch.Tensor,
) -> float:
    use_weight = weight[eligible]
    total = float(use_weight.sum())
    if total <= 0:
        raise ValueError("weighted diagnostic has no class mass")
    return float((value[eligible] * use_weight).sum() / total)


def build_quantile_oof_calibration(
    folds: Mapping[int, Mapping[str, object]],
) -> QuantileOOFCalibration:
    """Normalize three sealed OOF folds and test source-only gauge stability."""

    if set(folds) != set(range(NUM_FOLDS)):
        raise ValueError("quantile OOF requires exactly folds 0,1,2")
    reference = folds[0]
    required_scalars = (
        "scene_id",
        "protocol_hash",
        "capability_cache_sha256",
        "support_graph_sha256",
        "source_evidence_authority_sha256",
        "source_footprint_fold_authority_sha256",
        "query_diffusion_knn_sha256",
        "query_diffusion_relation_sha256",
        "method_contract_sha256",
    )
    required_tensors = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "heldout",
        "reference_weight",
        "population_positive_weight",
        "population_negative_weight",
        "matched_query_diffusion_probability",
        "training_positive_weight",
        "training_reference_weight",
    )
    normalized: dict[int, dict[str, torch.Tensor]] = {}
    for fold, payload in folds.items():
        if payload.get("artifact_type") != INPUT_MATCHED_OOF_ARTIFACT_TYPE:
            raise ValueError("unexpected matched OOF artifact type")
        if int(payload.get("heldout_fold", -1)) != fold:
            raise ValueError("matched OOF heldout-fold identity differs")
        if any(payload.get(key) != reference.get(key) for key in required_scalars):
            raise ValueError("matched OOF authority differs across folds")
        if any(
            payload.get(key) is not False
            for key in (
                "target_rgb_opened",
                "target_mask_opened",
                "target_metric_computed",
            )
        ):
            raise ValueError("matched OOF payload violates target-blind safety")
        missing = [key for key in required_tensors if key not in payload]
        if missing:
            raise ValueError(f"matched OOF payload lacks tensors: {missing}")
        tensors = {
            key: torch.as_tensor(payload[key]).detach().cpu()
            for key in required_tensors
        }
        shape = tensors["valid"].shape
        if tensors["valid"].dtype != torch.bool or tensors["valid"].ndim != 1:
            raise ValueError("matched OOF valid authority is malformed")
        for key in required_tensors:
            if key != "global_rows" and tensors[key].shape != shape:
                raise ValueError("matched OOF tensors do not share the global domain")
        if not torch.equal(
            tensors["global_rows"].long(), torch.where(tensors["valid"])[0]
        ):
            raise ValueError("matched OOF global rows differ from valid rows")
        expected_heldout = tensors["valid"].bool() & (
            tensors["fold_ids"].long() == fold
        )
        if not torch.equal(tensors["heldout"].bool(), expected_heldout):
            raise ValueError("matched OOF heldout rows differ from footprint folds")
        heldout = tensors["heldout"].bool()
        if float(tensors["training_positive_weight"][heldout].sum()) != 0.0 or float(
            tensors["training_reference_weight"][heldout].sum()
        ) != 0.0:
            raise ValueError("matched OOF held-out evidence was not cleared")
        normalized[fold] = tensors

    invariant = (
        "valid",
        "global_rows",
        "fold_ids",
        "observed",
        "reference_weight",
        "population_positive_weight",
        "population_negative_weight",
    )
    for fold in (1, 2):
        for key in invariant:
            if not torch.equal(normalized[fold][key], normalized[0][key]):
                raise ValueError(f"matched OOF invariant tensor differs: {key}")

    shape = normalized[0]["valid"].shape
    pooled_quantile = torch.zeros(shape, dtype=torch.float64)
    pooled_eligible = torch.zeros(shape, dtype=torch.bool)
    diagnostics: list[QuantileFoldDiagnostic] = []
    for fold in range(NUM_FOLDS):
        tensors = normalized[fold]
        probability = _finite_probability(
            tensors["matched_query_diffusion_probability"],
            name=f"fold {fold} matched probability",
        )
        training_reference = _finite_nonnegative(
            tensors["training_reference_weight"],
            name=f"fold {fold} training reference weight",
        )
        cdf = WeightedRightECDF.fit(probability, training_reference)
        quantile = cdf.map(probability)
        eligible = (
            tensors["valid"].bool()
            & tensors["observed"].bool()
            & tensors["heldout"].bool()
        )
        if bool((pooled_eligible & eligible).any()):
            raise ValueError("matched OOF held-out populations overlap")
        selection = select_responsibility_weighted_quantile_threshold(
            quantile,
            tensors["population_positive_weight"],
            tensors["population_negative_weight"],
            eligible,
        )
        positive = _finite_nonnegative(
            tensors["population_positive_weight"], name="positive weight"
        )
        negative = _finite_nonnegative(
            tensors["population_negative_weight"], name="negative weight"
        )
        training_use = training_reference > 0
        zero_score_weight = float(
            training_reference[training_use & (probability == 0)].sum()
        )
        diagnostics.append(
            QuantileFoldDiagnostic(
                fold=fold,
                threshold=selection.threshold,
                weighted_soft_iou=selection.weighted_soft_iou,
                positive_quantile_mean=_weighted_mean(
                    quantile, positive, eligible
                ),
                negative_quantile_mean=_weighted_mean(
                    quantile, negative, eligible
                ),
                training_source_rows=cdf.source_rows,
                training_source_weight=cdf.total_weight,
                training_zero_score_weight_fraction=(
                    zero_score_weight / cdf.total_weight
                ),
                ecdf=cdf,
            )
        )
        pooled_quantile[eligible] = quantile[eligible]
        pooled_eligible |= eligible

    pooled = select_responsibility_weighted_quantile_threshold(
        pooled_quantile,
        normalized[0]["population_positive_weight"],
        normalized[0]["population_negative_weight"],
        pooled_eligible,
    )
    thresholds = tuple(item.threshold for item in diagnostics)
    span = float(max(thresholds) - min(thresholds))
    source_visible = normalized[0]["valid"].bool() & (
        normalized[0]["reference_weight"].float() > 0
    )
    return QuantileOOFCalibration(
        t_completion_quantile=pooled.threshold,
        pooled_weighted_soft_iou=pooled.weighted_soft_iou,
        fold_diagnostics=(diagnostics[0], diagnostics[1], diagnostics[2]),
        threshold_span=span,
        stable=span <= MAX_FOLD_QUANTILE_THRESHOLD_SPAN + 1e-12,
        source_visible=source_visible,
        pooled_oof_quantile=pooled_quantile,
        pooled_oof_eligible=pooled_eligible,
    )


def compute_full_fit_quantile_gauge(
    relation_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    positive_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    t_seen_raw: float,
    device: str | torch.device = "cpu",
) -> FullFitQuantileGauge:
    """Fit the frozen K201 support and its source-reference ECDF.

    This is the only full-fit construction.  It uses the complete immutable
    source evidence; unlike an OOF fold, no row is cleared.  The ECDF is then
    fit to full-fit primitive probabilities using only full source-reference
    weights, and the already sealed raw seen threshold is mapped by that same
    ECDF.
    """

    features = torch.as_tensor(relation_features).detach().cpu()
    neighbors = torch.as_tensor(neighbor_indices).detach().long().cpu()
    positive = _finite_nonnegative(positive_weight, name="positive weight").float()
    reference = _finite_nonnegative(reference_weight, name="reference weight").float()
    count = int(features.shape[0]) if features.ndim == 2 else -1
    if count <= 1 or neighbors.ndim != 2 or neighbors.shape != (count, 201):
        raise ValueError("full-fit quantile gauge requires aligned K201 rows")
    if positive.shape != (count,) or reference.shape != (count,):
        raise ValueError("full-fit evidence must align with relation rows")
    if not features.is_floating_point():
        raise ValueError("relation features must be floating point")
    for start in range(0, count, 4096):
        if not bool(torch.isfinite(features[start : start + 4096]).all()):
            raise ValueError("relation features contain NaN or infinity")
    if neighbors.numel() and (
        int(neighbors.min()) < 0 or int(neighbors.max()) >= count
    ):
        raise ValueError("full-fit KNN index is outside relation rows")
    seen = float(t_seen_raw)
    if not np.isfinite(seen) or not 0 < seen < 1:
        raise ValueError("raw seen threshold must lie in (0,1)")
    capped_positive = cap_positive_reference_evidence(
        positive,
        max_positive_fraction=MAX_POSITIVE_FRACTION,
    )
    if int((capped_positive > 0).sum()) == 0 or float(reference.sum()) <= 0:
        raise ValueError("full-fit source evidence is class-degenerate")

    normalized_cpu = normalize_node_features(features)
    compatibility_cpu = weighted_logistic_query_compatibility(
        normalized_cpu,
        capped_positive,
        reference,
        logistic_c=LOGISTIC_C,
        regularizer_bandwidth=REGULARIZER_BANDWIDTH,
        fit_population="all_nodes_positive_only",
    )
    compute_device = torch.device(device)
    normalized = normalized_cpu.to(compute_device)
    neighbors_device = neighbors.to(compute_device)
    compatibility = compatibility_cpu.to(compute_device)
    distances = knn_feature_distances(
        normalized,
        neighbors_device,
        distance_chunk_size=64,
    )
    similarities = rbf_similarity_from_distances(
        distances,
        feature_bandwidth=FEATURE_BANDWIDTH,
        positive_reference_mask=capped_positive.to(compute_device) > 0,
    )
    support = run_query_conditioned_diffusion(
        capped_positive.to(compute_device),
        neighbors_device,
        similarities,
        compatibility,
        config=QueryConditionedDiffusionConfig(
            kernel="ludvig_release_compat",
            feature_bandwidth=FEATURE_BANDWIDTH,
            regularizer_bandwidth=REGULARIZER_BANDWIDTH,
            logistic_c=LOGISTIC_C,
            logistic_fit_population="all_nodes_positive_only",
            iterations=ITERATIONS,
            edge_binarize_threshold=EDGE_BINARIZE_THRESHOLD,
            distance_chunk_size=64,
        ),
    ).squeeze(1)
    probability = support.detach().float().cpu()
    compatibility_out = compatibility.detach().float().cpu()
    if probability.shape != (count,) or not bool(torch.isfinite(probability).all()):
        raise RuntimeError("full-fit K201 support is malformed")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise RuntimeError("full-fit K201 support lies outside [0,1]")
    ecdf = WeightedRightECDF.fit(probability, reference)
    seen_quantile = float(ecdf.map(seen))
    return FullFitQuantileGauge(
        probability=probability,
        query_compatibility=compatibility_out,
        capped_positive_weight=capped_positive.detach().float().cpu(),
        ecdf=ecdf,
        t_seen_raw=seen,
        t_seen_quantile=seen_quantile,
    )


def map_target_score_to_source_quantile(
    continuous_target_score: torch.Tensor,
    full_fit_source_ecdf: WeightedRightECDF,
) -> torch.Tensor:
    """Apply a pre-fit source-only gauge to a sealed continuous target score."""

    return full_fit_source_ecdf.map(continuous_target_score)


def quantile_visibility_adaptive_threshold(
    source_visible_coverage: torch.Tensor,
    *,
    t_seen_quantile: float,
    t_completion_quantile: float,
) -> torch.Tensor:
    """Interpolate the two fixed operating points only in the common gauge."""

    coverage = torch.as_tensor(source_visible_coverage)
    if not coverage.is_floating_point():
        coverage = coverage.float()
    if not bool(torch.isfinite(coverage).all()) or bool(
        ((coverage < 0) | (coverage > 1)).any()
    ):
        raise ValueError("source-visible coverage must be finite and in [0,1]")
    seen = float(t_seen_quantile)
    completion = float(t_completion_quantile)
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in (seen, completion)):
        raise ValueError("quantile thresholds must lie in [0,1]")
    return coverage * seen + (1.0 - coverage) * completion


def build_quantile_prediction_fields(
    continuous_target_score: torch.Tensor,
    source_visible_coverage: torch.Tensor,
    full_fit_source_ecdf: WeightedRightECDF,
    *,
    t_seen_quantile: float,
    t_completion_quantile: float,
) -> QuantilePredictionFields:
    """Build the sealed continuous-margin representation for one target view.

    The benchmark's frozen scoring adapter bilinearly resizes a continuous
    raster before applying its comparison.  With a spatial threshold the exact
    analogous representation is the continuous margin ``F(score)-t(x)``:
    bilinear interpolation commutes with subtraction, so resizing this margin
    and comparing it to zero preserves the frozen evaluation order without
    introducing a binary-mask resize.
    """

    score = torch.as_tensor(continuous_target_score).detach().double().cpu()
    coverage = torch.as_tensor(source_visible_coverage).detach().double().cpu()
    if score.shape != coverage.shape or score.numel() == 0:
        raise ValueError("target score and source-visible coverage must align")
    score_quantile = full_fit_source_ecdf.map(score)
    threshold = quantile_visibility_adaptive_threshold(
        coverage,
        t_seen_quantile=t_seen_quantile,
        t_completion_quantile=t_completion_quantile,
    ).double()
    margin = score_quantile - threshold
    if not bool(torch.isfinite(margin).all()):
        raise ValueError("quantile target margin must be finite")
    return QuantilePredictionFields(
        score_quantile=score_quantile,
        spatial_threshold_quantile=threshold,
        continuous_margin=margin,
        low_resolution_prediction=margin >= 0,
    )
