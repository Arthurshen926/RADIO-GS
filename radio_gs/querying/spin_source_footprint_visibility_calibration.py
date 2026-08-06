"""Source-only visibility calibration for the strict SPIn reference-mask task.

The contract deliberately separates three quantities:

* the full-reference operating point for source-visible primitives;
* a leak-free completion operating point reconstructed from three complete
  source-footprint holdouts; and
* query-independent source-visibility coverage rendered into a target pose.

No target raster, RGB value, feature, mask, or metric is an input to this
module.  The fixed query-diffusion configuration is the already selected Lego
development operating point and is not exposed as a sweepable argument.
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


METHOD = "source_footprint_crossfit_visibility_calibration_v1"
MATCHED_OOF_ARTIFACT_TYPE = "spin_source_footprint_crossfit_matched_oof_fold_v1"
CALIBRATION_ARTIFACT_TYPE = "spin_source_footprint_visibility_calibration_v1"
NUM_FOLDS = 3
FEATURE_BANDWIDTH = 0.5
REGULARIZER_BANDWIDTH = 1.0
LOGISTIC_C = 0.01
ITERATIONS = 100
EDGE_BINARIZE_THRESHOLD = 1e-5
MAX_POSITIVE_FRACTION = 0.1
MAX_FOLD_THRESHOLD_SPAN = 0.10


def reference_threshold_grid() -> tuple[float, ...]:
    """Return the release-compatible descending 0.99,...,0.03 grid."""

    return tuple(float(value) for value in np.arange(0.99, 0.02, -0.01))


def matched_oof_method_contract() -> dict[str, object]:
    """Machine-readable, non-sweepable matched-interface OOF contract."""

    return {
        "method": METHOD,
        "fold_assignment": "source_raster_dominant_footprint_blocks_v1_splitmix64_mod3",
        "num_folds": NUM_FOLDS,
        "base_stage": "surface_safe_k16_propagated",
        "query_diffusion_kernel": "ludvig_release_compat",
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
        "readout": "k201_propagated_all_components",
        "connected_selection": False,
        "threshold_grid": list(reference_threshold_grid()),
        "target_rgb_used": False,
        "target_mask_used": False,
        "target_metric_used": False,
        "parameter_scan": False,
    }


@dataclass(frozen=True)
class MatchedOOFSupport:
    """One fold's matched K201 support and leak audit tensors."""

    probability: torch.Tensor
    query_compatibility: torch.Tensor
    training_positive_weight: torch.Tensor
    training_reference_weight: torch.Tensor
    heldout: torch.Tensor


@dataclass(frozen=True)
class ThresholdSelection:
    """One deterministic responsibility-weighted threshold selection."""

    threshold: float
    weighted_soft_iou: float
    positive_mass: float
    selected_negative_mass: float
    eligible_rows: int


@dataclass(frozen=True)
class CrossfitCalibration:
    """Pooled source-only completion calibration and stability audit."""

    t_completion: float
    pooled_weighted_soft_iou: float
    fold_thresholds: tuple[float, float, float]
    fold_weighted_soft_iou: tuple[float, float, float]
    threshold_span: float
    stable: bool
    source_visible: torch.Tensor
    pooled_probability: torch.Tensor
    pooled_eligible: torch.Tensor


def _finite_nonnegative_vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError(f"{name} must be a finite non-negative vector")
    return tensor


def compute_matched_oof_support(
    relation_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
    positive_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    heldout: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> MatchedOOFSupport:
    """Run the frozen matched K201 interface with one footprint fold removed.

    Held-out rows are assigned exact zero positive evidence *and* exact zero
    logistic sample weight.  The latter is essential: the release-compatible
    classifier treats zero-positive rows with non-zero weight as negatives.
    Leaving the sample weight alive would therefore leak held-out background
    and relabel held-out foreground as background.
    """

    features = torch.as_tensor(relation_features).detach().cpu()
    neighbors = torch.as_tensor(neighbor_indices).detach().long().cpu()
    positive = _finite_nonnegative_vector(positive_weight, name="positive_weight")
    reference = _finite_nonnegative_vector(
        reference_weight, name="reference_weight"
    )
    heldout_cpu = torch.as_tensor(heldout).detach().bool().cpu().reshape(-1)
    count = int(features.shape[0]) if features.ndim == 2 else -1
    if count <= 1 or neighbors.ndim != 2 or neighbors.shape[0] != count:
        raise ValueError("relation features and KNN rows must be aligned and non-empty")
    if positive.shape != (count,) or reference.shape != (count,) or heldout_cpu.shape != (
        count,
    ):
        raise ValueError("matched OOF evidence must align with relation rows")
    if not features.is_floating_point():
        raise ValueError("relation features must be floating point")
    for start in range(0, count, 4096):
        if not bool(torch.isfinite(features[start : start + 4096]).all()):
            raise ValueError("relation features contain NaN or infinity")
    if neighbors.numel() and (
        int(neighbors.min()) < 0 or int(neighbors.max()) >= count
    ):
        raise ValueError("matched OOF KNN index is outside the relation rows")
    if neighbors.shape[1] != 201:
        raise ValueError("matched SPIn OOF requires the frozen effective K201 cache")
    if not bool(heldout_cpu.any()) or bool(heldout_cpu.all()):
        raise ValueError("matched OOF heldout must be a strict non-empty subset")

    training_positive = positive.clone()
    training_reference = reference.clone()
    training_positive[heldout_cpu] = 0.0
    training_reference[heldout_cpu] = 0.0
    if float(training_positive[heldout_cpu].sum()) != 0.0 or float(
        training_reference[heldout_cpu].sum()
    ) != 0.0:
        raise RuntimeError("held-out source evidence survived matched OOF clearing")
    training_positive = cap_positive_reference_evidence(
        training_positive,
        max_positive_fraction=MAX_POSITIVE_FRACTION,
    )
    if int((training_positive > 0).sum()) == 0:
        raise ValueError("matched OOF training fold has no positive evidence")
    if float(training_reference.sum()) <= 0:
        raise ValueError("matched OOF training fold has no reference weight")

    normalized_cpu = normalize_node_features(features)
    compatibility_cpu = weighted_logistic_query_compatibility(
        normalized_cpu,
        training_positive,
        training_reference,
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
        positive_reference_mask=training_positive.to(compute_device) > 0,
    )
    support = run_query_conditioned_diffusion(
        training_positive.to(compute_device),
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
        raise RuntimeError("matched OOF support is malformed")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise RuntimeError("matched OOF support is outside [0,1]")
    return MatchedOOFSupport(
        probability=probability,
        query_compatibility=compatibility_out,
        training_positive_weight=training_positive,
        training_reference_weight=training_reference,
        heldout=heldout_cpu,
    )


def select_responsibility_weighted_threshold(
    probability: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    eligible: torch.Tensor,
    *,
    thresholds: Sequence[float] | None = None,
) -> ThresholdSelection:
    """Select the descending-grid threshold by responsibility-weighted IoU."""

    score = torch.as_tensor(probability).detach().double().cpu().reshape(-1)
    positive = torch.as_tensor(positive_weight).detach().double().cpu().reshape(-1)
    negative = torch.as_tensor(negative_weight).detach().double().cpu().reshape(-1)
    use = torch.as_tensor(eligible).detach().bool().cpu().reshape(-1)
    if score.numel() == 0 or any(
        value.shape != score.shape for value in (positive, negative, use)
    ):
        raise ValueError("threshold inputs must be non-empty aligned vectors")
    if not bool(torch.isfinite(score).all()) or bool(
        ((score < 0) | (score > 1)).any()
    ):
        raise ValueError("completion probability must be finite and in [0,1]")
    if any(
        not bool(torch.isfinite(value).all()) or bool((value < 0).any())
        for value in (positive, negative)
    ):
        raise ValueError("responsibility weights must be finite and non-negative")
    if not bool(use.any()):
        raise ValueError("completion threshold has no eligible held-out rows")
    positive_mass = float(positive[use].sum())
    negative_mass = float(negative[use].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("completion threshold requires both responsibility classes")
    grid = tuple(reference_threshold_grid() if thresholds is None else thresholds)
    if not grid or any(
        not np.isfinite(value) or not 0 < float(value) < 1 for value in grid
    ):
        raise ValueError("threshold grid must contain finite values in (0,1)")
    if any(float(left) <= float(right) for left, right in zip(grid, grid[1:])):
        raise ValueError("threshold grid must be strictly descending")

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
        # Strict improvement preserves the first descending-grid maximizer.
        if weighted_iou > best_iou:
            best_iou = weighted_iou
            best_threshold = threshold
            best_selected_negative = selected_negative
    if best_threshold is None:
        raise RuntimeError("completion threshold selection failed")
    return ThresholdSelection(
        threshold=best_threshold,
        weighted_soft_iou=float(best_iou),
        positive_mass=positive_mass,
        selected_negative_mass=best_selected_negative,
        eligible_rows=int(use.sum()),
    )


def build_crossfit_calibration(
    folds: Mapping[int, Mapping[str, object]],
) -> CrossfitCalibration:
    """Validate exactly three matched folds and pool held-out predictions."""

    if set(folds) != set(range(NUM_FOLDS)):
        raise ValueError("crossfit calibration requires exactly folds 0,1,2")
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
    )
    normalized: dict[int, dict[str, torch.Tensor]] = {}
    for fold, payload in folds.items():
        if payload.get("artifact_type") != MATCHED_OOF_ARTIFACT_TYPE:
            raise ValueError("unexpected matched OOF artifact type")
        if int(payload.get("heldout_fold", -1)) != fold:
            raise ValueError("matched OOF heldout-fold identity differs")
        if any(payload.get(key) != reference.get(key) for key in required_scalars):
            raise ValueError("matched OOF authority differs across folds")
        if any(payload.get(key) is not False for key in (
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_computed",
        )):
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
        rows = tensors["global_rows"].long()
        if not torch.equal(rows, torch.where(tensors["valid"])[0]):
            raise ValueError("matched OOF global rows differ from valid rows")
        heldout = tensors["heldout"].bool()
        expected_heldout = tensors["valid"].bool() & (
            tensors["fold_ids"].long() == fold
        )
        if not torch.equal(heldout, expected_heldout):
            raise ValueError("matched OOF heldout rows differ from footprint folds")
        if float(
            torch.as_tensor(payload.get("heldout_training_positive_weight_sum", -1.0))
        ) != 0.0 or float(
            torch.as_tensor(payload.get("heldout_training_reference_weight_sum", -1.0))
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
    pooled_probability = torch.zeros(shape, dtype=torch.float32)
    pooled_eligible = torch.zeros(shape, dtype=torch.bool)
    fold_selections: list[ThresholdSelection] = []
    for fold in range(NUM_FOLDS):
        tensors = normalized[fold]
        eligible = (
            tensors["valid"].bool()
            & tensors["observed"].bool()
            & tensors["heldout"].bool()
        )
        if bool((pooled_eligible & eligible).any()):
            raise ValueError("matched OOF held-out populations overlap")
        probability = tensors["matched_query_diffusion_probability"].float()
        selection = select_responsibility_weighted_threshold(
            probability,
            tensors["population_positive_weight"],
            tensors["population_negative_weight"],
            eligible,
        )
        fold_selections.append(selection)
        pooled_probability[eligible] = probability[eligible]
        pooled_eligible |= eligible

    pooled = select_responsibility_weighted_threshold(
        pooled_probability,
        normalized[0]["population_positive_weight"],
        normalized[0]["population_negative_weight"],
        pooled_eligible,
    )
    fold_thresholds = tuple(item.threshold for item in fold_selections)
    threshold_span = float(max(fold_thresholds) - min(fold_thresholds))
    source_visible = normalized[0]["valid"].bool() & (
        normalized[0]["reference_weight"].float() > 0
    )
    return CrossfitCalibration(
        t_completion=pooled.threshold,
        pooled_weighted_soft_iou=pooled.weighted_soft_iou,
        fold_thresholds=(
            float(fold_thresholds[0]),
            float(fold_thresholds[1]),
            float(fold_thresholds[2]),
        ),
        fold_weighted_soft_iou=(
            float(fold_selections[0].weighted_soft_iou),
            float(fold_selections[1].weighted_soft_iou),
            float(fold_selections[2].weighted_soft_iou),
        ),
        threshold_span=threshold_span,
        stable=threshold_span <= MAX_FOLD_THRESHOLD_SPAN + 1e-12,
        source_visible=source_visible,
        pooled_probability=pooled_probability,
        pooled_eligible=pooled_eligible,
    )


def visibility_adaptive_threshold(
    source_visible_coverage: torch.Tensor,
    *,
    t_seen: float,
    t_completion: float,
) -> torch.Tensor:
    """Return ``c*t_seen + (1-c)*t_completion`` without tunable exponents."""

    coverage = torch.as_tensor(source_visible_coverage)
    if not coverage.is_floating_point():
        coverage = coverage.float()
    if not bool(torch.isfinite(coverage).all()) or bool(
        ((coverage < 0) | (coverage > 1)).any()
    ):
        raise ValueError("source-visible coverage must be finite and in [0,1]")
    seen = float(t_seen)
    completion = float(t_completion)
    if any(not np.isfinite(value) or not 0 < value < 1 for value in (seen, completion)):
        raise ValueError("seen and completion thresholds must be in (0,1)")
    return coverage * seen + (1.0 - coverage) * completion


def visibility_calibrated_prediction(
    continuous_score: torch.Tensor,
    source_visible_coverage: torch.Tensor,
    *,
    t_seen: float,
    t_completion: float,
) -> torch.Tensor:
    """Threshold a sealed continuous score with the source-only spatial rule."""

    score = torch.as_tensor(continuous_score)
    coverage = torch.as_tensor(source_visible_coverage, device=score.device)
    if score.shape != coverage.shape or not bool(torch.isfinite(score).all()):
        raise ValueError("continuous score and visibility coverage must align and be finite")
    threshold = visibility_adaptive_threshold(
        coverage,
        t_seen=t_seen,
        t_completion=t_completion,
    ).to(score.device)
    return score >= threshold
