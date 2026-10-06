"""Pure mesh-domain metrics for ScanNet-UQIS predictions.

All vertices have equal weight.  Binary selection uses the frozen ``>= 0.5``
threshold.  Average precision and oracle IoU operate on complete score-tie
groups so results cannot depend on mesh vertex ordering.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FIXED_THRESHOLD = 0.5


def scene_clustered_bootstrap(
    scene_values: Mapping[str, float],
    *,
    samples: int = 2000,
    seed: int = 20260813,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap a scene-macro mean by resampling whole scenes."""

    ordered = np.asarray(
        [float(scene_values[key]) for key in sorted(scene_values)], dtype=np.float64
    )
    if not len(ordered) or not np.isfinite(ordered).all():
        raise ValueError("scene bootstrap requires finite values from at least one scene")
    if int(samples) != samples or int(samples) <= 0:
        raise ValueError("bootstrap samples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(ordered), size=(int(samples), len(ordered)))
    estimates = ordered[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    return {
        "estimate": float(ordered.mean()),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "bootstrap_unit": "scene",
        "seed": int(seed),
    }


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 0.0


def _score_group_ends(sorted_scores: np.ndarray) -> np.ndarray:
    if not len(sorted_scores):
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )


def average_precision(probabilities: np.ndarray, target: np.ndarray) -> float:
    """Return non-interpolated, tie-aware binary average precision."""

    scores = np.asarray(probabilities, dtype=np.float64)
    positives = np.asarray(target, dtype=bool)
    positive_count = int(positives.sum())
    if scores.ndim != 1 or positives.shape != scores.shape or not positive_count:
        raise ValueError("average precision requires aligned scores and positives")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = positives[order]
    cumulative_true = np.cumsum(sorted_positive, dtype=np.int64)
    ends = _score_group_ends(sorted_scores)
    true_at_end = cumulative_true[ends]
    previous_true = np.r_[0, true_at_end[:-1]]
    recall_increment = (true_at_end - previous_true) / positive_count
    precision = true_at_end / (ends + 1)
    return float(np.sum(recall_increment * precision))


def oracle_threshold_iou(probabilities: np.ndarray, target: np.ndarray) -> float:
    """Return the best IoU attainable by a global ``score >= threshold`` cut."""

    scores = np.asarray(probabilities, dtype=np.float64)
    positives = np.asarray(target, dtype=bool)
    positive_count = int(positives.sum())
    if scores.ndim != 1 or positives.shape != scores.shape or not positive_count:
        raise ValueError("oracle IoU requires aligned scores and positives")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    cumulative_true = np.cumsum(positives[order], dtype=np.int64)
    ends = _score_group_ends(sorted_scores)
    selected = ends + 1
    true_positive = cumulative_true[ends]
    union = positive_count + selected - true_positive
    return float(np.max(true_positive / union))


def evaluate_query_probabilities(
    probabilities: np.ndarray,
    *,
    target_instance_id: int,
    same_class_distractor_instance_ids: Sequence[int],
    mesh_instance_ids: np.ndarray,
    mesh_xyz: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one query's probabilities on its official ScanNet mesh."""

    scores = np.asarray(probabilities)
    instance_ids = np.asarray(mesh_instance_ids)
    xyz = np.asarray(mesh_xyz, dtype=np.float64)
    if scores.ndim != 1 or instance_ids.ndim != 1:
        raise ValueError("probabilities and mesh instance ids must be 1-D")
    if scores.shape != instance_ids.shape or xyz.shape != (scores.size, 3):
        raise ValueError("prediction and ScanNet mesh domain do not align")
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise ValueError("probabilities must be real numeric values")
    if not np.isfinite(scores).all() or not np.isfinite(xyz).all():
        raise ValueError("prediction and mesh coordinates must be finite")
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("probabilities must lie in [0, 1]")

    target_id = int(target_instance_id)
    target = instance_ids == target_id
    if not bool(target.any()):
        raise ValueError(f"target instance {target_id} has no mesh vertices")
    distractor_ids = tuple(int(value) for value in same_class_distractor_instance_ids)
    if target_id in distractor_ids or len(set(distractor_ids)) != len(distractor_ids):
        raise ValueError("same-class distractor ids must be unique and exclude target")
    missing_distractors = [
        value for value in distractor_ids if not bool((instance_ids == value).any())
    ]
    if missing_distractors:
        raise ValueError(f"same-class distractors have no mesh vertices: {missing_distractors}")

    selected = np.asarray(scores >= FIXED_THRESHOLD, dtype=bool)
    intersection = int(np.logical_and(selected, target).sum())
    fixed_iou = _iou(selected, target)
    selected_count = int(selected.sum())
    target_count = int(target.sum())
    distractor_iou = (
        max(_iou(selected, instance_ids == value) for value in distractor_ids)
        if distractor_ids
        else None
    )

    target_centroid = xyz[target].mean(axis=0)
    if selected_count:
        selected_centroid = xyz[selected].mean(axis=0)
        centroid_error = float(np.linalg.norm(selected_centroid - target_centroid))
    else:
        centroid_error = float(
            np.linalg.norm(xyz - target_centroid[None, :], axis=1).max()
        )

    return {
        "average_precision": average_precision(scores, target),
        "oracle_iou": oracle_threshold_iou(scores, target),
        "fixed_iou_0.5": fixed_iou,
        "acc_at_iou_0.25": float(fixed_iou >= 0.25),
        "acc_at_iou_0.50": float(fixed_iou >= 0.50),
        "selected_purity": float(intersection / selected_count) if selected_count else 0.0,
        "positive_coverage": float(intersection / target_count),
        "same_class_distractor_iou": distractor_iou,
        "centroid_error_m": centroid_error,
        "empty_prediction": not bool(selected_count),
    }
