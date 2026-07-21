"""Track B: image-to-3-D full-instance selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _scene_macro(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["scene_id"])].append(float(row[key]))
    return _mean([float(np.mean(values)) for values in groups.values()])


def evaluate_instance_selection(
    query_records: Sequence[Mapping[str, Any]],
    masks_by_query: Mapping[str, np.ndarray],
    mesh_instance_ids_by_scene: Mapping[str, np.ndarray],
    mesh_xyz_by_scene: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate frozen-threshold/solver masks in the official mesh domain."""

    rows: list[dict[str, Any]] = []
    for source in query_records:
        query_id, scene_id = str(source["query_id"]), str(source["scene_id"])
        prediction = np.asarray(masks_by_query[query_id], dtype=bool).reshape(-1)
        instance_ids = np.asarray(mesh_instance_ids_by_scene[scene_id]).reshape(-1)
        xyz = np.asarray(mesh_xyz_by_scene[scene_id], dtype=np.float64)
        if prediction.shape != instance_ids.shape or xyz.shape != (instance_ids.size, 3):
            raise ValueError(f"{query_id}: prediction/ScanNet mesh mismatch")
        target_id = int(source["instance_id_3d"])
        target = instance_ids == target_id
        iou = _iou(prediction, target)
        target_centroid = xyz[target].mean(axis=0)
        if bool(prediction.any()):
            predicted_centroid = xyz[prediction].mean(axis=0)
            centroid_error = float(np.linalg.norm(predicted_centroid - target_centroid))
            empty = False
        else:
            # Deterministic scene-scale penalty prevents an empty prediction
            # from disappearing from the centroid metric.
            centroid_error = float(
                np.linalg.norm(xyz - target_centroid[None], axis=1).max()
            )
            empty = True
        class_ids = {
            int(key): int(value)
            for key, value in source["candidate_instance_class_ids"].items()
        }
        class_id = int(source["nyu40_class_id"])
        distractors = [
            instance_id
            for instance_id in map(int, source["candidate_instance_ids_3d"])
            if instance_id != target_id and class_ids[instance_id] == class_id
        ]
        if distractors:
            target_overlap = int(np.logical_and(prediction, target).sum())
            maximum_distractor_overlap = max(
                int(np.logical_and(prediction, instance_ids == value).sum())
                for value in distractors
            )
            distractor_success: bool | None = target_overlap > maximum_distractor_overlap
        else:
            distractor_success = None
        rows.append(
            {
                "query_id": query_id,
                "scene_id": scene_id,
                "iou": iou,
                "acc_iou_0.15": float(iou >= 0.15),
                "acc_iou_0.25": float(iou >= 0.25),
                "acc_iou_0.50": float(iou >= 0.50),
                "centroid_error_m": centroid_error,
                "empty_prediction": empty,
                "same_category_distractor_success": distractor_success,
            }
        )
    distractor = [
        float(row["same_category_distractor_success"])
        for row in rows
        if row["same_category_distractor_success"] is not None
    ]
    metrics = {
        "instance_macro_3d_miou": _mean([row["iou"] for row in rows]),
        "acc_at_iou_0.15": _mean([row["acc_iou_0.15"] for row in rows]),
        "acc_at_iou_0.25": _mean([row["acc_iou_0.25"] for row in rows]),
        "acc_at_iou_0.50": _mean([row["acc_iou_0.50"] for row in rows]),
        "centroid_error_m": _mean([row["centroid_error_m"] for row in rows]),
        "same_category_distractor_success": _mean(distractor),
        "same_category_query_count": len(distractor),
        "empty_prediction_rate": _mean(
            [float(row["empty_prediction"]) for row in rows]
        ),
    }
    return {
        "track": "B_image_to_3d_instance_selection",
        "query_count": len(rows),
        "query_micro": metrics,
        "scene_macro": {
            key: _scene_macro(rows, row_key)
            for key, row_key in (
                ("3d_miou", "iou"),
                ("acc_at_iou_0.15", "acc_iou_0.15"),
                ("acc_at_iou_0.25", "acc_iou_0.25"),
                ("acc_at_iou_0.50", "acc_iou_0.50"),
                ("centroid_error_m", "centroid_error_m"),
            )
        },
        "per_query": rows,
    }

