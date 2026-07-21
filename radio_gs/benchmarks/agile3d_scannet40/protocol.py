"""Frozen AGILE3D ScanNet40 single-object interaction protocol.

This module reproduces the data and metric behavior of AGILE3D's released
``eval_single_obj.py`` without importing MinkowskiEngine.  A method callback
only sees the 5 cm quantized point cloud and the clicks accumulated so far.
Ground truth is opened by the simulator solely to place the next corrective
click and to compute metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


IOU_CLICK_COUNTS = (1, 2, 3, 5, 10, 15)
NOC_THRESHOLDS = (0.50, 0.65, 0.80, 0.85, 0.90)


@dataclass(frozen=True)
class Agile3DObject:
    scene_id: str
    object_id: int
    semantic_class: str

    @property
    def key(self) -> str:
        return f"{self.scene_id}_obj_{self.object_id}"


@dataclass(frozen=True)
class Click:
    point_index: int
    is_positive: bool
    order: int


@dataclass(frozen=True)
class QuantizedScene:
    coordinates: np.ndarray
    raw_coordinates: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    inverse_map: np.ndarray
    unique_map: np.ndarray


def load_official_object_list(root: str | Path) -> list[Agile3DObject]:
    root = Path(root)
    object_ids = np.load(root / "single" / "object_ids.npy", allow_pickle=False)
    classes = np.loadtxt(root / "single" / "object_classes.txt", dtype=str)
    if object_ids.ndim != 2 or object_ids.shape[1] != 2:
        raise ValueError("official object_ids.npy must be [N,2]")
    if classes.shape != (object_ids.shape[0],):
        raise ValueError("official object class list does not align")
    return [
        Agile3DObject(str(scene), int(object_id), str(semantic_class))
        for (scene, object_id), semantic_class in zip(object_ids, classes)
    ]


def quantize_scannet_points(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    *,
    voxel_size: float = 0.05,
) -> QuantizedScene:
    """Match AGILE3D's shifted 5 cm sparse quantization contract.

    MinkowskiEngine's CPU coordinate map retains the first input row for each
    occupied voxel and orders voxels by that first occurrence.  We reproduce
    both maps explicitly so click-index tie breaks also match the release.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    if colors.shape[0] != xyz.shape[0] or labels.shape != (xyz.shape[0],):
        raise ValueError("xyz/colors/labels must align")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    shifted = xyz - xyz.min(axis=0, keepdims=True)
    discrete = np.floor(shifted / float(voxel_size)).astype(np.int32)
    _unique_sorted, first_sorted, inverse_sorted = np.unique(
        discrete,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    insertion_order = np.argsort(first_sorted, kind="stable")
    unique_map = first_sorted[insertion_order]
    sorted_to_insertion = np.empty_like(insertion_order)
    sorted_to_insertion[insertion_order] = np.arange(insertion_order.size)
    inverse = sorted_to_insertion[inverse_sorted]
    unique = discrete[unique_map]
    return QuantizedScene(
        coordinates=np.ascontiguousarray(unique),
        raw_coordinates=np.ascontiguousarray(shifted[unique_map]),
        colors=np.ascontiguousarray(colors[unique_map]),
        labels=np.ascontiguousarray(labels[unique_map]),
        inverse_map=np.ascontiguousarray(inverse.astype(np.int64)),
        unique_map=np.ascontiguousarray(unique_map.astype(np.int64)),
    )


def _cluster_center(
    coordinates: np.ndarray,
    error_mask: np.ndarray,
    *,
    workers: int = -1,
) -> tuple[int, float]:
    error_indices = np.flatnonzero(error_mask)
    background_indices = np.flatnonzero(~error_mask)
    if not len(error_indices) or not len(background_indices):
        raise ValueError("an error cluster needs foreground and complement points")
    tree = cKDTree(np.asarray(coordinates, dtype=np.float32)[background_indices])
    distances, _ = tree.query(
        np.asarray(coordinates, dtype=np.float32)[error_indices],
        k=1,
        workers=int(workers),
    )
    local = int(np.argmax(distances))
    return int(error_indices[local]), float(distances[local])


def select_next_click(
    coordinates: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    order: int,
    workers: int = -1,
) -> Click | None:
    """Place the next click at the center of the largest prediction error.

    False-positive and false-negative sets are AGILE3D's two binary error
    clusters.  Their size is the largest distance from an error point to any
    non-error point, not point count and not a connected-component volume.
    """

    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (target.size, 3) or prediction.shape != target.shape:
        raise ValueError("coordinates, prediction, and target must align")
    if int(workers) == 0 or int(workers) < -1:
        raise ValueError("workers must be -1 or a positive integer")
    choices: list[tuple[float, int, int, bool]] = []
    for cluster_id, mask, positive in (
        (11, prediction & ~target, False),
        (96, ~prediction & target, True),
    ):
        if not bool(mask.any()) or bool(mask.all()):
            continue
        point_index, radius = _cluster_center(coordinates, mask, workers=workers)
        choices.append((radius, cluster_id, point_index, positive))
    if not choices:
        return None
    # Descending radius; official FP cluster id 11 precedes FN id 96 on ties.
    _radius, _cluster, point_index, positive = sorted(
        choices,
        key=lambda row: (-row[0], row[1]),
    )[0]
    return Click(point_index=point_index, is_positive=positive, order=int(order))


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 0.0


Predictor = Callable[[np.ndarray, np.ndarray, Sequence[Click]], np.ndarray]


def evaluate_interactive_predictions(
    coordinates: np.ndarray,
    target_quantized: np.ndarray,
    target_full: np.ndarray,
    inverse_map: np.ndarray,
    predictor: Predictor,
    *,
    max_clicks: int = 20,
    click_workers: int = -1,
) -> dict:
    """Run one official single-object interaction trajectory."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    target_quantized = np.asarray(target_quantized, dtype=bool).reshape(-1)
    target_full = np.asarray(target_full, dtype=bool).reshape(-1)
    inverse_map = np.asarray(inverse_map, dtype=np.int64).reshape(-1)
    if coordinates.shape != (target_quantized.size, 3):
        raise ValueError("quantized coordinates and target must align")
    if inverse_map.shape != target_full.shape:
        raise ValueError("full target and inverse map must align")
    if max_clicks <= 0:
        raise ValueError("max_clicks must be positive")
    if int(click_workers) == 0 or int(click_workers) < -1:
        raise ValueError("click_workers must be -1 or a positive integer")
    prediction = np.zeros_like(target_quantized)
    clicks: list[Click] = []
    trajectory: dict[int, float] = {}
    for click_count in range(1, int(max_clicks) + 1):
        click = select_next_click(
            coordinates,
            prediction,
            target_quantized,
            order=click_count - 1,
            workers=int(click_workers),
        )
        if click is not None:
            clicks.append(click)
        prediction = np.asarray(
            predictor(coordinates, prediction.copy(), tuple(clicks)),
            dtype=bool,
        ).reshape(-1)
        if prediction.shape != target_quantized.shape:
            raise ValueError("predictor output does not align with quantized points")
        # AGILE3D explicitly overwrites model errors at every clicked point.
        for item in clicks:
            prediction[item.point_index] = item.is_positive
        trajectory[click_count] = _iou(prediction[inverse_map], target_full)
    return {
        "trajectory": trajectory,
        "clicks": [
            {
                "point_index": click.point_index,
                "is_positive": click.is_positive,
                "order": click.order,
            }
            for click in clicks
        ],
    }


def aggregate_official_metrics(
    trajectories: Iterable[Mapping[int, float]],
    *,
    max_clicks: int = 20,
) -> dict[str, float]:
    rows = [{int(key): float(value) for key, value in row.items()} for row in trajectories]
    if not rows:
        raise ValueError("no interactive trajectories to aggregate")
    metrics = {
        f"IoU@{count}": float(np.mean([row[count] for row in rows]))
        for count in IOU_CLICK_COUNTS
    }
    for threshold in NOC_THRESHOLDS:
        values = []
        for row in rows:
            reached = next(
                (count for count in range(1, int(max_clicks) + 1)
                 if row[count] >= threshold),
                int(max_clicks),
            )
            values.append(reached)
        metrics[f"NoC@{int(round(threshold * 100))}"] = float(np.mean(values))
    return metrics
