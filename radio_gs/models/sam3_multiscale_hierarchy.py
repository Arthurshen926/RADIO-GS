"""Query-independent geometry for an official-SAM3 multiscale region hierarchy.

This module deliberately contains no benchmark, text-query, or annotation API.
It turns crop-local binary masks into one full-image proposal set and a direct
containment graph.  The GPU-facing official-SAM3 runner lives in
``radio_gs.scripts.build_sam3_multiscale_hierarchy_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np
import torch


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class CropSpec:
    """One half-open crop box and its deterministic point-grid contract."""

    layer: int
    index: int
    row: int
    column: int
    box_xyxy: tuple[int, int, int, int]
    grid_side: int

    @property
    def width(self) -> int:
        return self.box_xyxy[2] - self.box_xyxy[0]

    @property
    def height(self) -> int:
        return self.box_xyxy[3] - self.box_xyxy[1]


def axis_crop_intervals(
    length: int, *, count: int, overlap_ratio: float
) -> tuple[tuple[int, int], ...]:
    """Return edge-anchored, gap-free, approximately equal crop intervals.

    ``overlap_ratio`` is the desired adjacent overlap divided by crop length.
    Rounding is deterministic: interior starts use round-half-up and the first
    and final crops are always anchored to the image boundaries.
    """

    length = int(length)
    count = int(count)
    ratio = float(overlap_ratio)
    if length <= 0:
        raise ValueError("axis length must be positive")
    if count <= 0:
        raise ValueError("crop count must be positive")
    if not 0.0 <= ratio < 1.0:
        raise ValueError("crop overlap ratio must lie in [0,1)")
    if count == 1:
        return ((0, length),)

    denominator = count - ratio * (count - 1)
    crop_length = min(length, max(1, int(math.ceil(length / denominator))))
    maximum_start = length - crop_length
    starts = [
        int(math.floor((maximum_start * index / (count - 1)) + 0.5))
        for index in range(count)
    ]
    starts[0] = 0
    starts[-1] = maximum_start
    intervals = tuple((start, min(length, start + crop_length)) for start in starts)
    if intervals[0][0] != 0 or intervals[-1][1] != length:
        raise AssertionError("crop intervals lost an image edge")
    if any(right[0] > left[1] for left, right in zip(intervals, intervals[1:])):
        raise AssertionError("crop intervals contain a coverage gap")
    return intervals


def build_crop_pyramid(
    *,
    image_width: int,
    image_height: int,
    crop_layers: int,
    overlap_ratio: float,
    points_per_side: int,
    point_grid_downscale_factor: int,
) -> tuple[CropSpec, ...]:
    """Build a full-image plus ``2**layer`` edge-anchored crop pyramid."""

    width, height = int(image_width), int(image_height)
    layers = int(crop_layers)
    base_grid = int(points_per_side)
    downscale = int(point_grid_downscale_factor)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if layers < 0:
        raise ValueError("crop_layers must be non-negative")
    if base_grid <= 0 or downscale <= 0:
        raise ValueError("point-grid sizes must be positive")

    crops: list[CropSpec] = []
    index = 0
    for layer in range(layers + 1):
        count = 2**layer
        xs = axis_crop_intervals(width, count=count, overlap_ratio=overlap_ratio)
        ys = axis_crop_intervals(height, count=count, overlap_ratio=overlap_ratio)
        grid_side = max(2, base_grid // (downscale**layer))
        if layer == 0:
            grid_side = base_grid
        for row, (y0, y1) in enumerate(ys):
            for column, (x0, x1) in enumerate(xs):
                crops.append(
                    CropSpec(
                        layer=layer,
                        index=index,
                        row=row,
                        column=column,
                        box_xyxy=(x0, y0, x1, y1),
                        grid_side=grid_side,
                    )
                )
                index += 1
    return tuple(crops)


def dense_point_grid(crop: CropSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return crop-local and full-image pixel-centre prompt coordinates."""

    side = int(crop.grid_side)
    xs = (np.arange(side, dtype=np.float32) + 0.5) * (crop.width / side)
    ys = (np.arange(side, dtype=np.float32) + 0.5) * (crop.height / side)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    local = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1).astype(np.float32)
    x0, y0, _, _ = crop.box_xyxy
    full = local + np.asarray([x0, y0], dtype=np.float32)
    return local, full


def remap_crop_mask(
    crop_mask: np.ndarray,
    *,
    crop_box_xyxy: Sequence[int],
    full_height: int,
    full_width: int,
) -> np.ndarray:
    """Embed one crop-local mask into full-image coordinates without resize."""

    x0, y0, x1, y1 = (int(value) for value in crop_box_xyxy)
    if not (0 <= x0 < x1 <= int(full_width) and 0 <= y0 < y1 <= int(full_height)):
        raise ValueError("crop box lies outside the full image")
    mask = np.asarray(crop_mask, dtype=bool)
    if mask.shape != (y1 - y0, x1 - x0):
        raise ValueError(
            f"crop mask shape {mask.shape} differs from box raster {(y1-y0, x1-x0)}"
        )
    full = np.zeros((int(full_height), int(full_width)), dtype=bool)
    full[y0:y1, x0:x1] = mask
    return full


def binary_mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        raise ValueError("binary mask must be non-empty [H,W]")
    ys, xs = np.where(values)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_edge_flags(
    local_box_xyxy: Sequence[int],
    *,
    crop: CropSpec,
    image_width: int,
    image_height: int,
    tolerance_pixels: int,
) -> tuple[tuple[bool, bool, bool, bool], tuple[bool, bool, bool, bool]]:
    """Return crop-edge and artificial-edge flags in left/top/right/bottom order."""

    x0, y0, x1, y1 = (int(value) for value in local_box_xyxy)
    tolerance = int(tolerance_pixels)
    if tolerance < 0:
        raise ValueError("edge tolerance must be non-negative")
    touches = (
        x0 <= tolerance,
        y0 <= tolerance,
        x1 >= crop.width - tolerance,
        y1 >= crop.height - tolerance,
    )
    crop_x0, crop_y0, crop_x1, crop_y1 = crop.box_xyxy
    crop_is_image_edge = (
        crop_x0 == 0,
        crop_y0 == 0,
        crop_x1 == int(image_width),
        crop_y1 == int(image_height),
    )
    artificial = tuple(
        bool(touch and not true_edge)
        for touch, true_edge in zip(touches, crop_is_image_edge)
    )
    return tuple(bool(value) for value in touches), artificial


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask IoU requires equal shapes")
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def containment_aware_deduplicate(
    masks: Sequence[np.ndarray],
    ranking_scores: Sequence[float],
    *,
    iou_threshold: float,
    near_equal_area_ratio: float,
    maximum_masks: int,
) -> list[int]:
    """Suppress near-identical masks while preserving proper containment."""

    if len(masks) != len(ranking_scores):
        raise ValueError("masks and ranking scores must align")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("dedup IoU threshold must lie in [0,1]")
    if not 0.0 < float(near_equal_area_ratio) <= 1.0:
        raise ValueError("near-equal area ratio must lie in (0,1]")
    values = [np.asarray(mask, dtype=bool) for mask in masks]
    areas = [int(mask.sum(dtype=np.int64)) for mask in values]
    boxes = [binary_mask_box(mask) if area else (0, 0, 0, 0) for mask, area in zip(values, areas)]
    order = sorted(range(len(masks)), key=lambda i: (-float(ranking_scores[i]), i))
    kept: list[int] = []
    for index in order:
        if areas[index] == 0:
            continue
        duplicate = False
        for other in kept:
            area_ratio = min(areas[index], areas[other]) / max(areas[index], areas[other])
            if area_ratio < float(near_equal_area_ratio):
                continue
            ax0, ay0, ax1, ay1 = boxes[index]
            bx0, by0, bx1, by1 = boxes[other]
            ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            bounding_intersection = (ix1 - ix0) * (iy1 - iy0)
            maximum_possible_iou = bounding_intersection / max(
                areas[index] + areas[other] - bounding_intersection, 1
            )
            if maximum_possible_iou <= float(iou_threshold):
                continue
            intersection = np.logical_and(
                values[index][iy0:iy1, ix0:ix1],
                values[other][iy0:iy1, ix0:ix1],
            ).sum(dtype=np.int64)
            union = areas[index] + areas[other] - intersection
            if float(intersection / max(union, 1)) > float(iou_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
            if int(maximum_masks) > 0 and len(kept) >= int(maximum_masks):
                break
    return kept


def direct_containment_graph(
    masks: Sequence[np.ndarray],
    quality: Sequence[float],
    *,
    containment_threshold: float,
    minimum_parent_area_ratio: float,
) -> dict[str, torch.Tensor]:
    """Construct the transitive-reduced one-parent containment forest.

    A child selects the smallest mask that contains the requested fraction of
    its pixels.  Quality and stable index order only break equal-area ties.
    The resulting parent is therefore the nearest observed enclosing region.
    """

    count = len(masks)
    if len(quality) != count:
        raise ValueError("masks and quality must align")
    threshold = float(containment_threshold)
    area_ratio = float(minimum_parent_area_ratio)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("containment threshold must lie in (0,1]")
    if area_ratio <= 1.0:
        raise ValueError("minimum parent area ratio must exceed one")
    values = [np.asarray(mask, dtype=bool) for mask in masks]
    if values and any(mask.shape != values[0].shape for mask in values):
        raise ValueError("hierarchy masks must share a raster")
    areas = np.asarray([mask.sum(dtype=np.int64) for mask in values], dtype=np.int64)
    boxes = [binary_mask_box(mask) if areas[index] else (0, 0, 0, 0) for index, mask in enumerate(values)]
    area_order = sorted(range(count), key=lambda index: (int(areas[index]), index))
    parent = np.full(count, -1, dtype=np.int64)
    parent_containment = np.zeros(count, dtype=np.float32)
    parent_area_ratio = np.zeros(count, dtype=np.float32)
    for child in range(count):
        if areas[child] <= 0:
            continue
        candidates: list[tuple[int, float, float, int]] = []
        smallest_parent_area: int | None = None
        child_x0, child_y0, child_x1, child_y1 = boxes[child]
        for possible_parent in area_order:
            if possible_parent == child:
                continue
            ratio = float(areas[possible_parent] / areas[child])
            if ratio < area_ratio:
                continue
            if smallest_parent_area is not None and int(areas[possible_parent]) > smallest_parent_area:
                break
            parent_x0, parent_y0, parent_x1, parent_y1 = boxes[possible_parent]
            ix0, iy0 = max(child_x0, parent_x0), max(child_y0, parent_y0)
            ix1, iy1 = min(child_x1, parent_x1), min(child_y1, parent_y1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            if (ix1 - ix0) * (iy1 - iy0) < threshold * areas[child]:
                continue
            contained = float(
                np.logical_and(
                    values[child][iy0:iy1, ix0:ix1],
                    values[possible_parent][iy0:iy1, ix0:ix1],
                ).sum(dtype=np.int64)
                / areas[child]
            )
            if contained >= threshold:
                smallest_parent_area = int(areas[possible_parent])
                candidates.append(
                    (
                        int(areas[possible_parent]),
                        -contained,
                        -float(quality[possible_parent]),
                        possible_parent,
                    )
                )
        if candidates:
            _, negative_contained, _, selected = min(candidates)
            parent[child] = selected
            parent_containment[child] = -negative_contained
            parent_area_ratio[child] = float(areas[selected] / areas[child])
    child_indices = np.flatnonzero(parent >= 0).astype(np.int64)
    edges = (
        np.stack((parent[child_indices], child_indices), axis=1)
        if child_indices.size
        else np.empty((0, 2), dtype=np.int64)
    )
    return {
        "parent_index": torch.from_numpy(parent),
        "parent_edges": torch.from_numpy(edges),
        "parent_containment": torch.from_numpy(parent_containment),
        "parent_area_ratio": torch.from_numpy(parent_area_ratio),
    }


def pack_masks(masks: np.ndarray) -> torch.Tensor:
    values = np.asarray(masks, dtype=np.uint8)
    if values.ndim != 3:
        raise ValueError("masks must be [M,H,W]")
    return torch.from_numpy(np.packbits(values, axis=-1, bitorder="little"))


def unpack_masks(packed: torch.Tensor, *, width: int) -> np.ndarray:
    values = np.unpackbits(
        torch.as_tensor(packed).cpu().numpy(), axis=-1, bitorder="little"
    )
    return values[..., : int(width)].astype(bool)


def validate_source_authority_payload(payload: Mapping[str, object]) -> tuple[dict, ...]:
    """Validate the explicit source-only information contract fail-closed."""

    expected_policy = {
        "registered_source_rgb_only": True,
        "query_text_used": False,
        "benchmark_ground_truth_used": False,
        "target_or_evaluation_rgb_used": False,
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != "sam3-query-free-source-rgb-authority-v1"
        or payload.get("information_policy") != expected_policy
    ):
        raise ValueError("source RGB authority contract or information policy differs")
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("source RGB authority must contain a non-empty images list")
    records: list[dict] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_images:
        if not isinstance(raw, Mapping):
            raise ValueError("source RGB image record must be an object")
        image_id = str(raw.get("image_id", ""))
        path = str(raw.get("path", ""))
        sha256 = require_sha256(raw.get("sha256", ""), label="source image sha256")
        if (
            IMAGE_ID_PATTERN.fullmatch(image_id) is None
            or not path
            or image_id in seen_ids
            or path in seen_paths
        ):
            raise ValueError("source RGB image ids and paths must be non-empty and unique")
        if raw.get("rgb_role") != "registered_source_or_mapping_view":
            raise ValueError(f"source RGB role differs for {image_id}")
        records.append({"image_id": image_id, "path": path, "sha256": sha256})
        seen_ids.add(image_id)
        seen_paths.add(path)
    return tuple(records)


def validate_multiscale_cache_payload(
    payload: Mapping[str, object],
    *,
    expected_metadata: Mapping[str, object],
) -> int:
    """Validate an existing cache before resume; unknown/missing fields fail."""

    if payload.get("schema_version") != 1 or payload.get("metadata") != expected_metadata:
        raise ValueError("multiscale SAM3 cache identity differs")
    shape = payload.get("mask_shape")
    if not isinstance(shape, list) or len(shape) != 2 or min(map(int, shape)) <= 0:
        raise ValueError("multiscale SAM3 mask shape is invalid")
    packed = torch.as_tensor(payload.get("packed_masks"))
    if packed.ndim != 3 or packed.dtype != torch.uint8:
        raise ValueError("multiscale SAM3 packed masks must be uint8 [M,H,ceil(W/8)]")
    count, height, packed_width = map(int, packed.shape)
    if height != int(shape[0]) or packed_width != (int(shape[1]) + 7) // 8:
        raise ValueError("multiscale SAM3 packed-mask raster differs")
    aligned = {
        "quality": (count,),
        "stability": (count,),
        "seed_xy_full": (count, 2),
        "seed_xy_crop": (count, 2),
        "prompt_index": (count,),
        "candidate_index": (count,),
        "crop_index": (count,),
        "crop_layer": (count,),
        "crop_grid_side": (count,),
        "crop_boxes_xyxy": (count, 4),
        "crop_scale_xy": (count, 2),
        "crop_window_area_fraction": (count,),
        "boxes_xyxy": (count, 4),
        "proposal_area_fraction": (count,),
        "crop_area_fraction": (count,),
        "touches_crop_edge": (count, 4),
        "touches_artificial_crop_edge": (count, 4),
        "parent_index": (count,),
        "parent_containment": (count,),
        "parent_area_ratio": (count,),
    }
    for key, expected_shape in aligned.items():
        if tuple(torch.as_tensor(payload.get(key)).shape) != expected_shape:
            raise ValueError(f"multiscale SAM3 field {key} differs")
    edges = torch.as_tensor(payload.get("parent_edges"))
    if edges.ndim != 2 or tuple(edges.shape[1:]) != (2,):
        raise ValueError("multiscale SAM3 parent edges must be [E,2]")
    if count and (
        not torch.isfinite(torch.as_tensor(payload["quality"])).all()
        or not torch.isfinite(torch.as_tensor(payload["stability"])).all()
    ):
        raise ValueError("multiscale SAM3 quality metadata is non-finite")
    return count
