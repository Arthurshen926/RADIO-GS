#!/usr/bin/env python3
"""Build auditable qualitative panels for the frozen PFIR and AGILE3D runs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from plyfile import PlyData
import torch

from radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache import (
    CanonicalPointPredictor,
    _load_feature_cache,
    _read_official_ply,
)
from radio_gs.benchmarks.agile3d_scannet40.protocol import Click, select_next_click
from radio_gs.benchmarks.agile3d_scannet40.protocol import quantize_scannet_points
from radio_gs.benchmarks.scannet_pfir.build_benchmark import find_scene_annotations
from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import SupportGraphConfig, SupportSolverConfig


TP_COLOR = (34, 139, 34)
FP_COLOR = (220, 53, 69)
FN_COLOR = (244, 180, 0)
BG_COLOR = (186, 190, 196)
PREDICTION_COLOR = (0, 166, 204)
POSITIVE_CLICK_COLOR = (25, 118, 210)
NEGATIVE_CLICK_COLOR = (179, 31, 97)
FOCUS_CONTEXT_SCALE = 3.0
FOCUS_MINIMUM_FRACTION = 0.28


def _as_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing {key} for {row.get('query_id', row.get('scene_id'))}")
    return float(value)


def select_pfir_cases(
    ranking_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select three fixed PFIR cases from saved formal per-query records.

    The selectors deliberately use only frozen evaluation fields.  They do not
    open raw predictions, masks, images, or GT meshes while choosing examples.
    """

    selection_by_id = {str(row["query_id"]): dict(row) for row in selection_rows}
    if len(selection_by_id) != len(selection_rows):
        raise ValueError("PFIR selection rows contain duplicate query IDs")
    combined: list[dict[str, Any]] = []
    for ranking in ranking_rows:
        query_id = str(ranking["query_id"])
        selection = selection_by_id.get(query_id)
        if selection is None:
            raise ValueError(f"PFIR selection row missing for {query_id}")
        combined.append({**dict(ranking), **selection})

    def choose(kind: str, rows: list[dict[str, Any]], key) -> dict[str, Any]:
        if not rows:
            raise ValueError(f"no PFIR case satisfies frozen selector {kind}")
        return {"kind": kind, **min(rows, key=key)}

    success = choose(
        "success",
        [
            row
            for row in combined
            if int(row["rank"]) == 1 and _as_float(row, "iou") >= 0.50
        ],
        lambda row: (-_as_float(row, "iou"), str(row["query_id"])),
    )
    rank_mask_gap = choose(
        "rank_mask_gap",
        [
            row
            for row in combined
            if int(row["rank"]) == 1 and _as_float(row, "iou") <= 0.05
        ],
        lambda row: (_as_float(row, "iou"), str(row["query_id"])),
    )
    same_class_confusion = choose(
        "same_class_confusion",
        [
            row
            for row in combined
            if bool(row.get("same_category"))
            and str(row["query_id"]).endswith("_hard")
            and int(row["rank"]) > 1
            and _as_float(row, "iou") <= 0.15
            and row.get("same_category_distractor_success") is False
        ],
        lambda row: (-int(row["rank"]), _as_float(row, "iou"), str(row["query_id"])),
    )
    return [success, rank_mask_gap, same_class_confusion]


def _trajectory_value(row: Mapping[str, Any], click_count: int) -> float:
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"AGILE row has no trajectory: {row}")
    if click_count in trajectory:
        return float(trajectory[click_count])
    if str(click_count) in trajectory:
        return float(trajectory[str(click_count)])
    raise ValueError(f"AGILE row lacks click {click_count}")


def select_agile_cases(
    object_rows: Sequence[Mapping[str, Any]],
    scene_coverage: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select high/middle/low-coverage AGILE cases from frozen formal rows."""

    coverage_by_scene = {
        str(row["scene_id"]): float(row["feature_coverage"])
        for row in scene_coverage
    }
    if len(coverage_by_scene) != len(scene_coverage):
        raise ValueError("AGILE scene coverage contains duplicate scene IDs")
    candidates: list[dict[str, Any]] = []
    for source in object_rows:
        scene_id = str(source["scene_id"])
        if scene_id not in coverage_by_scene:
            raise ValueError(f"AGILE coverage missing for {scene_id}")
        candidates.append(
            {
                **dict(source),
                "feature_coverage": coverage_by_scene[scene_id],
                "iou_at_15": _trajectory_value(source, 15),
            }
        )
    if len(candidates) < 3:
        raise ValueError("AGILE qualitative selection needs at least three objects")

    def identity(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row["scene_id"]), int(row["object_id"])

    high = min(
        candidates,
        key=lambda row: (-float(row["feature_coverage"]), -float(row["iou_at_15"]), *identity(row)),
    )
    low = min(
        candidates,
        key=lambda row: (float(row["feature_coverage"]), float(row["iou_at_15"]), *identity(row)),
    )
    excluded = {identity(high), identity(low)}
    available = [row for row in candidates if identity(row) not in excluded]
    median_coverage = float(np.median([row["feature_coverage"] for row in candidates]))
    median_iou = float(np.median([row["iou_at_15"] for row in candidates]))
    middle = min(
        available,
        key=lambda row: (
            abs(float(row["feature_coverage"]) - median_coverage)
            + abs(float(row["iou_at_15"]) - median_iou),
            *identity(row),
        ),
    )
    return [
        {"kind": "high_coverage_success", **high},
        {"kind": "middle_coverage_middle_iou", **middle},
        {"kind": "low_coverage_failure", **low},
    ]


def mask_error_colors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Assign a stable RGB color to TP, FP, FN, and background points."""

    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same length")
    colors = np.broadcast_to(np.asarray(BG_COLOR, dtype=np.uint8), (len(target), 3)).copy()
    colors[prediction & target] = TP_COLOR
    colors[prediction & ~target] = FP_COLOR
    colors[~prediction & target] = FN_COLOR
    return colors


def assert_trajectory_matches(
    expected: Mapping[int | str, float],
    actual: Mapping[int | str, float],
    *,
    atol: float = 1e-8,
) -> None:
    """Raise when a qualitative AGILE replay diverges from its formal result."""

    expected_by_click = {int(key): float(value) for key, value in expected.items()}
    actual_by_click = {int(key): float(value) for key, value in actual.items()}
    if set(expected_by_click) != set(actual_by_click):
        raise AssertionError(
            f"click set differs: expected={sorted(expected_by_click)}, actual={sorted(actual_by_click)}"
        )
    for click_count in sorted(expected_by_click):
        if not np.isclose(expected_by_click[click_count], actual_by_click[click_count], atol=atol):
            raise AssertionError(
                f"click {click_count} trajectory differs: "
                f"expected={expected_by_click[click_count]:.10f}, "
                f"actual={actual_by_click[click_count]:.10f}"
            )


def _mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 0.0


def replay_interactive_masks(
    coordinates: np.ndarray,
    target_quantized: np.ndarray,
    target_full: np.ndarray,
    inverse_map: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, Sequence[Click]], np.ndarray],
    *,
    max_clicks: int,
    capture_click_counts: Sequence[int],
    click_workers: int,
) -> dict[str, Any]:
    """Replay AGILE3D clicks while retaining masks needed for static panels."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    target_quantized = np.asarray(target_quantized, dtype=bool).reshape(-1)
    target_full = np.asarray(target_full, dtype=bool).reshape(-1)
    inverse_map = np.asarray(inverse_map, dtype=np.int64).reshape(-1)
    if coordinates.shape != (target_quantized.size, 3):
        raise ValueError("quantized coordinates and target must align")
    if inverse_map.shape != target_full.shape:
        raise ValueError("full target and inverse map must align")
    requested = {int(count) for count in capture_click_counts}
    if not requested or min(requested) <= 0 or max(requested) > int(max_clicks):
        raise ValueError("capture click counts must be inside [1, max_clicks]")

    prediction = np.zeros_like(target_quantized)
    clicks: list[Click] = []
    trajectory: dict[int, float] = {}
    snapshots: dict[int, dict[str, Any]] = {}
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
            predictor(coordinates, prediction.copy(), tuple(clicks)), dtype=bool
        ).reshape(-1)
        if prediction.shape != target_quantized.shape:
            raise ValueError("predictor output does not align with quantized scene")
        for item in clicks:
            prediction[item.point_index] = item.is_positive
        trajectory[click_count] = _mask_iou(prediction[inverse_map], target_full)
        if click_count in requested:
            snapshots[click_count] = {
                "prediction": prediction.copy(),
                "clicks": tuple(clicks),
            }
    return {"trajectory": trajectory, "snapshots": snapshots, "clicks": tuple(clicks)}


def _oriented_pca_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic screen/right, up, and viewing directions from geometry."""

    points = np.asarray(xyz, dtype=np.float32)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] < 3:
        raise ValueError("at least three finite 3-D points are required")
    sampled = finite[:: max(1, int(np.ceil(len(finite) / 50000)))]
    centered = sampled - sampled.mean(axis=0, keepdims=True)
    _u, _s, vectors = np.linalg.svd(centered, full_matrices=False)

    def oriented(vector: np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float64)
        pivot = int(np.argmax(np.abs(result)))
        if result[pivot] < 0:
            result = -result
        return result / max(float(np.linalg.norm(result)), 1e-8)

    right = oriented(vectors[0])
    view = oriented(np.cross(vectors[0], vectors[1]))
    if float(np.linalg.norm(view)) < 1e-6:
        view = np.array([0.0, 0.0, 1.0])
    up = np.cross(view, right)
    if float(np.linalg.norm(up)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    up = oriented(up)
    return right.astype(np.float32), up.astype(np.float32), view.astype(np.float32)


def _project_orthographic(
    xyz: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = map(int, size)
    if width < 8 or height < 8:
        raise ValueError("render size must be at least 8x8")
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    right, up, view = _oriented_pca_basis(points)
    centered = points - np.nanmean(points, axis=0, keepdims=True)
    screen = np.column_stack((centered @ right, centered @ up))
    extent = np.nanmax(np.ptp(screen, axis=0))
    if not np.isfinite(extent) or extent < 1e-6:
        extent = 1.0
    margin = 0.06
    scale = (1.0 - 2.0 * margin) / extent
    x = np.rint((screen[:, 0] - np.nanmean(screen[:, 0])) * scale * width + width * 0.5)
    y = np.rint(height * 0.5 - (screen[:, 1] - np.nanmean(screen[:, 1])) * scale * width)
    pixels = np.column_stack(
        (
            np.clip(x, 0, width - 1).astype(np.int64),
            np.clip(y, 0, height - 1).astype(np.int64),
        )
    )
    return pixels, (centered @ view).astype(np.float32), np.stack((right, up, view))


def _visible_indices(pixels: np.ndarray, depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = map(int, size)
    linear = pixels[:, 1] * width + pixels[:, 0]
    valid = np.isfinite(depth)
    z_buffer = np.full(width * height, -np.inf, dtype=np.float32)
    np.maximum.at(z_buffer, linear[valid], depth[valid])
    visible = valid & np.isclose(depth, z_buffer[linear], atol=1e-6)
    return np.flatnonzero(visible)


def _blend(base: np.ndarray, color: tuple[int, int, int], weight: float = 0.82) -> np.ndarray:
    return np.rint((1.0 - weight) * base + weight * np.asarray(color)).astype(np.uint8)


def _focus_crop_box(
    pixels: np.ndarray,
    visible: np.ndarray,
    focus_mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a same-aspect crop around an evaluator-only focus mask.

    The orthographic camera and rasterization stay fixed.  This function only
    crops that fixed projection after rendering, so it cannot affect a method
    prediction or use labels to select a view direction.
    """

    width, height = map(int, size)
    focus_visible = visible[focus_mask[visible]]
    candidates = focus_visible if len(focus_visible) else np.flatnonzero(focus_mask)
    if not len(candidates):
        return None
    focus_pixels = pixels[candidates]
    x_min, y_min = np.min(focus_pixels, axis=0)
    x_max, y_max = np.max(focus_pixels, axis=0)
    focus_width = int(x_max - x_min + 1)
    focus_height = int(y_max - y_min + 1)
    aspect = float(width) / float(height)
    desired_height = max(
        float(focus_height) * FOCUS_CONTEXT_SCALE,
        float(focus_width) * FOCUS_CONTEXT_SCALE / aspect,
        float(height) * FOCUS_MINIMUM_FRACTION,
    )
    crop_height = min(height, max(1, int(np.ceil(desired_height))))
    crop_width = min(width, max(1, int(np.ceil(crop_height * aspect))))
    center_x = (float(x_min) + float(x_max)) * 0.5
    center_y = (float(y_min) + float(y_max)) * 0.5
    left = int(round(center_x - crop_width * 0.5))
    top = int(round(center_y - crop_height * 0.5))
    left = int(np.clip(left, 0, width - crop_width))
    top = int(np.clip(top, 0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def render_mask_view(
    xyz: np.ndarray,
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    size: tuple[int, int] = (720, 540),
    overlay: str = "error",
    clicks: Sequence[Click] = (),
    focus_mask: np.ndarray | None = None,
) -> Image.Image:
    """Rasterize one geometry-only view with an optional evaluator-only crop."""

    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    target = np.asarray(target, dtype=bool).reshape(-1)
    prediction = np.asarray(prediction, dtype=bool).reshape(-1)
    if points.shape != (len(target), 3) or colors.shape != (len(target), 3):
        raise ValueError("xyz, rgb, target, and prediction must align")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must align")
    if overlay not in {"rgb", "gt", "prediction", "error"}:
        raise ValueError("overlay must be rgb, gt, prediction, or error")
    focus = None
    if focus_mask is not None:
        focus = np.asarray(focus_mask, dtype=bool).reshape(-1)
        if focus.shape != target.shape:
            raise ValueError("focus_mask and target must align")

    point_colors = colors.copy()
    if overlay == "gt":
        point_colors[target] = _blend(point_colors[target], TP_COLOR)
    elif overlay == "prediction":
        point_colors[prediction] = _blend(point_colors[prediction], PREDICTION_COLOR)
    elif overlay == "error":
        point_colors = mask_error_colors(prediction, target)

    pixels, depth, _basis = _project_orthographic(points, size)
    visible = _visible_indices(pixels, depth, size)
    width, height = map(int, size)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    # Stable point-index ordering resolves exact depth ties without a target-dependent rule.
    canvas[pixels[visible, 1], pixels[visible, 0]] = point_colors[visible]
    image = Image.fromarray(canvas, mode="RGB")
    crop_box = _focus_crop_box(pixels, visible, focus, size) if focus is not None else None
    if crop_box is not None:
        image = image.crop(crop_box).resize(size, resample=Image.Resampling.NEAREST)
    if clicks:
        draw = ImageDraw.Draw(image)
        for item in clicks:
            if item.point_index < 0 or item.point_index >= len(points):
                raise IndexError("click is outside the displayed point cloud")
            x, y = map(int, pixels[item.point_index])
            if crop_box is not None:
                left, top, right, bottom = crop_box
                if not (left <= x < right and top <= y < bottom):
                    continue
                x = int(round((x - left + 0.5) * width / (right - left) - 0.5))
                y = int(round((y - top + 0.5) * height / (bottom - top) - 0.5))
            color = POSITIVE_CLICK_COLOR if item.is_positive else NEGATIVE_CLICK_COLOR
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            draw.line((x - radius + 2, y, x + radius - 2, y), fill=color, width=2)
            if item.is_positive:
                draw.line((x, y - radius + 2, x, y + radius - 2), fill=color, width=2)
    return image


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mesh_rgb(mesh_path: str | Path, expected_rows: int) -> np.ndarray:
    vertex = PlyData.read(str(mesh_path))["vertex"].data
    names = set(vertex.dtype.names or ())
    candidates = (("red", "green", "blue"), ("R", "G", "B"))
    channels = next((value for value in candidates if set(value).issubset(names)), None)
    if channels is None:
        raise ValueError(f"{mesh_path} lacks RGB vertex properties")
    rgb = np.column_stack([vertex[name] for name in channels]).astype(np.uint8)
    if rgb.shape != (int(expected_rows), 3):
        raise ValueError("mesh RGB rows do not align with evaluator mesh")
    return rgb


def load_pfir_visual_case(
    case: Mapping[str, Any],
    *,
    benchmark_root: str | Path,
    prediction_root: str | Path,
    annotations_root: str | Path,
) -> dict[str, Any]:
    """Load a selected PFIR input, saved prediction, and evaluator-only mesh GT."""

    benchmark = Path(benchmark_root)
    manifest = _read_json(benchmark / "manifest.internal.json")
    query_id = str(case["query_id"])
    query_by_id = {str(row["query_id"]): row for row in manifest["queries"]}
    query = query_by_id.get(query_id)
    if query is None:
        raise ValueError(f"PFIR query is absent from frozen manifest: {query_id}")
    scene_id = str(query["scene_id"])
    mesh, aggregation, segmentation = find_scene_annotations(scene_id, [annotations_root])
    xyz, instances, metadata = load_mesh_instances(mesh, aggregation, segmentation)
    rgb = _read_mesh_rgb(mesh, len(xyz))
    target_id = int(query["instance_id_3d"])
    if target_id not in metadata:
        raise ValueError(f"PFIR target {target_id} is absent from {scene_id} annotations")
    target = instances == target_id
    prediction_path = Path(prediction_root) / f"{query_id}.npy"
    prediction = np.load(prediction_path, allow_pickle=False).astype(bool, copy=False)
    if prediction.shape != target.shape:
        raise ValueError(f"{query_id} prediction does not align with official mesh")
    crop_path = Path(str(query["crop_rgb_path"]))
    if not crop_path.is_file():
        raise FileNotFoundError(f"PFIR crop is missing: {crop_path}")
    with Image.open(crop_path) as source:
        crop = source.convert("RGB").copy()
    return {
        "case": dict(case),
        "query": query,
        "mesh_path": str(mesh.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "crop_path": str(crop_path.resolve()),
        "crop": crop,
        "xyz": xyz,
        "rgb": rgb,
        "target": target,
        "prediction": prediction,
    }


def _formal_agile_predictor(
    *,
    scene_id: str,
    benchmark_root: Path,
    feature_root: Path,
    protocol: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """Reconstruct the formal evaluator's cache reader for one frozen scene."""

    xyz, colors, labels_full = _read_official_ply(benchmark_root / "scans" / f"{scene_id}.ply")
    voxel_size = float(protocol["voxel_size_m"])
    quantized = quantize_scannet_points(xyz, colors, labels_full, voxel_size=voxel_size)
    cache = _load_feature_cache(
        feature_root / f"{scene_id}.npz",
        xyz,
        quantized_unique_map=quantized.unique_map,
    )
    cache_rows = slice(None) if bool(cache["_is_quantized"]) else quantized.unique_map
    cache_valid = np.asarray(
        cache.get("valid", np.ones(cache["radio_features"].shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)[cache_rows]
    if cache_valid.shape != (len(quantized.raw_coordinates),):
        raise ValueError(f"{scene_id} feature validity does not match quantized rows")
    predictor = CanonicalPointPredictor(
        quantized.raw_coordinates,
        cache["radio_features"][cache_rows],
        appearance_features=(
            cache["appearance_features"][cache_rows]
            if "appearance_features" in cache
            else None
        ),
        boundary_features=(
            cache["boundary_features"][cache_rows]
            if "boundary_features" in cache
            else None
        ),
        observation_valid=cache_valid,
        device=device,
        # These were the unchanged defaults used by the formal evaluator.
        graph_config=SupportGraphConfig(neighbors=16, topology_mode="symmetric_union"),
        solver_config=SupportSolverConfig(
            solver_type="confidence_random_walker",
            laplacian_weight=1.0,
            cg_iterations=64,
            support_threshold=0.50,
            hard_seed_threshold=0.20,
        ),
        selection_mode=SelectionMode(str(protocol["selection_mode"])),
        unary_mode=str(protocol["unary_mode"]),
        appearance_unary_weight=float(protocol["appearance_unary_weight"]),
        boundary_unary_weight=float(protocol["boundary_unary_weight"]),
        observation_lift_mode=str(protocol["observation_lift_mode"]),
        observation_lift_neighbors=int(protocol["observation_lift_neighbors"]),
        observation_lift_maximum_distance_m=float(
            protocol["observation_lift_maximum_distance_m"]
        ),
    )
    return xyz, labels_full, cache_valid, quantized, predictor


def replay_agile_visual_case(
    case: Mapping[str, Any],
    *,
    benchmark_root: str | Path,
    feature_root: str | Path,
    protocol: Mapping[str, Any],
    device: str,
    capture_click_counts: Sequence[int] = (1, 5, 15),
) -> dict[str, Any]:
    """Replay and validate one formal AGILE3D object while retaining masks."""

    scene_id = str(case["scene_id"])
    xyz, labels_full, cache_valid, quantized, predictor = _formal_agile_predictor(
        scene_id=scene_id,
        benchmark_root=Path(benchmark_root),
        feature_root=Path(feature_root),
        protocol=protocol,
        device=device,
    )
    try:
        target_full = labels_full == int(case["object_id"])
        target_quantized = quantized.labels == int(case["object_id"])
        replay = replay_interactive_masks(
            quantized.raw_coordinates,
            target_quantized,
            target_full,
            quantized.inverse_map,
            predictor,
            max_clicks=int(protocol["max_clicks"]),
            capture_click_counts=capture_click_counts,
            click_workers=int(protocol.get("click_search_workers", 2)),
        )
        assert_trajectory_matches(case["trajectory"], replay["trajectory"], atol=1e-6)
        return {
            "case": dict(case),
            "quantized_xyz": quantized.raw_coordinates,
            "quantized_rgb": np.clip(
                np.rint(quantized.colors * 255.0), 0, 255
            ).astype(np.uint8),
            "target": target_quantized,
            "snapshots": replay["snapshots"],
            "trajectory": replay["trajectory"],
            "feature_coverage": float(cache_valid.mean()),
            "projectable_fraction": float(predictor.observation_lift_report()["projectable_fraction"]),
        }
    finally:
        del predictor
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap_text(value: str, maximum_characters: int) -> str:
    lines: list[str] = []
    for raw_line in str(value).splitlines() or [""]:
        words = raw_line.split(" ")
        current = ""
        for word in words:
            if not word:
                continue
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= maximum_characters:
                current = candidate
                continue
            if current:
                lines.append(current)
            while len(word) > maximum_characters:
                lines.append(word[:maximum_characters])
                word = word[maximum_characters:]
            current = word
        lines.append(current)
    return "\n".join(line for line in lines if line) or " "


def _draw_panel(
    canvas: Image.Image,
    image: Image.Image,
    *,
    x: int,
    y: int,
    size: tuple[int, int],
    title: str,
    subtitle: str = "",
) -> None:
    width, height = map(int, size)
    draw = ImageDraw.Draw(canvas)
    title_font, small_font = _font(20), _font(15)
    frame = Image.new("RGB", (width, height), "white")
    draw_frame = ImageDraw.Draw(frame)
    draw_frame.rectangle((0, 0, width - 1, height - 1), outline=(190, 195, 202), width=1)
    text_height = 60 if subtitle else 40
    image_box = (width - 14, height - text_height - 12)
    fitted = ImageOps.contain(image.convert("RGB"), image_box, method=Image.Resampling.LANCZOS)
    frame.paste(fitted, ((width - fitted.width) // 2, text_height + (image_box[1] - fitted.height) // 2))
    draw_frame.multiline_text(
        (8, 7), _wrap_text(title, 33), fill=(20, 25, 33), font=title_font, spacing=2
    )
    if subtitle:
        draw_frame.multiline_text(
            (8, 33), _wrap_text(subtitle, 45), fill=(75, 82, 92), font=small_font, spacing=1
        )
    canvas.paste(frame, (int(x), int(y)))


def _draw_legend(canvas: Image.Image, *, y: int, include_clicks: bool) -> None:
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    values: list[tuple[str, tuple[int, int, int]]] = [
        ("TP", TP_COLOR),
        ("FP", FP_COLOR),
        ("FN", FN_COLOR),
        ("background", BG_COLOR),
    ]
    if include_clicks:
        values.extend((("positive click", POSITIVE_CLICK_COLOR), ("negative click", NEGATIVE_CLICK_COLOR)))
    x = 24
    for label, color in values:
        draw.rectangle((x, y + 3, x + 17, y + 20), fill=color, outline=(80, 80, 80))
        draw.text((x + 23, y + 2), label, fill=(36, 42, 50), font=font)
        x += 23 + int(draw.textlength(label, font=font)) + 22


def _compose_figure(
    title: str,
    rows: Sequence[tuple[str, Sequence[tuple[Image.Image, str, str]]]],
    *,
    panel_size: tuple[int, int],
    include_clicks: bool,
) -> Image.Image:
    if not rows or not rows[0][1]:
        raise ValueError("a qualitative figure needs at least one populated row")
    width, height = map(int, panel_size)
    columns = max(len(panels) for _caption, panels in rows)
    padding, title_height, caption_height, row_gap, footer = 18, 52, 48, 12, 42
    total_width = padding * (columns + 1) + width * columns
    total_height = title_height + footer + sum(
        caption_height + height for _caption, _panels in rows
    ) + row_gap * (len(rows) - 1)
    canvas = Image.new("RGB", (total_width, total_height), (250, 251, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 14), title, fill=(15, 23, 42), font=_font(27))
    y = title_height
    for row_caption, panels in rows:
        draw.multiline_text((padding, y + 5), _wrap_text(row_caption, 180), fill=(48, 56, 66), font=_font(17))
        y += caption_height
        for column, (panel, panel_title, panel_subtitle) in enumerate(panels):
            _draw_panel(
                canvas,
                panel,
                x=padding + column * (width + padding),
                y=y,
                size=panel_size,
                title=panel_title,
                subtitle=panel_subtitle,
            )
        y += height + row_gap
    _draw_legend(canvas, y=total_height - footer + 9, include_clicks=include_clicks)
    return canvas


def build_pfir_figure(visual_cases: Sequence[Mapping[str, Any]]) -> Image.Image:
    """Compose the PFIR input/GT/prediction/error comparison figure."""

    rows: list[tuple[str, list[tuple[Image.Image, str, str]]]] = []
    for item in visual_cases:
        case, query = item["case"], item["query"]
        xyz, rgb = item["xyz"], item["rgb"]
        target, prediction = item["target"], item["prediction"]
        iou = float(case["iou"])
        rank = int(case["rank"])
        caption = (
            f"{str(case['kind']).replace('_', ' ')} | {case['query_id']} | "
            f"Track-A target rank {rank}; Track-B 3D IoU {iou:.3f}"
        )
        panels = [
            (item["crop"], "Held-out RGB crop", "method-visible query input"),
            (
                render_mask_view(xyz, rgb, target, prediction, overlay="rgb"),
                "3D scene RGB",
                "official annotation mesh geometry",
            ),
            (
                render_mask_view(
                    xyz, rgb, target, prediction, overlay="gt", focus_mask=target
                ),
                "Ground-truth target (zoom)",
                "evaluator-only target-centered view",
            ),
            (
                render_mask_view(
                    xyz,
                    rgb,
                    target,
                    prediction,
                    overlay="prediction",
                    focus_mask=target,
                ),
                "Track-B predicted mask (zoom)",
                "frozen output; evaluator-only target-centered view",
            ),
            (
                render_mask_view(
                    xyz, rgb, target, prediction, overlay="error", focus_mask=target
                ),
                "Prediction vs. GT (zoom)",
                f"evaluator-only target-centered; IoU={iou:.3f}; rank={rank}",
            ),
        ]
        rows.append((caption, panels))
    return _compose_figure(
        "ScanNet-PFIR-Small v1: pose-free real-image exemplar query",
        rows,
        panel_size=(500, 350),
        include_clicks=False,
    )


def build_agile3d_figure(visual_cases: Sequence[Mapping[str, Any]]) -> Image.Image:
    """Compose exact AGILE3D click-replay mask panels at 1, 5, and 15 clicks."""

    rows: list[tuple[str, list[tuple[Image.Image, str, str]]]] = []
    for item in visual_cases:
        case = item["case"]
        xyz, rgb, target = item["quantized_xyz"], item["quantized_rgb"], item["target"]
        snapshots = item["snapshots"]
        caption = (
            f"{str(case['kind']).replace('_', ' ')} | {case['scene_id']} object {case['object_id']} "
            f"({case['semantic_class']}) | feature coverage {item['feature_coverage']:.3f}, "
            f"projectable {item['projectable_fraction']:.3f}"
        )
        empty = np.zeros_like(target)
        panels: list[tuple[Image.Image, str, str]] = [
            (
                render_mask_view(xyz, rgb, target, empty, overlay="rgb"),
                "5 cm scene RGB",
                "official interaction domain",
            ),
            (
                render_mask_view(
                    xyz, rgb, target, empty, overlay="gt", focus_mask=target
                ),
                "Ground-truth target (zoom)",
                "evaluator-only target-centered view",
            ),
        ]
        for click_count in (1, 5, 15):
            snapshot = snapshots[click_count]
            prediction = snapshot["prediction"]
            clicks = snapshot["clicks"]
            if click_count == 1:
                panel = render_mask_view(
                    xyz,
                    rgb,
                    target,
                    prediction,
                    overlay="prediction",
                    clicks=clicks,
                    focus_mask=target,
                )
                title = "Prediction after 1 click (zoom)"
                subtitle = (
                    f"IoU={item['trajectory'][1]:.3f}; evaluator-only target-centered view"
                )
            else:
                panel = render_mask_view(
                    xyz,
                    rgb,
                    target,
                    prediction,
                    overlay="error",
                    clicks=clicks,
                    focus_mask=target,
                )
                title = f"Error after {click_count} clicks (zoom)"
                subtitle = (
                    f"evaluator-only target-centered; IoU={item['trajectory'][click_count]:.3f}"
                )
            panels.append((panel, title, subtitle))
        rows.append((caption, panels))
    return _compose_figure(
        "AGILE3D ScanNet40: official corrective-click replay",
        rows,
        panel_size=(420, 320),
        include_clicks=True,
    )


def _case_summary(case: Mapping[str, Any], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={case[key]}" for key in keys if key in case)


def build_markdown(audit: Mapping[str, Any]) -> str:
    """Explain the visualization provenance and what each panel establishes."""

    pfir_cases = list(audit.get("pfir", {}).get("cases", []))
    agile_cases = list(audit.get("agile3d", {}).get("cases", []))
    lines = [
        "# PFIR and AGILE3D Qualitative Mask Evidence",
        "",
        "This folder contains static visualizations generated from the frozen formal evaluation artifacts. No model, threshold, click policy, feature cache, or test-set calibration was changed to create these figures.",
        "",
        "![PFIR mask comparison](pfir_mask_comparison.png)",
        "",
        "![AGILE3D click replay](agile3d_click_replay.png)",
        "",
        "## How to Read the Panels",
        "",
        "PFIR exposes only the held-out RGB crop to the method. The 3D target mask and TP/FP/FN overlay are evaluator-only visual evidence on the official ScanNet annotation mesh. The prediction is the saved Track-B boolean mask that was already evaluated in the formal run; Track-A rank is displayed only as an accompanying ranking metric.",
        "",
        "AGILE3D panels use the released 5 cm interaction point cloud. GT labels are evaluator-only. Positive and negative click markers are produced by the released corrective-click simulator, while the 1/5/15-click masks are replayed with the same cached canonical features and formal query configuration. Each selected replay is checked against its full stored 20-click trajectory before it is rendered.",
        "",
        "Each row retains a full-scene RGB panel. The remaining GT/prediction/error panels are target-centered crops of that same fixed geometry-only projection. Their crop center comes from the evaluator-only GT target solely to make small objects legible; it is not a method input, camera-selection rule, query change, or test-set optimization.",
        "",
        "## PFIR Cases",
        "",
    ]
    for case in pfir_cases:
        lines.append(f"- `{case.get('kind')}`: {_case_summary(case, ('query_id', 'scene_id', 'rank', 'iou'))}")
    lines.extend(("", "## AGILE3D Cases", ""))
    for case in agile_cases:
        lines.append(
            f"- `{case.get('kind')}`: {_case_summary(case, ('scene_id', 'object_id', 'semantic_class', 'feature_coverage', 'projectable_fraction', 'iou_at_15'))}"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The PFIR rows separate candidate identification from full-mask recovery: a rank-1 result can still yield an empty or spatially incorrect Track-B mask. The AGILE3D rows make the interaction contract visible: additional corrective clicks can repair supported regions, while low registered-observation coverage can leave parts of the official point cloud outside the canonical feature domain. These are qualitative explanations of this method's formal run, not comparisons against PLY-only AGILE3D baselines with a different input modality.",
            "",
            "The exact selected records, source hashes, replay settings, and trajectory checks are in `case_selection_audit.json`.",
            "",
        )
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: str | Path,
    pfir_figure: Image.Image,
    agile_figure: Image.Image,
    audit: Mapping[str, Any],
) -> dict[str, Path]:
    """Publish PNGs and their text provenance together in one output directory."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "pfir": root / "pfir_mask_comparison.png",
        "agile3d": root / "agile3d_click_replay.png",
        "markdown": root / "README.md",
        "audit": root / "case_selection_audit.json",
    }
    for key, figure in (("pfir", pfir_figure), ("agile3d", agile_figure)):
        temporary = paths[key].with_name(f".{paths[key].name}.tmp")
        figure.save(temporary, format="PNG")
        temporary.replace(paths[key])
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(build_markdown(audit), encoding="utf-8")
    return paths


def _pfir_audit_case(item: Mapping[str, Any]) -> dict[str, Any]:
    case = item["case"]
    query = item["query"]
    return {
        "kind": case["kind"],
        "query_id": case["query_id"],
        "scene_id": case["scene_id"],
        "difficulty": query["difficulty"],
        "target_instance_id": int(query["instance_id_3d"]),
        "target_nyu40_id": int(query["nyu40_class_id"]),
        "track_a_rank": int(case["rank"]),
        "track_b_iou": float(case["iou"]),
        "same_category": bool(case["same_category"]),
        "crop_path": item["crop_path"],
        "prediction_path": item["prediction_path"],
        "mesh_path": item["mesh_path"],
    }


def _agile_audit_case(item: Mapping[str, Any]) -> dict[str, Any]:
    case = item["case"]
    trajectory = {str(key): float(value) for key, value in item["trajectory"].items()}
    return {
        "kind": case["kind"],
        "scene_id": case["scene_id"],
        "object_id": int(case["object_id"]),
        "semantic_class": case["semantic_class"],
        "feature_coverage": float(item["feature_coverage"]),
        "projectable_fraction": float(item["projectable_fraction"]),
        "iou_at_1": trajectory["1"],
        "iou_at_5": trajectory["5"],
        "iou_at_15": trajectory["15"],
        "formal_trajectory_verified": True,
        "click_count": len(item["snapshots"][15]["clicks"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Generate qualitative evidence without modifying the frozen evaluators."""

    pfir_benchmark = Path(args.pfir_benchmark_root)
    pfir_run = Path(args.pfir_run_root)
    agile_results = Path(args.agile_results)
    pfir_ranking_path = pfir_run / "track_a_ranking.json"
    pfir_selection_path = pfir_run / "track_b_selection.json"
    ranking = _read_json(pfir_ranking_path)
    selection = _read_json(pfir_selection_path)
    if ranking.get("test_calibration", False) or selection.get("test_calibration", False):
        raise ValueError("PFIR qualitative report refuses test-calibrated results")
    pfir_selected = select_pfir_cases(ranking["per_query"], selection["per_query"])
    pfir_visual = [
        load_pfir_visual_case(
            case,
            benchmark_root=pfir_benchmark,
            prediction_root=pfir_run / "predictions" / "selection",
            annotations_root=args.pfir_annotations_root,
        )
        for case in pfir_selected
    ]

    agile = _read_json(agile_results)
    protocol = dict(agile.get("protocol", {}))
    if agile.get("benchmark") != "AGILE3D ScanNet40 single-object":
        raise ValueError("AGILE report is not the released single-object benchmark")
    if protocol.get("test_set_calibration") is not False:
        raise ValueError("AGILE qualitative report requires test_set_calibration=false")
    agile_selected = select_agile_cases(agile["rows"], agile["scene_coverage"])
    agile_visual = [
        replay_agile_visual_case(
            case,
            benchmark_root=args.agile_benchmark_root,
            feature_root=args.agile_feature_root,
            protocol=protocol,
            device=args.device,
        )
        for case in agile_selected
    ]
    audit: dict[str, Any] = {
        "schema_version": 2,
        "selection_is_frozen_result_only": True,
        "test_set_calibration": False,
        "rendering": {
            "camera": "geometry_only_pca_orthographic",
            "full_scene_rgb_panels": True,
            "mask_panel_crop": {
                "type": "evaluator_only_target_centered_fixed_projection_crop",
                "context_scale": FOCUS_CONTEXT_SCALE,
                "minimum_panel_fraction": FOCUS_MINIMUM_FRACTION,
                "changes_method_input_or_prediction": False,
            },
        },
        "pfir": {
            "benchmark": "ScanNet-PFIR-Small v1",
            "source_files": {
                "manifest": str((pfir_benchmark / "manifest.internal.json").resolve()),
                "ranking": str(pfir_ranking_path.resolve()),
                "selection": str(pfir_selection_path.resolve()),
            },
            "source_sha256": {
                "manifest": _sha256(pfir_benchmark / "manifest.internal.json"),
                "ranking": _sha256(pfir_ranking_path),
                "selection": _sha256(pfir_selection_path),
            },
            "cases": [_pfir_audit_case(item) for item in pfir_visual],
        },
        "agile3d": {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "formal_results": str(agile_results.resolve()),
            "formal_results_sha256": _sha256(agile_results),
            "feature_root": str(Path(args.agile_feature_root).resolve()),
            "replay_device": str(args.device),
            "replay_configuration": {
                "graph_neighbors": 16,
                "topology_mode": "symmetric_union",
                "solver_type": "confidence_random_walker",
                "laplacian_weight": 1.0,
                "cg_iterations": 64,
                "support_threshold": 0.50,
                "capture_click_counts": [1, 5, 15],
                "formal_protocol": protocol,
            },
            "cases": [_agile_audit_case(item) for item in agile_visual],
        },
    }
    paths = write_outputs(
        args.output_dir,
        build_pfir_figure(pfir_visual),
        build_agile3d_figure(agile_visual),
        audit,
    )
    return {"paths": {key: str(value.resolve()) for key, value in paths.items()}, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pfir-benchmark-root",
        default="output/scannet_pfir_small_v1/test_v1_final",
    )
    parser.add_argument(
        "--pfir-run-root",
        default="output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1",
    )
    parser.add_argument(
        "--pfir-annotations-root",
        default="/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations",
    )
    parser.add_argument(
        "--agile-results",
        default="output/agile3d_scannet40/formal_v1/results.json",
    )
    parser.add_argument(
        "--agile-benchmark-root",
        default="/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet",
    )
    parser.add_argument(
        "--agile-feature-root",
        default="output/agile3d_scannet40/formal_v1/features",
    )
    parser.add_argument("--output-dir", default="output/benchmark_qualitative_report")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Replay device; defaults to CPU so reporting does not reserve a GPU.",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["paths"], indent=2))


if __name__ == "__main__":
    main()
