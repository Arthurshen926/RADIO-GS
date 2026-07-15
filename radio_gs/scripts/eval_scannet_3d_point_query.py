#!/usr/bin/env python3
"""Evaluate a fixed 3-D point -> 3-D instance-mask query protocol on ScanNet.

One ground-truth vertex is sampled deterministically per instance to construct
the point query.  The method receives only that vertex index/coordinate; the
remaining instance mask is opened only by the metric code.  Region features
come from raw RADIO or an explicitly declared frozen official adaptor space.
No threshold or propagation parameter is fitted on evaluation ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from radio_gs.config import load_config
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import EvidenceScoringConfig
from radio_gs.querying.query_compilers import (
    compile_world_3d_query,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.support_solver import SupportSolverConfig
from radio_gs.querying.unified_query import (
    QueryKind,
    QuerySpace,
    QuerySpec,
    SupportPropagationConfig,
    binary_mask,
    build_support_graph,
    propagate_support,
    seed_connected_component,
)
from radio_gs.querying.query_spec import SoftSeedSet
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
    _decode_points_1280,
    _read_label_ply,
)


def _bilinear_sample_2d(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample one ``[V,H,W]`` stack at one coordinate per view."""

    maps = np.asarray(values, dtype=np.float32)
    if maps.ndim != 3 or x.shape != (maps.shape[0],) or y.shape != x.shape:
        raise ValueError("values/x/y must be [V,H,W], [V], and [V]")
    height, width = maps.shape[-2:]
    x0 = np.floor(x).astype(np.int64).clip(0, width - 1)
    y0 = np.floor(y).astype(np.int64).clip(0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    rows = np.arange(maps.shape[0])
    return (
        maps[rows, y0, x0] * (1.0 - wx) * (1.0 - wy)
        + maps[rows, y0, x1] * wx * (1.0 - wy)
        + maps[rows, y1, x0] * (1.0 - wx) * wy
        + maps[rows, y1, x1] * wx * wy
    ).astype(np.float32)


def select_depth_visible_views(
    point_xyz: np.ndarray,
    poses_w2c: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    alpha_maps: np.ndarray,
    *,
    max_views: int,
    depth_tolerance: float,
    relative_depth_tolerance: float,
    alpha_threshold: float,
) -> list[dict[str, float | int]]:
    """Choose depth-consistent rendered views without opening instance GT."""

    point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    poses = np.asarray(poses_w2c, dtype=np.float32)
    depths = np.asarray(depth_maps, dtype=np.float32)
    alphas = np.asarray(alpha_maps, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses_w2c must be [V,4,4]")
    if depths.shape != alphas.shape or depths.shape[0] != poses.shape[0]:
        raise ValueError("depth/alpha maps must align with views")
    if K.shape != (3, 3) or max_views <= 0:
        raise ValueError("intrinsics must be [3,3] and max_views positive")
    height, width = depths.shape[-2:]
    point_h = np.concatenate([point, np.ones(1, dtype=np.float32)])
    camera = np.einsum("vij,j->vi", poses, point_h)
    z = camera[:, 2]
    safe_z = np.maximum(z, 1e-6)
    u = K[0, 0] * camera[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * camera[:, 1] / safe_z + K[1, 2]
    in_frame = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= width - 1)
        & (v >= 0.0)
        & (v <= height - 1)
    )
    sampled_depth = _bilinear_sample_2d(depths, u, v)
    sampled_alpha = _bilinear_sample_2d(alphas, u, v)
    tolerance = np.maximum(
        float(depth_tolerance), np.abs(z) * float(relative_depth_tolerance)
    )
    depth_error = np.abs(sampled_depth - z)
    valid = (
        in_frame
        & (sampled_depth > 0.0)
        & (depth_error <= tolerance)
        & (sampled_alpha >= float(alpha_threshold))
    )
    if not bool(valid.any()):
        return []
    # Prefer a central, confidently opaque, depth-consistent observation.  All
    # terms are query-observable and the final view index is only a stable tie
    # break; labels/masks never participate.
    horizontal_margin = np.minimum(u, width - 1 - u) / max(0.5 * (width - 1), 1.0)
    vertical_margin = np.minimum(v, height - 1 - v) / max(0.5 * (height - 1), 1.0)
    centrality = np.minimum(horizontal_margin, vertical_margin).clip(0.0, 1.0)
    depth_quality = (1.0 - depth_error / np.maximum(tolerance, 1e-6)).clip(0.0, 1.0)
    quality = centrality + depth_quality + 0.25 * sampled_alpha.clip(0.0, 1.0)
    valid_indices = np.flatnonzero(valid)
    ordered = sorted(valid_indices.tolist(), key=lambda index: (-quality[index], index))
    return [
        {
            "view_index": int(index),
            "u": float(u[index]),
            "v": float(v[index]),
            "depth": float(z[index]),
            "depth_error": float(depth_error[index]),
            "alpha": float(sampled_alpha[index]),
            "quality": float(quality[index]),
        }
        for index in ordered[: int(max_views)]
    ]


def choose_official_sam3_point_mask(
    masks: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Use the official predicted-IoU score to resolve SAM3 multimasks."""

    candidates = np.asarray(masks)
    scores = np.asarray(quality, dtype=np.float32).reshape(-1)
    if candidates.ndim != 3 or candidates.shape[0] != scores.size or scores.size == 0:
        raise ValueError("SAM3 masks/scores must be aligned non-empty [M,H,W]/[M]")
    index = int(np.argmax(scores))
    return candidates[index].astype(bool), index, float(scores[index])


def load_scannet_instances(
    aggregation_path: str | Path,
    segmentation_path: str | Path,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Map ScanNet mesh vertices to ``objectId + 1`` instance IDs."""

    with Path(segmentation_path).open("r", encoding="utf-8") as handle:
        segmentation = json.load(handle)
    with Path(aggregation_path).open("r", encoding="utf-8") as handle:
        aggregation = json.load(handle)
    segment_ids = np.asarray(segmentation.get("segIndices", []), dtype=np.int64)
    if segment_ids.ndim != 1 or segment_ids.size == 0:
        raise ValueError("ScanNet segmentation must contain a non-empty segIndices array")
    instance_ids = np.zeros(segment_ids.shape[0], dtype=np.int32)
    metadata: dict[int, dict[str, Any]] = {}
    for group in aggregation.get("segGroups", []):
        instance_id = int(group["objectId"]) + 1
        segments = np.asarray(group.get("segments", []), dtype=np.int64)
        if segments.size == 0:
            continue
        selected = np.isin(segment_ids, segments)
        if bool((instance_ids[selected] != 0).any()):
            raise ValueError(f"Overlapping ScanNet segment groups for instance {instance_id}")
        instance_ids[selected] = instance_id
        metadata[instance_id] = {
            "object_id": int(group["objectId"]),
            "label": str(group.get("label", "")),
            "num_vertices": int(selected.sum()),
        }
    if not metadata:
        raise ValueError("ScanNet aggregation contains no non-empty segGroups")
    return instance_ids, metadata


def _intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 0.0


def _local_multiscale_positive_prototypes(
    xyz: np.ndarray,
    normalized_features: np.ndarray,
    seed_indices: tuple[int, ...],
    tree: cKDTree,
    *,
    neighbors: int,
) -> np.ndarray:
    """Expand clicks into fixed, label-free local appearance prototypes.

    Radii are multiples of the local mesh spacing, rather than benchmark-sized
    metric constants.  At each scale only the feature-coherent half of the
    geometric neighborhood contributes, limiting leakage across an immediate
    object boundary.  The clicked feature is always retained as its own
    prototype, so this is a conservative extension of the one-point readout.
    """
    normalized = np.asarray(normalized_features, dtype=np.float32)
    count = min(max(9, int(neighbors) + 1), xyz.shape[0])
    prototypes: list[np.ndarray] = []
    for seed_index in seed_indices:
        distances, indices = tree.query(xyz[seed_index], k=count)
        distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        valid_distance = distances[1 : min(9, distances.size)]
        valid_distance = valid_distance[np.isfinite(valid_distance) & (valid_distance > 0)]
        spacing = float(np.median(valid_distance)) if valid_distance.size else 1e-3
        seed_feature = normalized[seed_index]
        prototypes.append(seed_feature)
        similarities = normalized[indices] @ seed_feature
        for multiplier in (2.0, 4.0, 8.0):
            radius = max(multiplier * spacing, 1e-6)
            local = np.isfinite(distances) & (distances <= radius)
            if int(local.sum()) <= 1:
                continue
            local_similarity = similarities[local]
            coherent = local & (similarities >= np.median(local_similarity))
            local_indices = indices[coherent]
            local_distances = distances[coherent]
            weights = np.exp(-0.5 * np.square(local_distances / radius)).astype(np.float32)
            prototype = np.sum(normalized[local_indices] * weights[:, None], axis=0)
            norm = float(np.linalg.norm(prototype))
            if norm > 1e-12:
                prototypes.append(prototype / norm)
    return np.ascontiguousarray(np.stack(prototypes), dtype=np.float32)


def evaluate_point_queries(
    xyz: np.ndarray,
    region_features: np.ndarray,
    instance_ids: np.ndarray,
    instance_metadata: Mapping[int, Mapping[str, Any]],
    *,
    graph_features: np.ndarray | None = None,
    random_seed: int,
    min_instance_points: int,
    max_instances: int | None,
    propagation: SupportPropagationConfig,
    threshold: float,
    component_radius: float,
    clicks: int = 1,
    positive_mode: str = "single",
    local_prototype_neighbors: int = 64,
) -> dict[str, Any]:
    """Run deterministic point queries; GT is used only for query sampling/metrics."""

    xyz = np.asarray(xyz, dtype=np.float32)
    features = np.asarray(region_features, dtype=np.float32)
    instances = np.asarray(instance_ids, dtype=np.int32).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or features.ndim != 2:
        raise ValueError("xyz/features must be [N,3] and [N,D]")
    if xyz.shape[0] != features.shape[0] or instances.shape != (xyz.shape[0],):
        raise ValueError("xyz, region features, and instance IDs must align")
    eligible = [
        int(instance_id)
        for instance_id in sorted(instance_metadata)
        if int((instances == int(instance_id)).sum()) >= int(min_instance_points)
    ]
    if max_instances is not None and max_instances > 0:
        eligible = eligible[: int(max_instances)]
    if not eligible:
        raise ValueError("No ScanNet instances satisfy min_instance_points")
    if clicks <= 0:
        raise ValueError("clicks must be positive")
    if positive_mode not in {"single", "local_multiscale"}:
        raise ValueError("positive_mode must be single or local_multiscale")

    pairwise_features = (
        features
        if graph_features is None
        else np.asarray(graph_features, dtype=np.float32)
    )
    if pairwise_features.ndim != 2 or pairwise_features.shape[0] != xyz.shape[0]:
        raise ValueError("graph_features must be [N,Dg] and align with xyz")
    graph = build_support_graph(xyz, pairwise_features, propagation)
    spatial_tree = cKDTree(xyz) if positive_mode == "local_multiscale" else None
    normalized_features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12
    )
    background_prototype = features.mean(axis=0)
    background_prototype /= max(float(np.linalg.norm(background_prototype)), 1e-12)
    background_similarity = normalized_features @ background_prototype
    rows: list[dict[str, Any]] = []
    for instance_id in eligible:
        target = instances == instance_id
        candidates = np.flatnonzero(target)
        # A per-instance RNG makes the sampled point stable if other instances
        # are later filtered from the benchmark.
        rng = np.random.default_rng(int(random_seed) + 1_000_003 * instance_id)
        if int(clicks) == 1:
            # Preserve the original frozen one-click sampling exactly.
            selected = np.asarray([candidates[int(rng.integers(0, candidates.size))]])
        else:
            selected = rng.choice(
                candidates, size=min(int(clicks), candidates.size), replace=False
            )
        seed_indices = tuple(int(value) for value in np.asarray(selected).reshape(-1))
        seed_index = seed_indices[0]
        positive_prototypes = features[np.asarray(seed_indices, dtype=np.int64)]
        if positive_mode == "local_multiscale":
            assert spatial_tree is not None
            positive_prototypes = _local_multiscale_positive_prototypes(
                xyz,
                normalized_features,
                seed_indices,
                spatial_tree,
                neighbors=local_prototype_neighbors,
            )
        query = QuerySpec(
            kind=QueryKind.POINT_3D,
            space=QuerySpace.REGION,
            positive_prototypes=positive_prototypes,
            negative_prototypes=background_prototype,
            positive_seed_indices=seed_indices,
            metadata={
                "source": "registered_world_points",
                "positive_mode": positive_mode,
            },
        )
        raw_scores = np.ascontiguousarray(
            (normalized_features @ query.positive_prototypes.T).max(axis=1)
            - background_similarity,
            dtype=np.float32,
        )
        propagated_scores = propagate_support(
            xyz,
            pairwise_features,
            raw_scores,
            query,
            propagation,
            graph=graph,
        )
        raw_prediction = binary_mask(raw_scores, threshold=threshold)
        propagated_prediction = binary_mask(propagated_scores, threshold=threshold)
        connected_prediction = np.zeros_like(propagated_prediction, dtype=bool)
        for component_seed in seed_indices:
            connected_prediction |= seed_connected_component(
                propagated_prediction,
                component_seed,
                graph,
                max_edge_distance=component_radius,
            )
        meta = instance_metadata.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "label": str(meta.get("label", "")),
                "seed_index": seed_index,
                "seed_indices": list(seed_indices),
                "seed_xyz": xyz[seed_index].tolist(),
                "num_gt_points": int(target.sum()),
                "raw_iou": _intersection_over_union(raw_prediction, target),
                "propagated_iou": _intersection_over_union(propagated_prediction, target),
                "connected_iou": _intersection_over_union(connected_prediction, target),
                "num_predicted_points": int(connected_prediction.sum()),
            }
        )
    connected_ious = np.asarray([row["connected_iou"] for row in rows], dtype=np.float64)
    return {
        "num_queries": len(rows),
        "clicks_per_query": int(clicks),
        "positive_mode": positive_mode,
        "macro_raw_iou": float(np.mean([row["raw_iou"] for row in rows])),
        "macro_propagated_iou": float(np.mean([row["propagated_iou"] for row in rows])),
        "macro_connected_iou": float(connected_ious.mean()),
        "accuracy_at_025": float((connected_ious >= 0.25).mean()),
        "accuracy_at_050": float((connected_ious >= 0.50).mean()),
        "queries": rows,
    }


def extract_region_features(
    config_path: str,
    checkpoint_path: str,
    radio_checkpoint: str,
    *,
    device: torch.device,
    chunk_size: int,
    query_xyz: np.ndarray | None = None,
    query_k: int = 8,
    query_candidate_k: int = 80,
    region_space: str = "sam3",
    mesh_query_mode: str = "row_latent_at_mesh",
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the field in raw RADIO or an official frozen adaptor space."""

    config = load_config(config_path)
    model, codec = _build_hybrid_model(config, checkpoint_path, device)
    supported = {"radio_raw", "sam3", "dino_v3_7b", "sam3_dino_equal_fusion"}
    if region_space not in supported:
        raise ValueError(f"region_space must be one of {sorted(supported)}")
    adaptors: dict[str, torch.nn.Module] = {}
    requested = (
        ("sam3", "dino_v3_7b")
        if region_space == "sam3_dino_equal_fusion"
        else ()
        if region_space == "radio_raw"
        else (region_space,)
    )
    for name in requested:
        adaptors[name] = load_radio_adaptor_from_checkpoint(
            radio_checkpoint, name, kind="feature_projection"
        ).to(device).eval().requires_grad_(False)
    xyz = (
        model.get_xyz().detach().float().cpu().numpy()
        if query_xyz is None
        else np.asarray(query_xyz, dtype=np.float32)
    )
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, xyz.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), xyz.shape[0])
            if query_xyz is None:
                indices = torch.arange(start, stop, device=device, dtype=torch.long)
                decoded = _decode_gaussian_indices_1280(model, codec, indices)
            else:
                points = torch.from_numpy(xyz[start:stop]).to(device=device)
                if mesh_query_mode == "row_latent_at_mesh":
                    indices = torch.arange(start, stop, device=device, dtype=torch.long)
                    decoded = _decode_gaussian_indices_1280(
                        model, codec, indices, points_xyz=points
                    )
                elif mesh_query_mode == "knn_field":
                    decoded = _decode_points_1280(
                        model,
                        codec,
                        points,
                        int(query_k),
                        candidate_k=int(query_candidate_k),
                    )
                else:
                    raise ValueError(
                        "mesh_query_mode must be row_latent_at_mesh or knn_field"
                    )
            decoded = decoded.float()
            if region_space == "radio_raw":
                projected = decoded
            elif region_space == "sam3_dino_equal_fusion":
                sam = torch.nn.functional.normalize(adaptors["sam3"](decoded).float(), dim=-1)
                dino = torch.nn.functional.normalize(
                    adaptors["dino_v3_7b"](decoded).float(), dim=-1
                )
                projected = torch.cat([sam, dino], dim=-1) / np.sqrt(2.0)
            else:
                projected = adaptors[region_space](decoded).float()
            parts.append(projected.cpu().numpy())
    return xyz, np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def _gaussian_covariances(model) -> torch.Tensor:
    """Return full world-space covariance matrices for fixed 3D Gaussians."""

    rotation = model._quaternion_to_rotation_matrix(model.get_rotation().float())
    scale = model.get_scaling().float().clamp_min(1e-6)
    return rotation @ torch.diag_embed(scale.square()) @ rotation.transpose(1, 2)


def _load_or_build_rendered_view_context(
    args: argparse.Namespace,
    *,
    device: torch.device,
    expected_xyz: torch.Tensor,
) -> dict[str, Any]:
    """Prepare query-independent RGB/depth views for world-point prompting."""

    if not args.config or not args.checkpoint:
        raise ValueError("rendered SAM3 point readout requires --config and --checkpoint")
    from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
        SimpleRadioDataset,
    )
    from radio_gs.scripts.eval_lerf_direct_3d_selection import (
        build_mask_renderer,
    )
    from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline

    model, _codec, low_renderer, _sharpener, _refiner, config, _is_hybrid = (
        load_render_pipeline(
            args.config,
            args.checkpoint,
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    model_xyz = model.get_xyz().detach().float().cpu()
    if model_xyz.shape != expected_xyz.shape or not torch.allclose(
        model_xyz, expected_xyz.float().cpu(), atol=1e-6, rtol=0.0
    ):
        raise ValueError("rendered-view Gaussian geometry does not match canonical bank")
    feature_height = int(getattr(config, "feature_height", low_renderer.image_height))
    feature_width = int(getattr(config, "feature_width", low_renderer.image_width))
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=raw_pose_file or None,
        pose_dir=raw_pose_dir or None,
        feature_size=(feature_height, feature_width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "scannet")),
    )
    poses = torch.from_numpy(np.asarray(dataset.poses_w2c, dtype=np.float32))
    frame_indices = torch.as_tensor(dataset.frame_indices, dtype=torch.long)
    cache_path = (
        Path(args.canonical_sam3_visibility_cache)
        if str(args.canonical_sam3_visibility_cache).strip()
        else None
    )
    visibility: dict[str, Any] | None = None
    if cache_path is not None and cache_path.is_file():
        candidate = torch.load(cache_path, map_location="cpu")
        if (
            torch.equal(torch.as_tensor(candidate.get("frame_indices")), frame_indices)
            and torch.as_tensor(candidate.get("poses_w2c")).shape == poses.shape
            and torch.allclose(
                torch.as_tensor(candidate.get("poses_w2c")).float(), poses, atol=1e-6
            )
            and tuple(torch.as_tensor(candidate.get("depth_maps")).shape)
            == (len(dataset), feature_height, feature_width)
        ):
            visibility = candidate
    if visibility is None:
        depth_parts: list[torch.Tensor] = []
        alpha_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for view_index in range(len(dataset)):
                rendered = low_renderer.render_rgb(
                    model, poses[view_index].to(device=device)
                )
                depth_parts.append(rendered["depth"].detach().half().cpu())
                alpha_parts.append(rendered["alpha"].detach().half().cpu())
        visibility = {
            "schema_version": 1,
            "source": "query_independent_rendered_depth_visibility",
            "poses_w2c": poses,
            "frame_indices": frame_indices,
            "depth_maps": torch.stack(depth_parts),
            "alpha_maps": torch.stack(alpha_parts),
            "intrinsics": low_renderer.K.detach().float().cpu(),
            "metadata": {
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "height": feature_height,
                "width": feature_width,
                "benchmark_masks_opened": False,
                "query_points_opened": False,
            },
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(visibility, cache_path)
    image_height = int(getattr(config, "image_height", 968))
    image_width = int(getattr(config, "image_width", 1296))
    full_renderer = build_mask_renderer(
        config,
        height=image_height,
        width=image_width,
        device=device,
    )
    return {
        "model": model,
        "low_renderer": low_renderer,
        "full_renderer": full_renderer,
        "poses_w2c": poses,
        "frame_indices": frame_indices,
        "depth_maps": torch.as_tensor(visibility["depth_maps"]),
        "alpha_maps": torch.as_tensor(visibility["alpha_maps"]),
        "intrinsics": torch.as_tensor(visibility["intrinsics"]).float(),
        "feature_height": feature_height,
        "feature_width": feature_width,
        "image_height": image_height,
        "image_width": image_width,
        "visibility_cache": str(cache_path.resolve()) if cache_path else "in_memory",
    }


def _load_official_sam3_point_processor(args: argparse.Namespace) -> Any:
    from radio_gs.scripts.build_sam3_foundation_cache import (
        _load_sam3_model,
        set_requested_cuda_device,
    )

    set_requested_cuda_device(args.device)
    return _load_sam3_model(
        checkpoint_path=args.canonical_sam3_checkpoint,
        device=args.device,
        confidence_threshold=0.0,
        dtype=args.canonical_sam3_dtype,
        resolution=args.canonical_sam3_resolution,
        point_only=True,
    )


def _compile_rendered_multiview_point_query(
    *,
    query_point: torch.Tensor,
    direct_query: Any,
    core_probabilities: torch.Tensor,
    context: Mapping[str, Any],
    sam3_processor: Any,
    valid_rows: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any | None, dict[str, Any]]:
    """Render, point-prompt, and lift a world click through fixed cameras."""

    from radio_gs.scripts.eval_lerf_direct_3d_selection import (
        rasterize_registered_view_features,
    )

    assert direct_query.positive_seeds is not None
    direct_seed = direct_query.positive_seeds.weights.to(device)
    anchor_local = int(direct_seed.argmax().item())
    anchor_global = int(valid_rows[anchor_local].item())
    anchor_point = context["model"].get_xyz()[anchor_global].detach().float()
    # The world prompt first resolves to its strongest canonical primitive.
    # Only cameras in which that reconstructed primitive center agrees with
    # rendered depth may invoke the 2-D decoder; otherwise the query falls back
    # exactly to the direct 3-D field result.  This avoids forcing an MPR camera
    # whose visible Gaussian footprint does not contain the original mesh point.
    selected = select_depth_visible_views(
        anchor_point.cpu().numpy(),
        context["poses_w2c"].numpy(),
        context["intrinsics"].numpy(),
        context["depth_maps"].float().numpy(),
        context["alpha_maps"].float().numpy(),
        max_views=int(args.canonical_sam3_max_views),
        depth_tolerance=float(args.canonical_sam3_depth_tolerance),
        relative_depth_tolerance=float(args.canonical_sam3_relative_depth_tolerance),
        alpha_threshold=float(args.canonical_sam3_alpha_threshold),
    )
    selected = [
        {**candidate, "view_source": "primitive_center_depth_visibility"}
        for candidate in selected
    ]
    report: dict[str, Any] = {
        "selected_views": [],
        "rejected_views": [],
        "fallback": "",
        "visibility_anchor": "strongest anisotropic point-lift primitive",
        "prompt_coordinate": "resolved primitive center projected into selected camera",
        "anchor_global_row": anchor_global,
        "anchor_offset_m": float(
            torch.linalg.vector_norm(anchor_point - query_point.to(device)).item()
        ),
    }
    if not selected:
        report["fallback"] = "no_depth_visible_view"
        return None, report
    primitive_sum = torch.zeros(
        context["model"].get_xyz().shape[0], device=device, dtype=torch.float32
    )
    primitive_count = torch.zeros_like(primitive_sum)
    feature_height = int(context["feature_height"])
    feature_width = int(context["feature_width"])
    image_height = int(context["image_height"])
    image_width = int(context["image_width"])
    with torch.inference_mode():
        for view in selected:
            view_index = int(view["view_index"])
            pose = context["poses_w2c"][view_index].to(device=device)
            rendered = context["full_renderer"].render_rgb(context["model"], pose)
            rgb = (
                rendered["rgb"]
                .detach()
                .float()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
            click = np.array(
                [[
                    float(view["u"]) * image_width / feature_width,
                    float(view["v"]) * image_height / feature_height,
                ]],
                dtype=np.float32,
            )
            state = sam3_processor.set_image(image)
            masks, quality, _low_res = sam3_processor.model.predict_inst(
                state,
                point_coords=click,
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
            mask, candidate_index, predicted_quality = choose_official_sam3_point_mask(
                masks, quality
            )
            if predicted_quality < float(args.canonical_sam3_min_predicted_quality):
                report["rejected_views"].append(
                    {
                        **view,
                        "frame_index": int(context["frame_indices"][view_index]),
                        "sam3_candidate": int(candidate_index),
                        "sam3_predicted_quality": float(predicted_quality),
                        "minimum_predicted_quality": float(
                            args.canonical_sam3_min_predicted_quality
                        ),
                    }
                )
                del state, rendered
                continue
            mask_low = F.interpolate(
                torch.from_numpy(mask.astype(np.float32))[None, None].to(device),
                size=(feature_height, feature_width),
                mode="nearest",
            )
            frame_sum, frame_count = rasterize_registered_view_features(
                model=context["model"],
                renderer=context["low_renderer"],
                viewmat=pose,
                siglip_feat=mask_low,
                depth_map=context["depth_maps"][view_index : view_index + 1].to(device),
                alpha_map=context["alpha_maps"][view_index : view_index + 1].to(device),
                registration_depth_tolerance=float(
                    args.canonical_sam3_depth_tolerance
                ),
                registration_relative_depth_tolerance=float(
                    args.canonical_sam3_relative_depth_tolerance
                ),
                registration_alpha_threshold=float(args.canonical_sam3_alpha_threshold),
                registration_weight_mode="alpha_depth",
                gaussian_top1=True,
            )
            primitive_sum.add_(frame_sum[:, 0])
            primitive_count.add_(frame_count)
            report["selected_views"].append(
                {
                    **view,
                    "frame_index": int(context["frame_indices"][view_index]),
                    "sam3_candidate": int(candidate_index),
                    "sam3_predicted_quality": float(predicted_quality),
                    "sam3_mask_area": int(mask.sum()),
                }
            )
            del state, rendered, frame_sum, frame_count, mask_low
    observed = primitive_count > 0
    if not bool(observed.any()) or not bool((primitive_sum > 0).any()):
        report["fallback"] = (
            "no_quality_accepted_sam3_mask"
            if report["rejected_views"]
            else "empty_lifted_mask"
        )
        return None, report
    positive_global = torch.zeros_like(primitive_sum)
    positive_global[observed] = (
        primitive_sum[observed] / primitive_count[observed].clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    positive = positive_global.index_select(0, valid_rows.to(device))
    # Retain the original anisotropic world click as a conservative anchor;
    # only solver-significant seed mass is added to the lifted masks.  A
    # rendered view does not observe the whole 3-D background, so its
    # complement must never be installed as hard negative seeds.
    direct_anchor = torch.where(direct_seed >= 0.20, direct_seed, 0.0)
    core_gate = torch.as_tensor(core_probabilities).to(device).clamp(0.0, 1.0)
    positive = torch.maximum(
        float(args.canonical_sam3_seed_strength) * positive * core_gate,
        direct_anchor,
    )
    observed_valid = observed.index_select(0, valid_rows.to(device))
    query = replace(
        direct_query,
        positive_seeds=SoftSeedSet(positive, "world_point_rendered_sam3_positive"),
        negative_seeds=None,
        metadata={
            **dict(direct_query.metadata),
            "rendered_sam3_positive_seed_augmentation": True,
            "rendered_view_background_is_negative_seed": False,
        },
    )
    report.update(
        {
            "num_observed_primitives": int(observed_valid.sum().item()),
            "num_positive_primitives": int((positive >= 0.5).sum().item()),
            "num_negative_primitives": 0,
            "seed_strength": float(args.canonical_sam3_seed_strength),
            "core_probability_gate": True,
        }
    )
    return query, report


def evaluate_canonical_point_queries(
    args: argparse.Namespace,
    mesh_xyz: np.ndarray,
    instance_ids: np.ndarray,
    instance_metadata: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Query Gaussian support first, then project the one result to the mesh."""

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bank = load_canonical_capability_bank(
        args.canonical_capability_cache,
        expected_field_checkpoint_sha256=args.canonical_field_sha256,
    )
    graph_cpu = load_canonical_support_graph(args.canonical_support_graph, bank)
    valid_rows = bank.global_rows
    primitive_reliability = None
    if str(args.canonical_reliability_cache).strip():
        primitive_reliability = load_canonical_primitive_reliability(
            args.canonical_reliability_cache,
            expected_xyz=bank.xyz,
            expected_valid=bank.valid,
            expected_field_checkpoint_sha256=str(
                bank.metadata.get("field_checkpoint_sha256", "")
            ),
        )
    node_reliability = (
        primitive_reliability.valid_confidence().to(device)
        if primitive_reliability is not None
        else None
    )
    gaussian_xyz = bank.xyz[valid_rows].to(device)
    geometry_cache_path = Path(args.canonical_geometry_precision_cache) if str(
        args.canonical_geometry_precision_cache
    ).strip() else None
    geometry_cache_used = geometry_cache_path is not None and geometry_cache_path.is_file()
    if geometry_cache_used:
        geometry_payload = torch.load(geometry_cache_path, map_location="cpu")
        cached_xyz = torch.as_tensor(geometry_payload.get("xyz")).float()
        if cached_xyz.shape != bank.xyz[valid_rows].shape or not torch.allclose(
            cached_xyz, bank.xyz[valid_rows], atol=1e-6, rtol=0.0
        ):
            raise ValueError("canonical geometry precision cache xyz mismatch")
        covariance = torch.as_tensor(geometry_payload.get("covariance")).to(
            device=device, dtype=torch.float32
        )
        precision = torch.as_tensor(geometry_payload.get("precision")).to(
            device=device, dtype=torch.float32
        )
        if covariance.shape != (valid_rows.numel(), 3, 3) or precision.shape != covariance.shape:
            raise ValueError("canonical geometry precision cache tensors are malformed")
    else:
        if not args.config or not args.checkpoint:
            raise ValueError(
                "canonical point query without a geometry cache requires --config and --checkpoint"
            )
        config = load_config(args.config)
        model, _codec = _build_hybrid_model(config, args.checkpoint, device)
        geometry_xyz = model.get_xyz().detach().float().cpu()
        if geometry_xyz.shape != bank.xyz.shape or not torch.allclose(
            geometry_xyz, bank.xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("canonical capability geometry does not match ScanNet field")
        covariance = _gaussian_covariances(model).detach()[valid_rows.to(device)]
        covariance_identity = torch.eye(3, device=device, dtype=covariance.dtype)
        precision = torch.linalg.pinv(covariance + 1e-6 * covariance_identity)
        if geometry_cache_path is not None:
            geometry_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "source": "fixed_gaussian_geometry_query_acceleration",
                    "xyz": bank.xyz[valid_rows].clone(),
                    "covariance": covariance.detach().float().cpu(),
                    "precision": precision.detach().float().cpu(),
                    "uses_query_or_labels": False,
                },
                geometry_cache_path,
            )
    feature_banks = {
        name: values.to(device)
        for name, values in bank.valid_feature_banks().items()
    }
    graph = graph_cpu.to(device)
    engine = CanonicalQueryEngine(
        graph,
        scoring_config=EvidenceScoringConfig(
            semantic_weight=1.0,
            appearance_weight=args.canonical_appearance_weight,
            boundary_weight=args.canonical_boundary_weight,
            prototype_temperature=args.canonical_prototype_temperature,
            feature_calibration=args.canonical_feature_calibration,
            background_centroids=args.canonical_background_centroids,
            calibration_sample_size=args.canonical_calibration_sample_size,
            centroid_iterations=args.canonical_centroid_iterations,
            score_calibration=args.canonical_score_calibration,
            score_tanh_scale=args.canonical_score_tanh_scale,
            score_chunk_size=args.canonical_score_chunk_size,
        ),
        solver_config=SupportSolverConfig(
            iterations=args.canonical_solver_iterations,
            residual=args.residual,
            unary_temperature=args.canonical_unary_temperature,
            support_threshold=args.canonical_support_threshold,
        ),
        graph_policy=args.canonical_graph_policy,
        component_graph_policy=args.canonical_component_graph_policy,
        graph_legacy_residual=args.canonical_graph_legacy_residual,
        node_reliability=node_reliability,
    )
    # Query-independent projection from Gaussian support to official mesh
    # vertices.  Local graph sigma provides an adaptive scale; no labels enter.
    projection_k = min(max(1, int(args.mesh_projection_k)), int(valid_rows.numel()))
    distance, neighbor = cKDTree(bank.xyz[valid_rows].numpy()).query(
        np.asarray(mesh_xyz, dtype=np.float32), k=projection_k
    )
    distance = np.asarray(distance, dtype=np.float32)
    neighbor = np.asarray(neighbor, dtype=np.int64)
    if distance.ndim == 1:
        distance = distance[:, None]
        neighbor = neighbor[:, None]
    sigma = graph_cpu.local_sigma.numpy()[neighbor]
    projection_weight = np.exp(
        -0.5 * np.square(distance / np.maximum(sigma, 1e-6))
    ).astype(np.float32)
    projection_weight /= np.maximum(projection_weight.sum(axis=1, keepdims=True), 1e-8)

    eligible = [
        int(instance_id)
        for instance_id in sorted(instance_metadata)
        if int((instance_ids == int(instance_id)).sum()) >= int(args.min_instance_points)
    ]
    if args.max_instances > 0:
        eligible = eligible[: int(args.max_instances)]

    # Freeze every direct 3-D result before loading or invoking the optional
    # 2-D decoder.  Third-party inference stacks may change process-wide CUDA
    # math settings; interleaving them with the core query loop made the later
    # core rows drift even though their inputs were identical.  Apart from
    # fixing the audit contract, this also makes ``macro_core_iou`` an exact
    # ablation control for every post-processing variant.
    prepared: list[dict[str, Any]] = []
    for instance_id in eligible:
        target = instance_ids == instance_id
        candidates = np.flatnonzero(target)
        rng = np.random.default_rng(int(args.random_seed) + 1_000_003 * instance_id)
        seed_index = int(candidates[int(rng.integers(0, candidates.size))])
        query_point = torch.from_numpy(mesh_xyz[seed_index]).to(device=device)
        query = compile_world_3d_query(
            gaussian_xyz,
            covariance,
            query_point,
            appearance_features=feature_banks["appearance"],
            boundary_features=feature_banks["boundary"],
            appearance_signature=bank.signatures["appearance"],
            boundary_signature=bank.signatures["boundary"],
            prototype_count=args.canonical_prototype_count,
            prototype_strategy=args.canonical_prototype_strategy,
            scene_mean_negative=True,
            gaussian_precision=precision,
            euclidean_candidate_k=args.canonical_point_euclidean_candidate_k,
            seed_topk=args.canonical_point_seed_topk,
            seed_temperature=args.canonical_point_seed_temperature,
        )
        core_result = engine.execute(
            query,
            feature_banks,
            feature_signatures=bank.signatures,
        )
        prepared.append(
            {
                "instance_id": instance_id,
                "target": target,
                "seed_index": seed_index,
                "query_point": query_point,
                "query": query,
                "core_result": core_result,
            }
        )

    rendered_prompt_context = None
    sam3_point_processor = None
    if bool(args.canonical_sam3_multiview):
        rendered_prompt_context = _load_or_build_rendered_view_context(
            args,
            device=device,
            expected_xyz=bank.xyz,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sam3_point_processor = _load_official_sam3_point_processor(args)

    rows: list[dict[str, Any]] = []
    for item in prepared:
        instance_id = int(item["instance_id"])
        target = item["target"]
        seed_index = int(item["seed_index"])
        query_point = item["query_point"]
        query = item["query"]
        core_result = item["core_result"]
        result = core_result
        rendered_prompt_report = None
        if rendered_prompt_context is not None:
            assert sam3_point_processor is not None
            rendered_query, rendered_prompt_report = (
                _compile_rendered_multiview_point_query(
                    query_point=query_point,
                    direct_query=query,
                    core_probabilities=core_result.probabilities,
                    context=rendered_prompt_context,
                    sam3_processor=sam3_point_processor,
                    valid_rows=valid_rows,
                    args=args,
                    device=device,
                )
            )
            if rendered_query is not None:
                result = engine.execute(
                    rendered_query,
                    feature_banks,
                    feature_signatures=bank.signatures,
                )
        primitive_stages = {
            "core": core_result.selected_probabilities,
            "unary": torch.sigmoid(
                result.unary / float(args.canonical_unary_temperature)
            ),
            "propagated": result.probabilities,
            "selected": result.selected_probabilities,
        }
        predictions: dict[str, np.ndarray] = {}
        for stage, primitive_values in primitive_stages.items():
            values = primitive_values.detach().float().cpu().numpy()
            mesh_probability = (values[neighbor] * projection_weight).sum(axis=1)
            predictions[stage] = (
                mesh_probability >= float(args.canonical_support_threshold)
            )
        prediction = predictions["selected"]
        meta = instance_metadata.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "label": str(meta.get("label", "")),
                "seed_index": seed_index,
                "seed_xyz": mesh_xyz[seed_index].tolist(),
                "num_gt_points": int(target.sum()),
                "core_iou": _intersection_over_union(predictions["core"], target),
                "unary_iou": _intersection_over_union(predictions["unary"], target),
                "propagated_iou": _intersection_over_union(
                    predictions["propagated"], target
                ),
                "iou": _intersection_over_union(prediction, target),
                "num_predicted_points": int(prediction.sum()),
                "rendered_prompt": rendered_prompt_report,
            }
        )
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    core_ious = np.asarray([row["core_iou"] for row in rows], dtype=np.float64)
    unary_ious = np.asarray([row["unary_iou"] for row in rows], dtype=np.float64)
    propagated_ious = np.asarray(
        [row["propagated_iou"] for row in rows], dtype=np.float64
    )
    return {
        "num_queries": len(rows),
        "macro_core_iou": float(core_ious.mean()) if core_ious.size else 0.0,
        "macro_unary_iou": float(unary_ious.mean()) if unary_ious.size else 0.0,
        "macro_propagated_iou": (
            float(propagated_ious.mean()) if propagated_ious.size else 0.0
        ),
        "macro_connected_iou": float(ious.mean()) if ious.size else 0.0,
        "accuracy_at_025": float((ious >= 0.25).mean()) if ious.size else 0.0,
        "accuracy_at_050": float((ious >= 0.50).mean()) if ious.size else 0.0,
        "queries": rows,
        "protocol": {
            "query": "one deterministic GT-sampled mesh point per instance",
            "query_reveals": "only the sampled world coordinate",
            "field_domain": "canonical Gaussian primitives",
            "output_domain": "official ScanNet mesh vertices",
            "mesh_projection": f"adaptive-sigma Gaussian kNN mean, k={projection_k}",
            "point_seed_lifting": (
                "anisotropic Gaussian Mahalanobis"
                if int(args.canonical_point_seed_topk) <= 0
                else "anisotropic Gaussian Mahalanobis with fixed top-"
                f"{int(args.canonical_point_seed_topk)} support"
            ),
            "point_seed_temperature": float(args.canonical_point_seed_temperature),
            "prototype_strategy": str(args.canonical_prototype_strategy),
            "point_seed_euclidean_candidate_k": int(
                args.canonical_point_euclidean_candidate_k
            ),
            "unary": "official DINOv3 + official SAM3 prototypes minus unlabeled scene mean",
            "primitive_reliability": (
                {
                    "cache": str(Path(args.canonical_reliability_cache).resolve()),
                    "formula": primitive_reliability.metadata.get("formula"),
                    "application": "centered_unary_shrink",
                    "prototype_precision_weighting": False,
                    "centered_unary_shrink": node_reliability is not None,
                    "world_seed_constraints_shrunk": False,
                    "uses_query_or_target_labels": False,
                }
                if primitive_reliability is not None
                else None
            ),
            "rendered_multiview_prompt": {
                "enabled": bool(args.canonical_sam3_multiview),
                "query_compiler": (
                    "world point -> depth-visible rendered RGB -> official SAM3 "
                    "interactive point decoder -> alpha/depth raster lift"
                ),
                "max_views": int(args.canonical_sam3_max_views),
                "visibility_cache": (
                    rendered_prompt_context["visibility_cache"]
                    if rendered_prompt_context is not None
                    else ""
                ),
                "sam3_checkpoint": (
                    str(Path(args.canonical_sam3_checkpoint).resolve())
                    if args.canonical_sam3_multiview
                    else ""
                ),
                "sam3_dtype": args.canonical_sam3_dtype,
                "candidate_selection": "official predicted-IoU argmax",
                "minimum_predicted_quality": float(
                    args.canonical_sam3_min_predicted_quality
                ),
                "seed_fusion": (
                    "positive-only, core-probability-gated augmentation of the "
                    "original world-point query"
                ),
                "seed_strength": float(args.canonical_sam3_seed_strength),
                "rendered_background_as_negative_seed": False,
                "render_source": "method RGB Gaussian renderer",
                "uses_training_or_test_rgb": False,
                "uses_target_labels_or_masks": False,
            },
            "score_calibration": {
                "feature_calibration": args.canonical_feature_calibration,
                "background_centroids": int(args.canonical_background_centroids),
                "sample_size": int(args.canonical_calibration_sample_size),
                "centroid_iterations": int(args.canonical_centroid_iterations),
                "score_calibration": args.canonical_score_calibration,
                "score_tanh_scale": float(args.canonical_score_tanh_scale),
                "uses_target_labels": False,
                "uses_target_masks": False,
                "uses_query_conditioned_scores": (
                    args.canonical_score_calibration != "none"
                ),
                "uses_unlabeled_scene_statistics": (
                    args.canonical_feature_calibration != "none"
                    or int(args.canonical_background_centroids) > 0
                    or args.canonical_score_calibration != "none"
                ),
            },
            "support_solver": {
                "graph_policy": args.canonical_graph_policy,
                "component_graph_policy": args.canonical_component_graph_policy,
                "graph_legacy_residual": float(args.canonical_graph_legacy_residual),
                "iterations": int(args.canonical_solver_iterations),
                "residual": float(args.residual),
                "unary_temperature": float(args.canonical_unary_temperature),
                "support_threshold": float(args.canonical_support_threshold),
            },
            "test_calibration": False,
            "test_calibration_definition": (
                "no target labels, target masks, or metric feedback are used; "
                "unlabeled evaluation-scene statistics are disclosed separately"
            ),
            "benchmark_masks_opened_during_field_or_graph_build": False,
            "geometry_precision_cache": (
                str(geometry_cache_path.resolve()) if geometry_cache_path else ""
            ),
            "geometry_precision_cache_reused": bool(geometry_cache_used),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--label_ply")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--feature_cache", default="")
    parser.add_argument(
        "--graph_feature_cache",
        help="Optional feature cache used only for pairwise graph affinities.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument(
        "--evaluation_domain",
        choices=("mesh_query", "gaussian_row"),
        default="mesh_query",
    )
    parser.add_argument(
        "--region_space",
        choices=("radio_raw", "sam3", "dino_v3_7b", "sam3_dino_equal_fusion"),
        default="radio_raw",
    )
    parser.add_argument("--query_k", type=int, default=8)
    parser.add_argument("--query_candidate_k", type=int, default=80)
    parser.add_argument(
        "--mesh_query_mode",
        choices=("row_latent_at_mesh", "knn_field"),
        default="row_latent_at_mesh",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_instance_points", type=int, default=100)
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--spatial_sigma", type=float, default=0.08)
    parser.add_argument("--feature_temperature", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--residual", type=float, default=0.35)
    parser.add_argument(
        "--graph_mode",
        choices=("directed", "symmetric_union"),
        default="directed",
    )
    parser.add_argument("--adaptive_spatial", action="store_true")
    parser.add_argument("--spatial_scale", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--component_radius", type=float, default=0.16)
    parser.add_argument("--clicks", type=int, default=1)
    parser.add_argument(
        "--positive_mode",
        choices=("single", "local_multiscale"),
        default="single",
    )
    parser.add_argument("--local_prototype_neighbors", type=int, default=64)
    parser.add_argument("--canonical_capability_cache", default="")
    parser.add_argument("--canonical_support_graph", default="")
    parser.add_argument("--canonical_reliability_cache", default="")
    parser.add_argument(
        "--canonical_graph_policy",
        choices=(
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--canonical_component_graph_policy",
        choices=(
            "same",
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="same",
    )
    parser.add_argument("--canonical_graph_legacy_residual", type=float, default=0.0)
    parser.add_argument("--canonical_field_sha256", default="")
    parser.add_argument("--canonical_geometry_precision_cache", default="")
    parser.add_argument("--canonical_prototype_count", type=int, default=4)
    parser.add_argument(
        "--canonical_prototype_strategy",
        choices=("weighted_fps", "spherical_mean_fps"),
        default="weighted_fps",
    )
    parser.add_argument("--canonical_point_seed_topk", type=int, default=0)
    parser.add_argument(
        "--canonical_point_euclidean_candidate_k", type=int, default=64
    )
    parser.add_argument("--canonical_point_seed_temperature", type=float, default=1.0)
    parser.add_argument("--canonical_appearance_weight", type=float, default=1.0)
    parser.add_argument("--canonical_boundary_weight", type=float, default=0.35)
    parser.add_argument("--canonical_prototype_temperature", type=float, default=0.07)
    parser.add_argument(
        "--canonical_feature_calibration",
        choices=("none", "diagonal_robust"),
        default="none",
    )
    parser.add_argument("--canonical_background_centroids", type=int, default=0)
    parser.add_argument("--canonical_calibration_sample_size", type=int, default=8192)
    parser.add_argument("--canonical_centroid_iterations", type=int, default=4)
    parser.add_argument(
        "--canonical_score_calibration",
        choices=("none", "robust_tanh", "robust_tanh_centered", "robust_tanh_zero"),
        default="none",
    )
    parser.add_argument("--canonical_score_tanh_scale", type=float, default=2.0)
    parser.add_argument("--canonical_score_chunk_size", type=int, default=65536)
    parser.add_argument("--canonical_solver_iterations", type=int, default=12)
    parser.add_argument("--canonical_unary_temperature", type=float, default=0.10)
    parser.add_argument("--canonical_support_threshold", type=float, default=0.50)
    parser.add_argument("--mesh_projection_k", type=int, default=8)
    parser.add_argument(
        "--canonical_sam3_multiview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reproject each world click into depth-visible method-rendered views, "
            "run the official SAM3 interactive point decoder, and lift its masks."
        ),
    )
    parser.add_argument("--canonical_sam3_max_views", type=int, default=1)
    parser.add_argument("--canonical_sam3_seed_strength", type=float, default=0.50)
    parser.add_argument(
        "--canonical_sam3_min_predicted_quality",
        type=float,
        default=0.50,
        help=(
            "Fixed, benchmark-independent minimum official SAM3 predicted-IoU "
            "for accepting a rendered point mask."
        ),
    )
    parser.add_argument("--canonical_sam3_visibility_cache", default="")
    parser.add_argument(
        "--canonical_sam3_checkpoint",
        default="checkpoints/sam3_modelscope/sam3.pt",
    )
    parser.add_argument(
        "--canonical_sam3_dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--canonical_sam3_resolution", type=int, default=1008)
    parser.add_argument("--canonical_sam3_depth_tolerance", type=float, default=0.08)
    parser.add_argument(
        "--canonical_sam3_relative_depth_tolerance", type=float, default=0.02
    )
    parser.add_argument("--canonical_sam3_alpha_threshold", type=float, default=0.02)
    args = parser.parse_args()
    if not 0.0 <= float(args.canonical_sam3_min_predicted_quality) <= 1.0:
        raise ValueError("--canonical_sam3_min_predicted_quality must be in [0,1]")

    instance_ids, metadata = load_scannet_instances(args.aggregation, args.segmentation)
    mesh_xyz = None
    if args.evaluation_domain == "mesh_query":
        if not args.label_ply:
            raise ValueError("--label_ply is required for evaluation_domain=mesh_query")
        mesh_xyz, _ = _read_label_ply(args.label_ply)
        if mesh_xyz.shape[0] != instance_ids.shape[0]:
            raise ValueError("Official mesh vertices and instance segIndices do not align")
    if args.canonical_capability_cache:
        if not args.canonical_support_graph:
            raise ValueError("--canonical_support_graph is required for canonical queries")
        if mesh_xyz is None:
            raise ValueError("canonical point query requires evaluation_domain=mesh_query")
        metrics = evaluate_canonical_point_queries(
            args, mesh_xyz, instance_ids, metadata
        )
        payload = {
            "scene": args.scene,
            "protocol": metrics.pop("protocol"),
            "metrics": metrics,
            "canonical_capability_cache": str(
                Path(args.canonical_capability_cache).resolve()
            ),
            "canonical_support_graph": str(Path(args.canonical_support_graph).resolve()),
            "canonical_reliability_cache": (
                str(Path(args.canonical_reliability_cache).resolve())
                if str(args.canonical_reliability_cache).strip()
                else ""
            ),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in metrics.items() if key != "queries"},
                indent=2,
            )
        )
        return
    if not args.feature_cache:
        raise ValueError("--feature_cache is required for the legacy field evaluator")
    cache_path = Path(args.feature_cache)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            xyz = np.asarray(cache["xyz"], dtype=np.float32)
            features = np.asarray(cache["region_features"], dtype=np.float32)
            cached_space = str(cache["region_space"].item()) if "region_space" in cache else "sam3"
            cached_domain = str(cache["evaluation_domain"].item()) if "evaluation_domain" in cache else "gaussian_row"
            cached_mesh_mode = str(cache["mesh_query_mode"].item()) if "mesh_query_mode" in cache else "not_recorded"
        if (
            cached_space != args.region_space
            or cached_domain != args.evaluation_domain
            or (
                args.evaluation_domain == "mesh_query"
                and cached_mesh_mode != args.mesh_query_mode
            )
        ):
            raise ValueError(
                "Feature cache contract mismatch: "
                f"space/domain={cached_space}/{cached_domain}, requested="
                f"{args.region_space}/{args.evaluation_domain}; mesh_mode="
                f"{cached_mesh_mode}/{args.mesh_query_mode}"
            )
    else:
        if not args.config or not args.checkpoint:
            raise ValueError("--config and --checkpoint are required when feature_cache is absent")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        xyz, features = extract_region_features(
            args.config,
            args.checkpoint,
            args.radio_checkpoint,
            device=device,
            chunk_size=args.chunk_size,
            query_xyz=mesh_xyz,
            query_k=args.query_k,
            query_candidate_k=args.query_candidate_k,
            region_space=args.region_space,
            mesh_query_mode=args.mesh_query_mode,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            xyz=xyz,
            region_features=features,
            region_space=np.asarray(args.region_space),
            evaluation_domain=np.asarray(args.evaluation_domain),
            mesh_query_mode=np.asarray(args.mesh_query_mode),
        )
    if instance_ids.shape[0] != xyz.shape[0]:
        raise ValueError(
            "Instance vertices and evaluation coordinates are not aligned: "
            f"{instance_ids.shape[0]} vs {xyz.shape[0]}"
        )
    graph_features = None
    graph_space = args.region_space
    if args.graph_feature_cache:
        with np.load(args.graph_feature_cache, allow_pickle=False) as cache:
            graph_xyz = np.asarray(cache["xyz"], dtype=np.float32)
            graph_features = np.asarray(cache["region_features"], dtype=np.float32)
            graph_space = str(cache["region_space"].item())
            graph_domain = str(cache["evaluation_domain"].item())
        if graph_xyz.shape != xyz.shape or not np.allclose(graph_xyz, xyz, atol=1e-6):
            raise ValueError("Graph feature cache coordinates do not match unary cache")
        if graph_domain != args.evaluation_domain:
            raise ValueError("Graph feature cache evaluation domain does not match")

    propagation = SupportPropagationConfig(
        neighbors=args.neighbors,
        spatial_sigma=args.spatial_sigma,
        feature_temperature=args.feature_temperature,
        iterations=args.iterations,
        residual=args.residual,
        graph_mode=args.graph_mode,
        adaptive_spatial=args.adaptive_spatial,
        spatial_scale=args.spatial_scale,
    )
    metrics = evaluate_point_queries(
        xyz,
        features,
        instance_ids,
        metadata,
        graph_features=graph_features,
        random_seed=args.random_seed,
        min_instance_points=args.min_instance_points,
        max_instances=args.max_instances or None,
        propagation=propagation,
        threshold=args.threshold,
        component_radius=args.component_radius,
        clicks=args.clicks,
        positive_mode=args.positive_mode,
        local_prototype_neighbors=args.local_prototype_neighbors,
    )
    payload = {
        "scene": args.scene,
        "protocol": {
            "query": f"{args.clicks} deterministic GT-sampled 3D point(s) per instance",
            "positive_mode": args.positive_mode,
            "local_prototype_neighbors": args.local_prototype_neighbors,
            "readout": args.region_space,
            "unary_feature_space": args.region_space,
            "pairwise_feature_space": graph_space,
            "negative_prototype": (
                "label-free mean over all evaluation-domain features; "
                "may include the queried object"
            ),
            "threshold": args.threshold,
            "threshold_comparison": "greater_or_equal",
            "threshold_source": "fixed command-line protocol; never test-set calibrated",
            "propagation": vars(propagation),
            "component_radius": args.component_radius,
            "evaluation_domain": args.evaluation_domain,
            "row_alignment_required": args.evaluation_domain == "gaussian_row",
            "query_k": args.query_k,
            "query_candidate_k": args.query_candidate_k,
            "mesh_query_mode": args.mesh_query_mode,
            "mesh_query_provenance": (
                "same-row latent with hash branch evaluated at official mesh xyz; "
                "requires preserved initialization order"
                if args.mesh_query_mode == "row_latent_at_mesh"
                else "continuous kNN field query at official mesh xyz"
            ),
        },
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
