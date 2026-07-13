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
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree

from radio_gs.config import load_config
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_support_graph,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import EvidenceScoringConfig
from radio_gs.querying.query_compilers import compile_world_3d_query
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
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
    _decode_points_1280,
    _read_label_ply,
)


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


def evaluate_canonical_point_queries(
    args: argparse.Namespace,
    mesh_xyz: np.ndarray,
    instance_ids: np.ndarray,
    instance_metadata: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Query Gaussian support first, then project the one result to the mesh."""

    if not args.config or not args.checkpoint:
        raise ValueError("canonical point query requires --config and --checkpoint")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bank = load_canonical_capability_bank(
        args.canonical_capability_cache,
        expected_field_checkpoint_sha256=args.canonical_field_sha256,
    )
    graph_cpu = load_canonical_support_graph(args.canonical_support_graph, bank)
    config = load_config(args.config)
    model, _codec = _build_hybrid_model(config, args.checkpoint, device)
    geometry_xyz = model.get_xyz().detach().float().cpu()
    if geometry_xyz.shape != bank.xyz.shape or not torch.allclose(
        geometry_xyz, bank.xyz, atol=1e-6, rtol=0.0
    ):
        raise ValueError("canonical capability geometry does not match ScanNet field")

    valid_rows = bank.global_rows
    covariance = _gaussian_covariances(model).detach()[valid_rows.to(device)]
    gaussian_xyz = bank.xyz[valid_rows].to(device)
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
        ),
        solver_config=SupportSolverConfig(
            iterations=args.iterations,
            residual=args.residual,
            unary_temperature=args.canonical_unary_temperature,
            support_threshold=args.canonical_support_threshold,
        ),
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
    rows: list[dict[str, Any]] = []
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
            scene_mean_negative=True,
        )
        result = engine.execute(
            query,
            feature_banks,
            feature_signatures=bank.signatures,
        )
        primitive_probability = result.selected_probabilities.detach().float().cpu().numpy()
        mesh_probability = (
            primitive_probability[neighbor] * projection_weight
        ).sum(axis=1)
        prediction = mesh_probability >= float(args.canonical_support_threshold)
        meta = instance_metadata.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "label": str(meta.get("label", "")),
                "seed_index": seed_index,
                "seed_xyz": mesh_xyz[seed_index].tolist(),
                "num_gt_points": int(target.sum()),
                "iou": _intersection_over_union(prediction, target),
                "num_predicted_points": int(prediction.sum()),
            }
        )
    ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    return {
        "num_queries": len(rows),
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
            "unary": "official DINOv3 + official SAM3 prototypes minus unlabeled scene mean",
            "support_solver": {
                "iterations": int(args.iterations),
                "residual": float(args.residual),
                "unary_temperature": float(args.canonical_unary_temperature),
                "support_threshold": float(args.canonical_support_threshold),
            },
            "test_calibration": False,
            "benchmark_masks_opened_during_field_or_graph_build": False,
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
    parser.add_argument("--canonical_field_sha256", default="")
    parser.add_argument("--canonical_prototype_count", type=int, default=4)
    parser.add_argument("--canonical_appearance_weight", type=float, default=1.0)
    parser.add_argument("--canonical_boundary_weight", type=float, default=0.35)
    parser.add_argument("--canonical_prototype_temperature", type=float, default=0.07)
    parser.add_argument("--canonical_unary_temperature", type=float, default=0.10)
    parser.add_argument("--canonical_support_threshold", type=float, default=0.50)
    parser.add_argument("--mesh_projection_k", type=int, default=8)
    args = parser.parse_args()

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
