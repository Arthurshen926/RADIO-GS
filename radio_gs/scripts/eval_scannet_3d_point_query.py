#!/usr/bin/env python3
"""Evaluate a fixed 3-D point -> 3-D instance-mask query protocol on ScanNet.

One ground-truth vertex is sampled deterministically per instance to construct
the point query.  The method receives only that vertex index/coordinate; the
remaining instance mask is opened only by the metric code.  Region features
come from the frozen official RADIO ``sam3`` feature-projection adaptor.
No threshold or propagation parameter is fitted on evaluation ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.unified_query import (
    QueryKind,
    QuerySpace,
    QuerySpec,
    SupportPropagationConfig,
    binary_mask,
    build_support_graph,
    propagate_support,
    score_features,
    seed_connected_component,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
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


def evaluate_point_queries(
    xyz: np.ndarray,
    region_features: np.ndarray,
    instance_ids: np.ndarray,
    instance_metadata: Mapping[int, Mapping[str, Any]],
    *,
    random_seed: int,
    min_instance_points: int,
    max_instances: int | None,
    propagation: SupportPropagationConfig,
    threshold: float,
    component_radius: float,
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

    graph = build_support_graph(xyz, features, propagation)
    background_prototype = features.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for instance_id in eligible:
        target = instances == instance_id
        candidates = np.flatnonzero(target)
        # A per-instance RNG makes the sampled point stable if other instances
        # are later filtered from the benchmark.
        rng = np.random.default_rng(int(random_seed) + 1_000_003 * instance_id)
        seed_index = int(candidates[int(rng.integers(0, candidates.size))])
        query = QuerySpec(
            kind=QueryKind.POINT_3D,
            space=QuerySpace.REGION,
            positive_prototypes=features[seed_index],
            negative_prototypes=background_prototype,
            positive_seed_indices=(seed_index,),
            metadata={"source": "single_registered_world_point"},
        )
        raw_scores = score_features(features, query)
        propagated_scores = propagate_support(
            xyz,
            features,
            raw_scores,
            query,
            propagation,
            graph=graph,
        )
        raw_prediction = binary_mask(raw_scores, threshold=threshold)
        propagated_prediction = binary_mask(propagated_scores, threshold=threshold)
        connected_prediction = seed_connected_component(
            propagated_prediction,
            seed_index,
            graph,
            max_edge_distance=component_radius,
        )
        meta = instance_metadata.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "label": str(meta.get("label", "")),
                "seed_index": seed_index,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the canonical field and apply the official frozen SAM3 adaptor."""

    config = load_config(config_path)
    model, codec = _build_hybrid_model(config, checkpoint_path, device)
    adaptor = load_radio_adaptor_from_checkpoint(
        radio_checkpoint, "sam3", kind="feature_projection"
    ).to(device).eval().requires_grad_(False)
    if adaptor.input_dim != 1280 or adaptor.output_dim != 1024:
        raise ValueError("Official RADIO sam3 feature projection must be 1280 -> 1024")
    xyz = model.get_xyz().detach().float().cpu().numpy()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, model.num_gaussians, int(chunk_size)):
            stop = min(start + int(chunk_size), model.num_gaussians)
            indices = torch.arange(start, stop, device=device, dtype=torch.long)
            decoded = _decode_gaussian_indices_1280(model, codec, indices)
            projected = adaptor(decoded.float()).float()
            parts.append(projected.cpu().numpy())
    return xyz, np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_instance_points", type=int, default=100)
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--spatial_sigma", type=float, default=0.08)
    parser.add_argument("--feature_temperature", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--residual", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--component_radius", type=float, default=0.16)
    args = parser.parse_args()

    instance_ids, metadata = load_scannet_instances(args.aggregation, args.segmentation)
    cache_path = Path(args.feature_cache)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            xyz = np.asarray(cache["xyz"], dtype=np.float32)
            features = np.asarray(cache["region_features"], dtype=np.float32)
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
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, xyz=xyz, region_features=features)
    if instance_ids.shape[0] != xyz.shape[0]:
        raise ValueError(
            "Instance vertices and Gaussian rows are not aligned: "
            f"{instance_ids.shape[0]} vs {xyz.shape[0]}. Use the prepared row-aligned ScanNet field."
        )

    propagation = SupportPropagationConfig(
        neighbors=args.neighbors,
        spatial_sigma=args.spatial_sigma,
        feature_temperature=args.feature_temperature,
        iterations=args.iterations,
        residual=args.residual,
    )
    metrics = evaluate_point_queries(
        xyz,
        features,
        instance_ids,
        metadata,
        random_seed=args.random_seed,
        min_instance_points=args.min_instance_points,
        max_instances=args.max_instances or None,
        propagation=propagation,
        threshold=args.threshold,
        component_radius=args.component_radius,
    )
    payload = {
        "scene": args.scene,
        "protocol": {
            "query": "one deterministic GT-sampled 3D point per instance",
            "readout": "official frozen RADIO sam3 feature_projection",
            "negative_prototype": "unlabeled scene feature mean",
            "threshold": args.threshold,
            "threshold_comparison": "greater_or_equal",
            "threshold_source": "fixed command-line protocol; never test-set calibrated",
            "propagation": vars(propagation),
            "component_radius": args.component_radius,
            "row_alignment_required": True,
        },
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
