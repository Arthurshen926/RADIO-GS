#!/usr/bin/env python3
"""Evaluate a frozen PFIR unary under one shared graph/readout contract."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.benchmarks.scannet_pfir.build_benchmark import find_scene_annotations
from radio_gs.benchmarks.scannet_pfir.evaluation.eval_instance_selection import (
    evaluate_instance_selection,
)
from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances
from radio_gs.interfaces import (
    load_canonical_capability_bank,
    load_canonical_support_graph,
)
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import SupportSolverConfig
from radio_gs.scripts.redecode_posefree_image_support import (
    decode_posefree_image_unary,
)


def prepare_mesh_interpolator(
    source_xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    *,
    neighbors: int,
    maximum_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the GT-independent primitive-to-mesh interpolation."""

    source = np.asarray(source_xyz, dtype=np.float32)
    target = np.asarray(mesh_xyz, dtype=np.float32)
    distance, index = cKDTree(source).query(
        target, k=min(int(neighbors), len(source))
    )
    distance = np.asarray(distance, dtype=np.float32)
    index = np.asarray(index, dtype=np.int64)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    valid_neighbors = distance <= float(maximum_distance_m)
    weight = np.where(valid_neighbors, 1.0 / np.maximum(distance, 1e-4), 0.0)
    weight_sum = weight.sum(axis=1)
    normalized = weight / np.maximum(weight_sum[:, None], 1e-8)
    return index, normalized.astype(np.float32), weight_sum > 0


def interpolate_scores(
    scores: np.ndarray,
    index: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    mapped = (values[index] * weight).sum(axis=1)
    mapped[~valid] = -np.inf
    return mapped


def run(args: argparse.Namespace) -> dict:
    benchmark_root = Path(args.benchmark_dir)
    source_root = Path(args.source_run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_root = output_root / "predictions" / "selection"
    prediction_root.mkdir(parents=True, exist_ok=True)
    internal = json.loads(
        (benchmark_root / "manifest.internal.json").read_text(encoding="utf-8")
    )
    records = internal["queries"]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene_id"])].append(record)

    masks: dict[str, np.ndarray] = {}
    instances_by_scene: dict[str, np.ndarray] = {}
    xyz_by_scene: dict[str, np.ndarray] = {}
    scene_audit: dict[str, dict] = {}
    device = torch.device(args.device)
    config = SupportSolverConfig(
        iterations=args.iterations,
        residual=args.residual,
        unary_temperature=args.unary_temperature,
        support_threshold=args.support_threshold,
    )
    selection_mode = SelectionMode(args.selection_mode)
    for scene, scene_records in sorted(by_scene.items()):
        field_root = source_root / "canonical_fields" / scene
        bank = load_canonical_capability_bank(
            field_root / "official_dino_sam3_views.pt"
        )
        graph_path = field_root / args.support_graph_name
        graph = load_canonical_support_graph(graph_path, bank).to(device)
        mesh, aggregation, segmentation = find_scene_annotations(
            scene, args.annotations_root
        )
        mesh_xyz, mesh_instances, _ = load_mesh_instances(
            mesh, aggregation, segmentation
        )
        xyz_by_scene[scene] = mesh_xyz
        instances_by_scene[scene] = mesh_instances
        index, weight, mesh_valid = prepare_mesh_interpolator(
            bank.xyz[bank.global_rows].numpy(),
            mesh_xyz,
            neighbors=args.neighbors,
            maximum_distance_m=args.maximum_distance_m,
        )
        selected_counts = []
        for record in scene_records:
            query_id = str(record["query_id"])
            cache = torch.load(
                source_root / "query_caches" / f"{query_id}.pt",
                map_location="cpu",
            )
            if not torch.equal(
                torch.as_tensor(cache["valid"]).bool().cpu(), bank.valid
            ):
                raise ValueError(f"{query_id}: query/capability valid rows differ")
            unary = (
                torch.as_tensor(cache["unary"])
                .float()
                .reshape(-1)[bank.global_rows]
                .to(device)
            )
            probabilities, selected = decode_posefree_image_unary(
                graph,
                unary,
                solver_config=config,
                graph_policy=args.graph_policy,
                channel_confidence_mode=args.channel_confidence_mode,
                selection_mode=selection_mode,
            )
            selected_probability = (
                probabilities * selected.to(probabilities.dtype)
            ).detach().cpu().numpy()
            mesh_score = interpolate_scores(
                selected_probability, index, weight, mesh_valid
            )
            mask = mesh_valid & (mesh_score >= float(args.support_threshold))
            masks[query_id] = mask
            np.save(prediction_root / f"{query_id}.npy", mask, allow_pickle=False)
            selected_counts.append(int(selected.sum()))
        scene_audit[scene] = {
            "queries": len(scene_records),
            "graph": str(graph_path.resolve()),
            "valid_primitives": int(bank.valid.sum()),
            "selected_primitives_mean": float(np.mean(selected_counts)),
            "mesh_mapping_coverage": float(mesh_valid.mean()),
        }
        del graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = evaluate_instance_selection(
        records, masks, instances_by_scene, xyz_by_scene
    )
    report.update(
        {
            "benchmark_version": internal["benchmark_version"],
            "prediction_domain": "official_scannet_annotation_mesh_vertices",
            "test_calibration": False,
            "support_redecode": {
                "source_run_root": str(source_root.resolve()),
                "query_encoder_rerun": False,
                "frozen_unary": True,
                "target_masks_opened_by_method": False,
                "query_intent": "instance",
                "selection_mode": selection_mode.value,
                "graph_policy": str(args.graph_policy),
                "channel_confidence_mode": str(args.channel_confidence_mode),
                "support_graph_name": str(args.support_graph_name),
                "solver": {
                    "iterations": int(config.iterations),
                    "residual": float(config.residual),
                    "unary_temperature": float(config.unary_temperature),
                    "support_threshold": float(config.support_threshold),
                },
                "scene_audit": scene_audit,
            },
        }
    )
    output = output_root / "track_b_selection.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_query"},
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--annotations-root", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--support-graph-name", default="shared_support_graph_k16.pt"
    )
    parser.add_argument("--graph-policy", default="typed_if_available")
    parser.add_argument(
        "--channel-confidence-mode",
        choices=("none", "affinity_mass", "max_affinity"),
        default="none",
    )
    parser.add_argument(
        "--selection-mode",
        choices=tuple(mode.value for mode in SelectionMode),
        default=SelectionMode.TOP_COMPONENT.value,
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--residual", type=float, default=0.30)
    parser.add_argument("--unary-temperature", type=float, default=0.10)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--maximum-distance-m", type=float, default=0.10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
