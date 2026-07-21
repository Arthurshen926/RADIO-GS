#!/usr/bin/env python3
"""GT-only diagnostic for scale-conditioned one-click surface regions.

This is not an evaluation result.  It asks whether a query-free support graph
contains a useful *anchor-local* part/object/context candidate before a text or
prompt solver is trained on top of it.  ScanNet instances are opened only after
the graph and all Dijkstra regions have been constructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.eval_scannet_3d_point_query import load_scannet_instances
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply


def _support_graph(payload: dict) -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"], edge_weight=payload["edge_weight"],
        raw_affinity=payload["raw_affinity"], local_sigma=payload["local_sigma"],
        num_nodes=int(torch.as_tensor(payload["xyz"]).shape[0]),
        edge_channels=payload.get("edge_channels", {}),
    )


def _contract(args: argparse.Namespace) -> SurfaceRegionContractV2:
    return SurfaceRegionContractV2(
        radii_m=tuple(float(value) for value in str(args.region_radii).replace(",", " ").split()),
        context_ratio=float(args.context_ratio),
        neighbors=int(args.graph_neighbors),
        maximum_tokens=int(args.maximum_tokens),
        minimum_tokens=1,
        path_cost_mode=str(args.path_cost_mode),
        path_affinity_floor=float(args.path_affinity_floor),
        token_subsampling=str(args.token_subsampling),
        token_candidate_limit=int(args.token_candidate_limit),
        core_token_fraction=float(args.core_token_fraction),
    )


def _instance_queries(
    instance_ids: np.ndarray,
    metadata: dict,
    mesh_leaf: np.ndarray,
    *,
    minimum_vertices: int,
    seed: int,
    maximum_instances: int,
) -> list[dict]:
    queries: list[dict] = []
    for instance_id in sorted(metadata):
        target_rows = np.flatnonzero(instance_ids == int(instance_id))
        if target_rows.size < int(minimum_vertices):
            continue
        rng = np.random.default_rng(int(seed) + 1_000_003 * int(instance_id))
        seed_vertex = int(target_rows[int(rng.integers(0, target_rows.size))])
        queries.append({
            "instance_id": int(instance_id),
            "label": str(metadata[instance_id]["label"]),
            "target_rows": target_rows,
            "seed_vertex": seed_vertex,
            "seed_leaf": int(mesh_leaf[seed_vertex]),
        })
    if maximum_instances > 0:
        queries = queries[: int(maximum_instances)]
    if not queries:
        raise RuntimeError("no instance satisfies the diagnostic minimum")
    return queries


def _region_iou(
    selected_rows: np.ndarray,
    mesh_leaf: np.ndarray,
    target_rows: np.ndarray,
    primitive_count: int,
) -> tuple[float, int]:
    selected = np.zeros(int(primitive_count), dtype=bool)
    selected[np.asarray(selected_rows, dtype=np.int64)] = True
    predicted = selected[mesh_leaf]
    target = np.zeros(mesh_leaf.shape[0], dtype=bool)
    target[target_rows] = True
    intersection = int(np.logical_and(predicted, target).sum())
    union = int(np.logical_or(predicted, target).sum())
    return float(intersection / union) if union else 0.0, int(predicted.sum())


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.support_graph, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    graph = _support_graph(payload)
    contract = _contract(args)
    prepared = contract.prepare_graph(graph, xyz)

    mesh_xyz, _ = _read_label_ply(args.label_ply)
    instance_ids, metadata = load_scannet_instances(args.aggregation, args.segmentation)
    if mesh_xyz.shape[0] != instance_ids.shape[0]:
        raise ValueError("ScanNet label mesh and instance annotations differ in length")
    mesh_leaf = cKDTree(xyz.numpy()).query(mesh_xyz, k=1, workers=-1)[1].astype(np.int64)
    queries = _instance_queries(
        instance_ids, metadata, mesh_leaf,
        minimum_vertices=int(args.minimum_instance_vertices),
        seed=int(args.seed), maximum_instances=int(args.maximum_instances),
    )

    all_iou = {float(radius): [] for radius in contract.radii_m}
    details = [
        {
            "instance_id": query["instance_id"], "label": query["label"],
            "seed_vertex": query["seed_vertex"], "seed_leaf": query["seed_leaf"],
            "per_scale": {},
        }
        for query in queries
    ]
    anchors = [query["seed_leaf"] for query in queries]
    for radius in contract.radii_m:
        regions: list[tuple[np.ndarray, int, int]] = []
        if args.membership_mode == "raw_core":
            for anchor in anchors:
                distances = dijkstra(
                    prepared, directed=False, indices=int(anchor), limit=float(radius)
                )
                rows = np.flatnonzero(np.isfinite(distances))
                regions.append((rows, int(rows.size), 0))
        else:
            for start in range(0, len(anchors), int(args.batch_size)):
                for rows, core, _distance in contract.expand_batch(
                    graph, xyz, anchors[start:start + int(args.batch_size)], radius,
                    prepared_graph=prepared,
                ):
                    core_rows = torch.as_tensor(rows)[torch.as_tensor(core).bool()].numpy()
                    regions.append((core_rows, int(core.sum()), int(len(rows) - int(core.sum()))))
        for index, (query, (selected_rows, core_count, context_count)) in enumerate(zip(queries, regions)):
            iou, predicted_vertices = _region_iou(
                selected_rows, mesh_leaf, query["target_rows"], len(xyz)
            )
            all_iou[float(radius)].append(iou)
            details[index]["per_scale"][str(radius)] = {
                "iou": iou,
                "core_primitives": core_count,
                "context_primitives": context_count,
                "predicted_mesh_vertices": predicted_vertices,
            }

    oracle = []
    for record in details:
        values = record["per_scale"]
        best_radius = max(values, key=lambda radius: values[radius]["iou"])
        record["scale_oracle_iou"] = float(values[best_radius]["iou"])
        record["scale_oracle_radius_m"] = float(best_radius)
        oracle.append(record["scale_oracle_iou"])
    report = {
        "schema_version": 1,
        "diagnostic_only_gt_oracle": True,
        "scene": str(args.scene),
        "support_graph": str(Path(args.support_graph).resolve()),
        "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest,
        "candidate_type": "one_click_anchor_local_dijkstra_core_region",
        "membership_mode": str(args.membership_mode),
        "mesh_projection": "nearest_support_graph_primitive",
        "minimum_instance_vertices": int(args.minimum_instance_vertices),
        "num_instances": len(details),
        "macro_iou_by_radius": {
            str(radius): float(np.mean(values)) for radius, values in all_iou.items()
        },
        "macro_scale_oracle_iou": float(np.mean(oracle)),
        "instances": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--label-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region-radii", default="0.25,0.45,0.70")
    parser.add_argument("--context-ratio", type=float, default=1.20)
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument("--maximum-tokens", type=int, default=256)
    parser.add_argument(
        "--token-subsampling",
        choices=("nearest_geodesic_then_node_index", "core_context_radial_stratified_v1"),
        default="nearest_geodesic_then_node_index",
    )
    parser.add_argument("--token-candidate-limit", type=int, default=256)
    parser.add_argument("--core-token-fraction", type=float, default=0.60)
    parser.add_argument(
        "--path-cost-mode",
        choices=("euclidean", "appearance_boundary_geometric"),
        default="euclidean",
    )
    parser.add_argument("--path-affinity-floor", type=float, default=1e-4)
    parser.add_argument("--minimum-instance-vertices", type=int, default=100)
    parser.add_argument(
        "--membership-mode", choices=("raw_core", "selected_tokens"),
        default="raw_core",
        help="Use full raw Dijkstra core for topology diagnosis, or the token budget ablation.",
    )
    parser.add_argument("--maximum-instances", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
