#!/usr/bin/env python3
"""Audit raw Dijkstra regions before a fixed token budget truncates them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph(payload: dict) -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"], edge_weight=payload["edge_weight"],
        raw_affinity=payload["raw_affinity"], local_sigma=payload["local_sigma"],
        num_nodes=int(torch.as_tensor(payload["xyz"]).shape[0]),
        edge_channels=payload.get("edge_channels", {}),
    )


def _summary(values: list[float | int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()), "mean": float(array.mean()), "max": float(array.max()),
        "p50": float(np.quantile(array, 0.50)), "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)),
    }


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.support_graph, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(payload["xyz"]).float()
    graph = _graph(payload)
    contract = SurfaceRegionContractV2(
        radii_m=tuple(float(value) for value in str(args.region_radii).replace(",", " ").split()),
        context_ratio=float(args.context_ratio), neighbors=int(args.graph_neighbors),
        maximum_tokens=int(args.maximum_tokens), minimum_tokens=1,
        path_cost_mode=str(args.path_cost_mode),
        path_affinity_floor=float(args.path_affinity_floor),
        token_subsampling=str(args.token_subsampling),
        token_candidate_limit=int(args.token_candidate_limit),
        core_token_fraction=float(args.core_token_fraction),
    )
    matrix = contract.prepare_graph(graph, xyz)
    rng = random.Random(int(args.seed))
    anchors = rng.sample(range(graph.num_nodes), min(int(args.anchors), graph.num_nodes))
    per_scale: dict[str, dict] = {}
    for radius in contract.radii_m:
        raw_count: list[int] = []
        raw_core: list[int] = []
        raw_context: list[int] = []
        selected_count: list[int] = []
        selected_context: list[int] = []
        raw_max_distance: list[float] = []
        selected_max_distance: list[float] = []
        reach_ratio: list[float] = []
        truncated = 0
        for anchor in anchors:
            distance = dijkstra(
                matrix, directed=False, indices=int(anchor),
                limit=float(radius) * contract.context_ratio,
            )
            finite = np.isfinite(distance)
            raw_dist = distance[finite]
            core = raw_dist <= float(radius) + 1e-7
            raw_count.append(int(finite.sum()))
            raw_core.append(int(core.sum()))
            raw_context.append(int((~core).sum()))
            raw_max_distance.append(float(raw_dist.max()))
            rows, selected_core_mask, selected_distance = contract.expand(
                graph, xyz, anchor, radius, prepared_graph=matrix
            )
            selected_count.append(int(len(rows)))
            selected_context.append(int((~selected_core_mask).sum()))
            selected_max = float(selected_distance.max())
            selected_max_distance.append(selected_max)
            reach_ratio.append(selected_max / max(float(raw_dist.max()), 1e-12))
            truncated += int(finite.sum() > contract.maximum_tokens)
        per_scale[str(radius)] = {
            "raw_token_count": _summary(raw_count),
            "raw_core_token_count": _summary(raw_core),
            "raw_context_token_count": _summary(raw_context),
            "selected_token_count": _summary(selected_count),
            "selected_context_token_count": _summary(selected_context),
            "raw_max_geodesic_distance_m": _summary(raw_max_distance),
            "selected_max_geodesic_distance_m": _summary(selected_max_distance),
            "selected_to_raw_radius_ratio": _summary(reach_ratio),
            "truncated_region_fraction": float(truncated / len(anchors)),
        }
    report = {
        "schema_version": 1,
        "audit": "surface_region_raw_token_and_context_truncation",
        "support_graph": str(Path(args.support_graph).resolve()),
        "anchors": len(anchors), "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest, "per_scale": per_scale,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=int, default=128)
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
    parser.add_argument("--seed", type=int, default=0)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
