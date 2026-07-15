#!/usr/bin/env python3
"""Attach reconstructed fallback rows to a frozen primary support graph.

The primary graph and its transition weights are copied exactly.  Each new
fallback primitive receives directed, query-independent edges to nearby
primary primitives, weighted by geometry and the official DINOv3/SAM3 RADIO
adaptors.  Consequently support can flow from the established field into
completed rows, while no fallback row can perturb the old primary solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_canonical_support_graph import (
    deterministic_feature_hash,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_primary_anchored_completion_graph(
    *,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    primary_valid: torch.Tensor,
    primary_global_rows: torch.Tensor,
    primary_graph: PrimitiveSupportGraph,
    appearance_features: torch.Tensor,
    boundary_features: torch.Tensor,
    neighbors: int,
    spatial_scale: float,
    appearance_temperature: float,
    boundary_temperature: float,
) -> tuple[PrimitiveSupportGraph, torch.Tensor, dict]:
    """Return an all-valid graph with an invariant primary transition block."""

    from scipy.spatial import cKDTree

    points = torch.as_tensor(xyz).float().cpu()
    support = torch.as_tensor(valid).bool().cpu().reshape(-1)
    primary = torch.as_tensor(primary_valid).bool().cpu().reshape(-1)
    old_rows = torch.as_tensor(primary_global_rows).long().cpu().reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    if support.shape != (points.shape[0],) or primary.shape != support.shape:
        raise ValueError("valid masks must align with xyz")
    if bool((primary & ~support).any()):
        raise ValueError("primary rows must be a subset of valid rows")
    expected_primary = torch.where(primary)[0]
    if not torch.equal(old_rows, expected_primary):
        raise ValueError("primary graph rows must exactly match primary_valid")
    if primary_graph.num_nodes != int(old_rows.numel()):
        raise ValueError("primary graph node count does not match primary rows")
    if int(neighbors) <= 0 or min(
        float(spatial_scale),
        float(appearance_temperature),
        float(boundary_temperature),
    ) <= 0:
        raise ValueError("graph scales and neighbors must be positive")

    global_rows = torch.where(support)[0]
    fallback_rows = torch.where(support & ~primary)[0]
    global_to_local = torch.full((points.shape[0],), -1, dtype=torch.long)
    global_to_local[global_rows] = torch.arange(global_rows.numel())
    primary_local = global_to_local[old_rows]
    fallback_local = global_to_local[fallback_rows]

    primary_edges = primary_graph.edge_index
    remapped_primary_edges = torch.stack(
        [primary_local[primary_edges[0]], primary_local[primary_edges[1]]]
    )
    local_sigma = torch.zeros(global_rows.numel(), dtype=torch.float32)
    local_sigma[primary_local] = primary_graph.local_sigma.float()

    app = F.normalize(torch.as_tensor(appearance_features).float().cpu(), dim=-1)
    boundary = F.normalize(
        torch.as_tensor(boundary_features).float().cpu(), dim=-1
    )
    expected_feature_rows = int(global_rows.numel())
    if app.ndim != 2 or boundary.ndim != 2:
        raise ValueError("capability features must be matrices")
    if app.shape[0] != expected_feature_rows or boundary.shape[0] != expected_feature_rows:
        raise ValueError("capability features must follow all-valid local row order")

    if fallback_rows.numel():
        k = min(int(neighbors), int(old_rows.numel()))
        distances_np, neighbor_np = cKDTree(points[old_rows].numpy()).query(
            points[fallback_rows].numpy(), k=k
        )
        distances = torch.from_numpy(
            np.asarray(distances_np, dtype=np.float32).reshape(fallback_rows.numel(), k)
        )
        neighbor_primary = torch.from_numpy(
            np.asarray(neighbor_np, dtype=np.int64).reshape(fallback_rows.numel(), k)
        ).long()
        fallback_sigma = distances.median(dim=1).values.mul(float(spatial_scale))
        fallback_sigma.clamp_(min=1e-4)
        local_sigma[fallback_local] = fallback_sigma
        fallback_edge_rows = fallback_local[:, None].expand(-1, k).reshape(-1)
        fallback_edge_cols = primary_local[neighbor_primary.reshape(-1)]
        fallback_edges = torch.stack([fallback_edge_rows, fallback_edge_cols])

        source_sigma = fallback_sigma[:, None].expand(-1, k).reshape(-1)
        target_sigma = primary_graph.local_sigma[neighbor_primary.reshape(-1)].float()
        geometry_log = -0.5 * distances.square().reshape(-1) / (
            source_sigma * target_sigma
        ).clamp_min(1e-8)
        app_cosine = (
            app[fallback_edge_rows] * app[fallback_edge_cols]
        ).sum(dim=-1)
        appearance_log = (app_cosine - 1.0) / float(appearance_temperature)
        boundary_cosine = (
            boundary[fallback_edge_rows] * boundary[fallback_edge_cols]
        ).sum(dim=-1)
        boundary_log = (boundary_cosine - 1.0) / float(boundary_temperature)
        fallback_channels = {
            "geometry": geometry_log.clamp(min=-60.0, max=0.0).exp(),
            "appearance": appearance_log.clamp(min=-60.0, max=0.0).exp(),
            "boundary": boundary_log.clamp(min=-60.0, max=0.0).exp(),
        }
        fallback_raw = (
            geometry_log + appearance_log + boundary_log
        ).clamp(min=-60.0, max=0.0).exp()
        row_sum = torch.zeros(global_rows.numel(), dtype=torch.float32)
        row_sum.index_add_(0, fallback_edge_rows, fallback_raw)
        fallback_weight = fallback_raw / row_sum[fallback_edge_rows].clamp_min(1e-12)
    else:
        fallback_edges = torch.empty(2, 0, dtype=torch.long)
        fallback_raw = torch.empty(0)
        fallback_weight = torch.empty(0)
        fallback_channels = {
            "geometry": torch.empty(0),
            "appearance": torch.empty(0),
            "boundary": torch.empty(0),
        }

    old_channels = dict(primary_graph.edge_channels)
    required_channels = set(fallback_channels)
    if set(old_channels) != required_channels:
        raise ValueError(
            "primary graph must contain exactly geometry/appearance/boundary channels"
        )
    graph = PrimitiveSupportGraph(
        edge_index=torch.cat([remapped_primary_edges, fallback_edges], dim=1),
        edge_weight=torch.cat(
            [primary_graph.edge_weight.float(), fallback_weight], dim=0
        ),
        raw_affinity=torch.cat(
            [primary_graph.raw_affinity.float(), fallback_raw], dim=0
        ),
        local_sigma=local_sigma,
        num_nodes=int(global_rows.numel()),
        edge_channels={
            name: torch.cat(
                [old_channels[name].float(), fallback_channels[name]], dim=0
            )
            for name in sorted(required_channels)
        },
    )
    primary_edge_count = int(primary_edges.shape[1])
    if not torch.equal(
        graph.edge_weight[:primary_edge_count], primary_graph.edge_weight.float()
    ):
        raise AssertionError("primary transition weights changed during augmentation")
    stats = {
        "num_global_rows": int(points.shape[0]),
        "valid_nodes": int(global_rows.numel()),
        "primary_nodes": int(old_rows.numel()),
        "fallback_nodes": int(fallback_rows.numel()),
        "primary_edges": primary_edge_count,
        "fallback_anchor_edges": int(fallback_edges.shape[1]),
        "neighbors": int(neighbors),
        "primary_transition_exact": True,
        "fallback_to_primary_only": True,
    }
    return graph, global_rows, stats


def build(args: argparse.Namespace) -> dict:
    primary_cache = torch.load(args.primary_support_graph, map_location="cpu")
    capability = torch.load(args.capability_cache, map_location="cpu")
    completion = torch.load(args.completion_feature_cache, map_location="cpu")
    valid = torch.as_tensor(completion["valid"]).bool().cpu()
    primary_valid = torch.as_tensor(completion["primary_valid"]).bool().cpu()
    if not torch.equal(valid, torch.as_tensor(capability["valid"]).bool().cpu()):
        raise ValueError("capability and completion valid masks differ")
    xyz = torch.as_tensor(completion["xyz"]).float().cpu()
    if not torch.allclose(xyz, torch.as_tensor(capability["xyz"]).float().cpu()):
        raise ValueError("capability and completion xyz differ")
    all_rows = torch.where(valid)[0]
    appearance = deterministic_feature_hash(
        torch.as_tensor(capability["appearance_dino_v3"])[all_rows],
        int(args.affinity_dim),
        batch_size=int(args.hash_batch_size),
    )
    boundary = deterministic_feature_hash(
        torch.as_tensor(capability["boundary_sam3"])[all_rows],
        int(args.affinity_dim),
        batch_size=int(args.hash_batch_size),
    )
    primary_graph = PrimitiveSupportGraph(
        edge_index=primary_cache["edge_index"],
        edge_weight=torch.as_tensor(primary_cache["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(primary_cache["raw_affinity"]).float(),
        local_sigma=torch.as_tensor(primary_cache["local_sigma"]).float(),
        num_nodes=int(torch.as_tensor(primary_cache["global_rows"]).numel()),
        edge_channels={
            str(name): torch.as_tensor(values).float()
            for name, values in dict(primary_cache["edge_channels"]).items()
        },
    )
    graph, global_rows, stats = build_primary_anchored_completion_graph(
        xyz=xyz,
        valid=valid,
        primary_valid=primary_valid,
        primary_global_rows=primary_cache["global_rows"],
        primary_graph=primary_graph,
        appearance_features=appearance,
        boundary_features=boundary,
        neighbors=int(args.neighbors),
        spatial_scale=float(args.spatial_scale),
        appearance_temperature=float(args.appearance_temperature),
        boundary_temperature=float(args.boundary_temperature),
    )
    metadata = {
        "schema_version": 1,
        "source": "frozen_primary_graph_plus_directed_official_adaptor_fallback_anchors",
        "primary_support_graph": str(Path(args.primary_support_graph).resolve()),
        "primary_support_graph_sha256": _sha256_file(args.primary_support_graph),
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "completion_feature_cache": str(Path(args.completion_feature_cache).resolve()),
        "feature_hash": {
            "algorithm": "signed_multiplicative_hash",
            "output_dim": int(args.affinity_dim),
            "query_independent": True,
        },
        "stats": stats,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "global_rows": global_rows,
            "num_global_rows": int(valid.numel()),
            "xyz": xyz[global_rows],
            "edge_index": graph.edge_index,
            "edge_weight": graph.edge_weight.half(),
            "raw_affinity": graph.raw_affinity.half(),
            "edge_channels": {
                name: values.half() for name, values in graph.edge_channels.items()
            },
            "local_sigma": graph.local_sigma,
            "metadata": metadata,
        },
        output,
    )
    report = {**metadata, "output": str(output)}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-support-graph", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--completion-feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--spatial-scale", type=float, default=2.0)
    parser.add_argument("--appearance-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-temperature", type=float, default=0.10)
    parser.add_argument("--affinity-dim", type=int, default=256)
    parser.add_argument("--hash-batch-size", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
