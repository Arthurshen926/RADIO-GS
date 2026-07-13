#!/usr/bin/env python3
"""Build the shared query-independent 3D support graph from official views."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    build_primitive_support_graph,
)


def deterministic_feature_hash(
    features: torch.Tensor,
    output_dim: int,
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Signed feature hashing for query-free approximate cosine affinities."""
    values = torch.as_tensor(features)
    if values.ndim != 2 or output_dim <= 0:
        raise ValueError("features must be [N,D] and output_dim must be positive")
    input_dim = int(values.shape[1])
    index = torch.arange(input_dim, dtype=torch.long)
    hashed = index * 2654435761 + 2246822519
    buckets = torch.remainder(hashed, int(output_dim))
    signs = torch.where(
        torch.bitwise_and(hashed, 1) == 0,
        torch.ones(input_dim),
        -torch.ones(input_dim),
    )
    result = torch.empty(values.shape[0], int(output_dim), dtype=torch.float32)
    expanded_buckets = buckets.unsqueeze(0)
    for start in range(0, values.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), values.shape[0])
        batch = values[start:stop].float()
        projected = torch.zeros(batch.shape[0], int(output_dim), dtype=torch.float32)
        projected.scatter_add_(
            1,
            expanded_buckets.expand(batch.shape[0], -1),
            batch * signs,
        )
        result[start:stop] = F.normalize(projected, dim=-1, eps=1e-8)
    return result


def build(args: argparse.Namespace) -> dict:
    capability = torch.load(args.capability_cache, map_location="cpu")
    required = {"xyz", "valid", "appearance_dino_v3", "boundary_sam3", "metadata"}
    if not isinstance(capability, dict) or not required.issubset(capability):
        raise ValueError(f"capability cache must contain {sorted(required)}")
    valid = torch.as_tensor(capability["valid"]).bool().cpu()
    global_rows = torch.where(valid)[0]
    xyz = torch.as_tensor(capability["xyz"]).float().cpu()[global_rows]
    appearance = deterministic_feature_hash(
        torch.as_tensor(capability["appearance_dino_v3"])[global_rows],
        int(args.affinity_dim),
        batch_size=int(args.hash_batch_size),
    )
    boundary = deterministic_feature_hash(
        torch.as_tensor(capability["boundary_sam3"])[global_rows],
        int(args.affinity_dim),
        batch_size=int(args.hash_batch_size),
    )
    config = SupportGraphConfig(
        neighbors=int(args.neighbors),
        spatial_scale=float(args.spatial_scale),
        appearance_temperature=float(args.appearance_temperature),
        boundary_temperature=float(args.boundary_temperature),
        normal_temperature=0.20,
        covisibility_weight=0.0,
        affinity_chunk_size=int(args.affinity_chunk_size),
    )
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=appearance,
        boundary_features=boundary,
        config=config,
    )
    metadata = {
        "schema_version": 1,
        "source": "canonical_official_dino_sam3_shared_support_graph",
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "capability_metadata": capability["metadata"],
        "feature_hash": {
            "algorithm": "signed_multiplicative_hash",
            "output_dim": int(args.affinity_dim),
            "query_independent": True,
        },
        "graph_config": asdict(config),
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
            "xyz": xyz,
            "edge_index": graph.edge_index,
            "edge_weight": graph.edge_weight.half(),
            "raw_affinity": graph.raw_affinity.half(),
            "local_sigma": graph.local_sigma,
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--spatial-scale", type=float, default=2.0)
    parser.add_argument("--appearance-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-temperature", type=float, default=0.10)
    parser.add_argument("--affinity-dim", type=int, default=256)
    parser.add_argument("--hash-batch-size", type=int, default=8192)
    parser.add_argument("--affinity-chunk-size", type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
