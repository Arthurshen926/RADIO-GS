#!/usr/bin/env python3
"""Apply one frozen 3D support solver to a row-aligned query score cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from radio_gs.interfaces.capability_cache import (
    load_canonical_primitive_reliability,
)
from radio_gs.querying.evidence_scorer import shrink_unary_by_reliability
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    solve_primitive_support,
)


def apply(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    score_cache = torch.load(args.query_score_cache, map_location="cpu")
    graph_cache = torch.load(args.support_graph_cache, map_location="cpu")
    scores = torch.as_tensor(score_cache["features"]).float().cpu()
    valid = torch.as_tensor(score_cache["valid"]).bool().cpu()
    global_rows = torch.as_tensor(graph_cache["global_rows"]).long().cpu()
    if scores.ndim != 2 or valid.shape != (scores.shape[0],):
        raise ValueError("query scores must be row-aligned [N,Q]")
    if int(graph_cache["num_global_rows"]) != scores.shape[0]:
        raise ValueError("support graph and query score global row counts differ")
    if not torch.equal(torch.where(valid)[0], global_rows):
        raise ValueError("support graph nodes must exactly match valid query-score rows")
    primitive_reliability = None
    node_reliability = None
    if str(args.reliability_cache).strip():
        score_metadata = dict(score_cache.get("metadata", {}))
        primitive_reliability = load_canonical_primitive_reliability(
            args.reliability_cache,
            expected_xyz=torch.as_tensor(score_cache["xyz"]).float(),
            expected_valid=valid,
            expected_field_checkpoint_sha256=str(
                score_metadata.get("field_checkpoint_sha256", "")
            ),
        )
        node_reliability = primitive_reliability.valid_confidence().to(device)
    graph = PrimitiveSupportGraph(
        edge_index=graph_cache["edge_index"],
        edge_weight=torch.as_tensor(graph_cache["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(graph_cache["raw_affinity"]).float(),
        local_sigma=graph_cache["local_sigma"],
        num_nodes=int(global_rows.numel()),
        edge_channels={
            str(name): torch.as_tensor(values).float()
            for name, values in dict(graph_cache.get("edge_channels", {})).items()
        },
    ).to(device)
    config = SupportSolverConfig(
        iterations=int(args.iterations),
        residual=float(args.residual),
        unary_temperature=float(args.unary_temperature),
        support_threshold=float(args.support_threshold),
    )
    solved = torch.zeros_like(scores, dtype=torch.float16)
    for query in range(scores.shape[1]):
        unary = scores[global_rows, query].to(device) - float(args.unary_center)
        if node_reliability is not None:
            unary = shrink_unary_by_reliability(unary, node_reliability)
        probability = solve_primitive_support(graph, unary, config=config)
        solved[global_rows, query] = probability.half().cpu()
    metadata = {
        **dict(score_cache.get("metadata", {})),
        "construction": "shared_3d_support_solver_probabilities",
        "unary_query_score_cache": str(Path(args.query_score_cache).resolve()),
        "support_graph_cache": str(Path(args.support_graph_cache).resolve()),
        "support_solver_config": asdict(config),
        "unary_center": float(args.unary_center),
        "solver_device": str(device),
        "benchmark_masks_opened": False,
        "primitive_reliability": (
            {
                "cache": str(Path(args.reliability_cache).resolve()),
                "formula": primitive_reliability.metadata.get("formula"),
                "centered_unary_shrink": True,
                "uses_query_or_target_labels": False,
            }
            if primitive_reliability is not None
            else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "xyz": score_cache["xyz"],
        "geometry_fingerprint": score_cache.get("geometry_fingerprint", {}),
        "features": solved,
        "valid": valid,
        "view_counts": score_cache.get("view_counts"),
        "reliability": score_cache.get("reliability"),
        "metadata": metadata,
    }
    # Preserve the query-independent completion audit contract.  The solver
    # changes only query probabilities; it must not erase which rows came from
    # the frozen primary field or how reliable reconstructed fallback rows are.
    for key in ("primary_valid", "semantic_confidence"):
        if key in score_cache:
            payload[key] = score_cache[key]
    torch.save(payload, output)
    report = {
        "output": str(output),
        "num_queries": int(scores.shape[1]),
        "valid_gaussians": int(valid.sum()),
        "metadata": metadata,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-score-cache", required=True)
    parser.add_argument("--support-graph-cache", required=True)
    parser.add_argument("--reliability-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--residual", type=float, default=0.30)
    parser.add_argument("--unary-temperature", type=float, default=0.10)
    parser.add_argument("--unary-center", type=float, default=0.60)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device only; it does not change the support algorithm.",
    )
    args = parser.parse_args()
    print(json.dumps(apply(args), indent=2))


if __name__ == "__main__":
    main()
