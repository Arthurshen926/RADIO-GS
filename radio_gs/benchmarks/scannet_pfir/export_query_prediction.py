#!/usr/bin/env python3
"""Export one PFIR query cache to official-mesh Track-A and Track-B files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch

from radio_gs.benchmarks.scannet_pfir.evaluation.map_gaussians_to_mesh import (
    gaussian_scores_to_mesh,
)


def _atomic_npy_save(output: Path, array: np.ndarray) -> None:
    """Publish one evaluator input only after NumPy has finished its archive."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def export(args: argparse.Namespace) -> dict:
    cache = torch.load(args.query_cache, map_location="cpu")
    xyz = torch.as_tensor(cache["xyz"]).float().numpy()
    valid = torch.as_tensor(cache["valid"]).bool().numpy()
    unary = torch.as_tensor(cache["unary"]).float().numpy().reshape(-1)
    support = torch.as_tensor(cache["features"]).float().numpy().reshape(-1)
    if xyz.shape != (len(valid), 3) or unary.shape != valid.shape:
        raise ValueError("PFIR query cache rows do not align")
    if not bool(valid.any()):
        raise ValueError("PFIR query cache has no valid canonical rows")
    vertex = PlyData.read(args.mesh_ply)["vertex"].data
    mesh_xyz = np.column_stack(
        [vertex[name] for name in ("x", "y", "z")]
    ).astype(np.float32)
    ranking, ranking_valid = gaussian_scores_to_mesh(
        xyz[valid],
        unary[valid],
        mesh_xyz,
        neighbors=int(args.neighbors),
        maximum_distance_m=float(args.maximum_distance_m),
    )
    selection_score, selection_valid = gaussian_scores_to_mesh(
        xyz[valid],
        support[valid],
        mesh_xyz,
        neighbors=int(args.neighbors),
        maximum_distance_m=float(args.maximum_distance_m),
    )
    selection = (
        selection_valid & (selection_score >= float(args.support_threshold))
    )
    ranking_output = Path(args.ranking_output)
    selection_output = Path(args.selection_output)
    _atomic_npy_save(ranking_output, ranking.astype(np.float32))
    _atomic_npy_save(selection_output, selection.astype(bool))
    report = {
        "schema_version": 1,
        "query_cache": str(Path(args.query_cache).resolve()),
        "mesh_ply": str(Path(args.mesh_ply).resolve()),
        "ranking_output": str(ranking_output.resolve()),
        "selection_output": str(selection_output.resolve()),
        "mesh_vertices": len(mesh_xyz),
        "ranking_coverage": float(ranking_valid.mean()),
        "selection_coverage": float(selection_valid.mean()),
        "support_threshold": float(args.support_threshold),
        "track_a": "fused_unary_before_graph_or_threshold",
        "track_b": "frozen_solver_support_threshold",
        "instance_labels_opened": False,
        "queries_or_metrics_used_for_mapping": False,
    }
    ranking_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--mesh-ply", required=True)
    parser.add_argument("--ranking-output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--maximum-distance-m", type=float, default=0.10)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2))


if __name__ == "__main__":
    main()
