#!/usr/bin/env python3
"""Compare two row-aligned official capability caches without task labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "rows": int(values.numel()),
        "mean_cosine": float(values.mean()),
        "p05_cosine": float(values.quantile(0.05)),
        "p01_cosine": float(values.quantile(0.01)),
    }


def _graph_summary(reference_path: str, candidate_path: str) -> dict:
    if not reference_path or not candidate_path:
        return {}
    reference = torch.load(reference_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    reference_edges = torch.as_tensor(reference["edge_index"]).long()
    candidate_edges = torch.as_tensor(candidate["edge_index"]).long()
    if not torch.equal(reference_edges, candidate_edges):
        raise ValueError("support graphs do not share exact edge topology")
    output = {"edges": int(reference_edges.shape[1]), "topology_identical": True}
    for name in ("edge_weight", "raw_affinity"):
        left = torch.as_tensor(reference[name]).float()
        right = torch.as_tensor(candidate[name]).float()
        centered_left = left - left.mean()
        centered_right = right - right.mean()
        pearson = (centered_left * centered_right).mean() / (
            centered_left.square().mean().sqrt()
            * centered_right.square().mean().sqrt()
        ).clamp_min(1e-12)
        output[name] = {
            "mean_absolute_error": float((left - right).abs().mean()),
            "pearson": float(pearson),
            "reference_mean": float(left.mean()),
            "candidate_mean": float(right.mean()),
        }
    return output


def audit(args: argparse.Namespace) -> dict:
    reference = load_canonical_capability_bank(args.reference_cache)
    candidate = load_canonical_capability_bank(args.candidate_cache)
    if not torch.equal(reference.valid, candidate.valid):
        raise ValueError("capability caches have different valid rows")
    if reference.xyz.shape != candidate.xyz.shape or not torch.allclose(
        reference.xyz, candidate.xyz, atol=1e-6, rtol=0.0
    ):
        raise ValueError("capability caches have different geometry")
    rows = reference.global_rows
    spaces = {}
    for name, left, right in (
        ("official_dino_v3", reference.appearance, candidate.appearance),
        ("official_sam3", reference.boundary, candidate.boundary),
    ):
        chunks = []
        for start in range(0, rows.numel(), int(args.chunk_size)):
            selected = rows[start : start + int(args.chunk_size)]
            chunks.append(
                F.cosine_similarity(
                    left[selected].float(), right[selected].float(), dim=-1, eps=1e-8
                )
            )
        spaces[name] = _summary(torch.cat(chunks))
    report = {
        "schema_version": 1,
        "audit": "query_free_official_capability_cache_fidelity",
        "reference_cache": str(Path(args.reference_cache).resolve()),
        "candidate_cache": str(Path(args.candidate_cache).resolve()),
        "geometry_rows_verified": True,
        "spaces": spaces,
        "graph": _graph_summary(args.reference_graph, args.candidate_graph),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "query_prompts_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--reference-graph", default="")
    parser.add_argument("--candidate-graph", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
