#!/usr/bin/env python3
"""Compile query-independent primitive semantic features into text unaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def compile_scores(
    features: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    peak_normalize: bool,
) -> torch.Tensor:
    values = torch.as_tensor(features).float().cpu()
    queries = F.normalize(torch.as_tensor(text).float().cpu(), dim=-1, eps=1e-8)
    mask = torch.as_tensor(valid).bool().cpu()
    if values.ndim != 2 or queries.ndim != 2 or values.shape[1] != queries.shape[1]:
        raise ValueError("primitive features and text embeddings must be [N,D]/[Q,D]")
    if mask.shape != (values.shape[0],) or not bool(mask.any()):
        raise ValueError("valid must keep at least one primitive row")
    result = torch.zeros(values.shape[0], queries.shape[0], dtype=torch.float32)
    rows = torch.where(mask)[0]
    for start in range(0, rows.numel(), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        visual = F.normalize(values[selected], dim=-1, eps=1e-8)
        result[selected] = torch.softmax(
            (visual @ queries.T) * float(temperature), dim=-1
        )
    if peak_normalize:
        peaks = result[mask].amax(dim=0, keepdim=True).clamp_min(1e-12)
        result[mask] = (result[mask] / peaks).clamp_(0.0, 1.0)
    return result.half()


def build(args: argparse.Namespace) -> dict:
    feature_cache = torch.load(args.feature_cache, map_location="cpu")
    text_cache = torch.load(args.text_embedding_cache, map_location="cpu")
    scores = compile_scores(
        feature_cache["features"],
        text_cache["embeddings"],
        feature_cache["valid"],
        temperature=float(args.temperature),
        chunk_size=int(args.chunk_size),
        peak_normalize=bool(args.peak_normalize),
    )
    metadata = {
        "schema_version": 1,
        "feature_space": "primitive_text_query_scores",
        "construction": "scene_softmax_then_query_peak_normalize"
        if args.peak_normalize
        else "scene_softmax",
        "feature_cache": str(Path(args.feature_cache).resolve()),
        "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
        "temperature": float(args.temperature),
        "query_names": [str(value) for value in text_cache["queries"]],
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "xyz": feature_cache["xyz"],
            "geometry_fingerprint": feature_cache.get("geometry_fingerprint", {}),
            "features": scores,
            "valid": feature_cache["valid"],
            "metadata": metadata,
        },
        output,
    )
    report = {
        "output": str(output),
        "num_queries": int(scores.shape[1]),
        "valid_gaussians": int(torch.as_tensor(feature_cache["valid"]).sum()),
        "metadata": metadata,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--peak-normalize", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
