#!/usr/bin/env python3
"""Compile frozen dense semantic descriptors into query score maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def normalize_query_heatmaps_by_peak(scores: torch.Tensor) -> torch.Tensor:
    """Apply LERF's GT-free peak-relative convention to dense query maps."""
    if scores.ndim != 3:
        raise ValueError(f"expected [Q,H,W] scores, got {tuple(scores.shape)}")
    peaks = scores.flatten(1).amax(dim=1).clamp_min(1.0e-12)
    return (scores / peaks[:, None, None]).clamp_(0.0, 1.0)


def build(args: argparse.Namespace) -> dict:
    cache = torch.load(args.text_embedding_cache, map_location="cpu")
    queries = [str(value) for value in cache["queries"]]
    text = F.normalize(torch.as_tensor(cache["embeddings"]).float(), dim=-1)
    output_root = Path(args.output_root)
    reports = {}
    for scene in [value.strip() for value in args.scenes.split(",") if value.strip()]:
        source = Path(args.feature_root) / scene
        paths = sorted(source.glob("rgb_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no dense descriptors found for {scene}: {source}")
        destination = output_root / scene
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        for path in paths:
            descriptor = torch.load(path, map_location="cpu").float()
            channels, height, width = descriptor.shape
            if channels != text.shape[1]:
                raise ValueError("descriptor and text dimensions differ")
            pixels = F.normalize(descriptor.flatten(1).T, dim=-1, eps=1e-8)
            scores = torch.softmax(
                (pixels @ text.T) * float(args.temperature), dim=-1
            ).T.reshape(len(queries), height, width)
            if bool(args.peak_normalize):
                scores = normalize_query_heatmaps_by_peak(scores)
            torch.save(
                scores.half(),
                destination / path.name,
            )
            frames.append(int(path.stem.split("_")[-1]))
        reports[scene] = {"frames": frames, "num_frames": len(frames)}
    report = {
        "schema_version": 1,
        "feature_space": "dense_text_query_scores",
        "scoring": "scene_softmax",
        "temperature": float(args.temperature),
        "peak_normalize_per_view": bool(args.peak_normalize),
        "queries": queries,
        "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
        "feature_root": str(Path(args.feature_root).resolve()),
        "benchmark_masks_opened": False,
        "text_queries_opened": True,
        "scenes": reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument(
        "--peak-normalize",
        action="store_true",
        help="Normalize each query heatmap by its per-view peak before registration.",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
