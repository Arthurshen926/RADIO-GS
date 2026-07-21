#!/usr/bin/env python3
"""Summarize query-free scale-ordered relation teacher quality by scene.

The audit deliberately separates *coverage* (how many graph edges received a
constraint) from *consistency* (whether a weighted lower and upper merge-scale
bound agree).  It never opens semantic labels or selects a downstream test
protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    for value in str(raw).replace(",", " ").split():
        paths.extend(sorted(Path().glob(value)) if any(mark in value for mark in "*?[") else [Path(value)])
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("scale-ordered relation cache list is empty or missing")
    return paths


def summarize_cache(path: Path, *, require_raster_responsibility: bool) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError(f"{path} is not a schema-v2 scale-ordered relation cache")
    if metadata.get("teacher") != "official_sam3_multimask_scale_ordered_regions":
        raise ValueError(f"{path} has an unexpected relation teacher")
    if any(metadata.get(key, True) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError(f"{path} violates query-free teacher provenance")
    if require_raster_responsibility and not bool(metadata.get("raster_responsibility_used", False)):
        raise ValueError(f"{path} did not use geometry-aligned raster responsibility")
    same = torch.as_tensor(payload["same_mass"]).float()
    separate = torch.as_tensor(payload["separate_mass"]).float()
    entropy = torch.as_tensor(payload["constraint_entropy"]).float()
    lower = torch.as_tensor(payload["has_lower"]).bool()
    upper = torch.as_tensor(payload["has_upper"]).bool()
    consistent = torch.as_tensor(payload["interval_consistent"]).bool()
    if not (same.shape == separate.shape == entropy.shape == lower.shape == upper.shape == consistent.shape):
        raise ValueError(f"{path} has malformed edge-aligned interval fields")
    constrained = (same + separate) > 0
    both = lower & upper
    return {
        "cache": str(path.resolve()), "scene": str(payload.get("scene", "")),
        "membership_lifting": str(metadata.get("membership_lifting", "")),
        "raster_responsibility_used": bool(metadata.get("raster_responsibility_used", False)),
        "edges": int(len(same)), "constrained_edges": int(constrained.sum()),
        "constrained_edge_fraction": float(constrained.float().mean()),
        "both_bounds_edges": int(both.sum()),
        "both_bounds_fraction": float(both.float().mean()),
        "interval_consistent_fraction_among_both": float(
            consistent[both].float().mean() if bool(both.any()) else 0.0
        ),
        "constraint_entropy": {
            "mean_constrained": float(entropy[constrained].mean() if bool(constrained.any()) else 0.0),
            "p90_constrained": float(entropy[constrained].quantile(0.90) if bool(constrained.any()) else 0.0),
        },
        "same_mass": float(same.sum()), "separate_mass": float(separate.sum()),
        "mask_frames": len(metadata.get("mask_frames", [])),
        "mask_schema_versions": metadata.get("mask_schema_versions", []),
    }


def run(args: argparse.Namespace) -> dict:
    rows = [
        summarize_cache(path, require_raster_responsibility=bool(args.require_raster_responsibility))
        for path in _paths(args.caches)
    ]
    for key in ("scene",):
        values = [row[key] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError("one scale-ordered teacher cache per scene is required")
    macro_keys = (
        "constrained_edge_fraction", "both_bounds_fraction",
        "interval_consistent_fraction_among_both",
    )
    report = {
        "schema_version": 1, "diagnostic_only": True,
        "require_raster_responsibility": bool(args.require_raster_responsibility),
        "scenes": rows,
        "macro": {key: float(sum(row[key] for row in rows) / len(rows)) for key in macro_keys},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-raster-responsibility", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
