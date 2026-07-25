#!/usr/bin/env python3
"""Write a label-free scene-exclusion manifest for global crop-adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _physical_space(scene: str) -> str:
    return str(scene).split("_", 1)[0]


def build(args: argparse.Namespace) -> dict:
    names: set[str] = set()
    sources: list[dict[str, str]] = []
    if str(args.pfpr_benchmark_dir).strip():
        path = Path(args.pfpr_benchmark_dir) / "manifest.public.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("benchmark_version") != "scannet-pfpr-small-v1":
            raise ValueError("PFPR exclusion source is not a frozen v1 public manifest")
        names.update(str(item["scene_id"]) for item in payload.get("scene_domains", []))
        sources.append({"kind": "pfpr_public_scene_domains_only", "path": str(path.resolve())})
    if str(args.agile_root).strip():
        path = Path(args.agile_root) / "single" / "object_ids.npy"
        object_ids = np.load(path, allow_pickle=False)
        if object_ids.ndim != 2 or object_ids.shape[1] < 1:
            raise ValueError("AGILE object IDs are malformed")
        # Only the first string column is read to exclude scan identities.  No
        # class, mask, click, metric, or object-target information is emitted.
        names.update(str(item[0]) for item in object_ids)
        sources.append({"kind": "agile_scene_column_only", "path": str(path.resolve())})
    names.update(
        item for item in str(args.extra_scene_names).replace(",", " ").split() if item
    )
    if not names:
        raise ValueError("at least one evaluation scene must be excluded")
    payload = {
        "schema_version": 1,
        "purpose": "global_crop_spatial_adapter_scene_exclusion",
        "scene_names": sorted(names),
        "physical_spaces": sorted({_physical_space(name) for name in names}),
        "sources": sources,
        "uses_labels": False,
        "uses_masks": False,
        "uses_clicks": False,
        "uses_metrics": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"output": str(output.resolve()), "sha256": digest, "scene_count": len(names), "physical_space_count": len(payload["physical_spaces"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfpr-benchmark-dir", default="")
    parser.add_argument("--agile-root", default="")
    parser.add_argument("--extra-scene-names", default="")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
