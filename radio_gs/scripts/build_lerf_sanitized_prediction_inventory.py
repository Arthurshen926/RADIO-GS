#!/usr/bin/env python3
"""Freeze LERF frame/category/shape metadata without retaining polygon coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from radio_gs.scripts.eval_lerf_direct_3d_selection import (  # noqa: E402
    OPEN_GAUSSIAN_LERF_FRAMES,
    sha256_file,
)
from radio_gs.scripts.eval_lerf_grounding import (  # noqa: E402
    load_lerf_ovs_labels,
    resolve_lerf_label_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    label_dir = resolve_lerf_label_dir(args.label_dir)
    annotations, categories, height, width = load_lerf_ovs_labels(label_dir, args.scene)
    official = set(OPEN_GAUSSIAN_LERF_FRAMES[args.scene])
    frames = [
        {
            "frame_id": int(frame_id),
            "categories": sorted({str(row["category"]) for row in objects}),
        }
        for frame_id, objects in sorted(annotations.items())
        if frame_id in official
    ]
    if {row["frame_id"] for row in frames} != official:
        raise ValueError("sanitized inventory does not cover the exact official frame set")
    scene_label_root = Path(label_dir).expanduser().resolve() / args.scene
    source_files = sorted(path for path in scene_label_root.rglob("*") if path.is_file())
    payload = {
        "schema_version": 1,
        "artifact_type": "lerf_sanitized_prediction_inventory_v1",
        "scene": args.scene,
        "categories": categories,
        "image_height": int(height),
        "image_width": int(width),
        "frames": frames,
        "contains_polygon_coordinates": False,
        "builder_opened_full_annotations": True,
        "prediction_process_must_not_open_label_dir": True,
        "source_label_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_files
        ],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite different inventory: {output}")
    else:
        output.write_bytes(encoded)
    print(output)


if __name__ == "__main__":
    main()
