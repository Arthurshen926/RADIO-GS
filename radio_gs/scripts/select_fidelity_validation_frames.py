#!/usr/bin/env python3
"""Freeze deterministic, label-free held-out views for field fidelity gates.

The canonical field is fitted from the remaining registered observations.  The
selected views are never chosen from a benchmark label, query, mask, or metric;
they only provide a scene-internal feature-level validation gate for v2 render
fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POLICY = "deterministic_even_interior_frame_manifest_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_frame_ids(frame_ids: list[int], views: int) -> list[int]:
    """Select evenly spaced interior IDs while retaining at least one train view."""

    ordered = sorted({int(frame_id) for frame_id in frame_ids})
    if len(ordered) < 2:
        raise ValueError("at least two registered feature frames are required")
    if views <= 0:
        raise ValueError("views must be positive")
    count = min(int(views), len(ordered) - 1)
    # Integer positions avoid floating-point/platform differences.  For the
    # usual 24-view ScanNet input and four dev views this yields 4,9,14,19,
    # deliberately retaining temporal endpoints for field fitting.
    positions = [((index + 1) * len(ordered)) // (count + 1) for index in range(count)]
    selected = [ordered[min(len(ordered) - 1, position)] for position in positions]
    return sorted(set(selected))


def build(feature_dir: Path, output: Path, views: int) -> dict:
    manifest_path = feature_dir / "frame_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing RADIO frame manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_ids = [int(row["frame_idx"]) for row in manifest.get("frames", [])]
    selected = select_frame_ids(frame_ids, views)
    payload = {
        "schema_version": 1,
        "policy": POLICY,
        "feature_dir": str(feature_dir.resolve()),
        "frame_manifest": str(manifest_path.resolve()),
        "frame_manifest_sha256": _sha256(manifest_path),
        "available_frame_ids": sorted({int(value) for value in frame_ids}),
        "validation_frame_ids": selected,
        "requested_validation_views": int(views),
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "query_opened": False,
        "selection_uses_only": "registered_feature_frame_ids",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--print-csv", action="store_true")
    args = parser.parse_args()
    payload = build(Path(args.feature_dir), Path(args.output), int(args.views))
    if args.print_csv:
        print(",".join(str(value) for value in payload["validation_frame_ids"]))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
