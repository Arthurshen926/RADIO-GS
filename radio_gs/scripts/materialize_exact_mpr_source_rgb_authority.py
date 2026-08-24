#!/usr/bin/env python3
"""Seal a uniformly sampled source-RGB cohort compatible with exact MPR.

The output is a query-free, source-only authority consumed by official SAM
and other native image teachers.  Every selected frame must occur both in the
immutable exact-marginal authority and in the hash-bound extraction manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(value)


def _uniform_indices(count: int, maximum: int) -> list[int]:
    selected = min(int(count), int(maximum))
    if selected <= 0:
        raise ValueError("maximum-images must be positive")
    if selected == 1:
        return [0]
    # Integer rounding keeps the result deterministic across NumPy/PyTorch
    # versions and preserves both ends of the compatible source sequence.
    positions = [
        (index * (int(count) - 1) + (selected - 1) // 2) // (selected - 1)
        for index in range(selected)
    ]
    if len(set(positions)) != selected:
        raise RuntimeError("uniform source selection unexpectedly duplicated a frame")
    return positions


def build(args: argparse.Namespace) -> dict[str, Any]:
    exact_path = Path(args.exact_mpr_authority).expanduser().resolve(strict=True)
    manifest_path = Path(args.frame_manifest).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    exact = _load_json(exact_path, "exact-MPR authority")
    manifest = _load_json(manifest_path, "source frame manifest")
    metadata = exact.get("metadata")
    exact_frames = [int(value) for value in exact.get("frame_indices", [])]
    if (
        exact.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or exact.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or not exact_frames
        or bool(metadata.get("benchmark_images_opened", True))
        or bool(metadata.get("benchmark_masks_opened", True))
    ):
        raise ValueError("exact-MPR source-only contract differs")
    declared_image_dir = str(manifest.get("image_dir", ""))
    image_dir = Path(
        args.image_dir_override or declared_image_dir
    ).expanduser().resolve(strict=True)
    raw_frames = manifest.get("frames")
    if not image_dir.is_dir() or not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("source frame manifest has no image cohort")
    excluded_names = {str(value) for value in manifest.get("excluded_image_names", [])}
    excluded_stems = {str(value) for value in manifest.get("excluded_image_stems", [])}
    indexed: dict[int, dict[str, str]] = {}
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise ValueError("source frame record is not an object")
        frame = int(raw.get("frame_idx", -1))
        relative = str(raw.get("source_file", ""))
        declared_sha256 = str(raw.get("source_sha256", ""))
        path = (image_dir / relative).resolve()
        if (
            frame < 0
            or frame in indexed
            or not relative
            or relative in excluded_names
            or Path(relative).stem in excluded_stems
            or path.parent != image_dir
            or not path.is_file()
            or path.is_symlink()
            or len(declared_sha256) != 64
        ):
            raise ValueError(f"invalid source frame identity: {frame}")
        indexed[frame] = {"path": str(path), "sha256": declared_sha256}
    compatible = [frame for frame in exact_frames if frame in indexed]
    if not compatible:
        raise ValueError("exact-MPR/source-manifest frame intersection is empty")
    selected = [compatible[index] for index in _uniform_indices(len(compatible), args.maximum_images)]
    for frame in selected:
        if sha256_file(Path(indexed[frame]["path"])) != indexed[frame]["sha256"]:
            raise ValueError(f"source RGB SHA-256 differs for selected frame {frame}")
    scene = str(args.scene or manifest.get("scene", ""))
    if not scene:
        raise ValueError("scene identity is absent")
    payload = {
        "contract": "sam3-query-free-source-rgb-authority-v1",
        "schema_version": 1,
        "scene": scene,
        "cohort": "source-only-exact-mpr-compatible",
        "images": [
            {
                "image_id": f"frame_{frame:05d}",
                "path": indexed[frame]["path"],
                "sha256": indexed[frame]["sha256"],
                "rgb_role": "registered_source_or_mapping_view",
            }
            for frame in selected
        ],
        "information_policy": {
            "registered_source_rgb_only": True,
            "target_or_evaluation_rgb_used": False,
            "query_text_used": False,
            "benchmark_ground_truth_used": False,
        },
        "construction": {
            "contract": "source-only-exact-mpr-intersection-uniform-v1",
            "exact_mpr_authority": {
                "path": str(exact_path),
                "sha256": sha256_file(exact_path),
                "compatible_count": len(compatible),
            },
            "frame_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "source_only_count": len(indexed),
                "declared_image_dir": declared_image_dir,
                "resolved_image_dir": str(image_dir),
                "image_dir_relocated_under_per_frame_sha256": bool(
                    args.image_dir_override
                ),
            },
            "selection": {
                "type": "uniform_ordered_compatible_cohort_edge_preserving",
                "maximum_images": int(args.maximum_images),
                "selected_count": len(selected),
                "selected_frame_indices": selected,
            },
        },
    }
    write_frozen_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="")
    parser.add_argument("--exact-mpr-authority", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--maximum-images", type=int, default=8)
    parser.add_argument("--image-dir-override", default="")
    parser.add_argument("--output", required=True)
    result = build(parser.parse_args())
    print(json.dumps(result["construction"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
