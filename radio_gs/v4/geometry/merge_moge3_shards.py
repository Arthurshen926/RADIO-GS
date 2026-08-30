"""Validate and merge exact MoGe-3 shard manifests into one scene receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.v4.contracts.geometry_receipt import sha256_file


def merge(input_dir: Path, output: Path) -> dict[str, object]:
    shard_paths = sorted(input_dir.glob("shard_*_of_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard manifests under {input_dir}")
    shards = [json.loads(path.read_text()) for path in shard_paths]
    first = shards[0]
    invariant_keys = (
        "scene_label", "scene_root", "source_authority", "source_authority_sha256",
        "model_family", "model_id", "model_revision", "model_checkpoint_sha256",
        "fov_x_degrees", "resolution_level", "refine_steps", "num_shards",
        "total_authority_frames",
    )
    for shard in shards:
        if any(shard[key] != first[key] for key in invariant_keys):
            raise ValueError("MoGe-3 shard manifests disagree on a scene invariant")
        if shard["target_rgb_opened"] or shard["benchmark_images_opened"] or shard["benchmark_masks_opened"]:
            raise ValueError("MoGe-3 shard opened a forbidden benchmark/target input")
    expected_shards = set(range(int(first["num_shards"])))
    actual_shards = {int(shard["shard_id"]) for shard in shards}
    if actual_shards != expected_shards:
        raise ValueError(f"incomplete shard set: expected {expected_shards}, got {actual_shards}")
    records = sorted(
        [record for shard in shards for record in shard["records"]],
        key=lambda record: int(record["frame_index"]),
    )
    frame_indices = [int(record["frame_index"]) for record in records]
    if len(frame_indices) != len(set(frame_indices)) or len(frame_indices) != int(first["total_authority_frames"]):
        raise ValueError("merged MoGe-3 records are duplicated or incomplete")
    for record in records:
        if sha256_file(record["source_image"]) != record["source_image_sha256"]:
            raise ValueError(f"source image digest changed for frame {record['frame_index']}")
        if sha256_file(record["prediction"]) != record["prediction_sha256"]:
            raise ValueError(f"prediction digest changed for frame {record['frame_index']}")
    report = {
        "schema": "radio_gs.surface_object_memory_v4.moge3_scene_manifest.v1",
        **{key: first[key] for key in invariant_keys},
        "source_rgb_opened": True,
        "target_rgb_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "records": records,
        "shard_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in shard_paths
        ],
        "valid_fraction": {
            "minimum": min(float(record["valid_fraction"]) for record in records),
            "mean": sum(float(record["valid_fraction"]) for record in records) / len(records),
        },
        "median_depth_range": [
            min(float(record["median_depth"]) for record in records),
            max(float(record["median_depth"]) for record in records),
        ],
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = merge(args.input_dir.resolve(strict=True), args.output.resolve())
    print(json.dumps({key: report[key] for key in ("scene_label", "total_authority_frames", "valid_fraction", "median_depth_range")}, indent=2))


if __name__ == "__main__":
    main()
