"""Seal official-SAM source-mask caches as training-only field supervision.

The caller supplies an explicit JSON inventory instead of a directory glob.
This prevents a target/evaluation frame from being included merely because it
shares a folder with legal mapping frames.  The inventory is expected to be
created from the frozen dataset split and contains only file records plus the
two target-exclusion booleans checked below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.training.source_only_sam_structure import (
    OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


INVENTORY_SCHEMA = "radio_gs.source_rgb_official_sam_inventory.v1"


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def validate_inventory(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("official-SAM source inventory must be a mapping")
    payload = dict(value)
    if (
        set(payload)
        != {
            "schema",
            "scene_id",
            "source_frames",
            "source_split_authority",
            "target_or_evaluation_frames_excluded",
            "benchmark_queries_excluded",
            "mask_caches",
        }
        or payload.get("schema") != INVENTORY_SCHEMA
        or not str(payload.get("scene_id", ""))
        or payload.get("target_or_evaluation_frames_excluded") is not True
        or payload.get("benchmark_queries_excluded") is not True
    ):
        raise ValueError("official-SAM source inventory contract differs")
    frames = payload.get("source_frames")
    caches = payload.get("mask_caches")
    if (
        not isinstance(frames, list)
        or not frames
        or len(frames) != len(set(str(item) for item in frames))
        or not isinstance(caches, list)
        or len(caches) != len(frames)
    ):
        raise ValueError("official-SAM source inventory frame/cache axis differs")
    payload["source_frames"] = [str(item) for item in frames]
    payload["source_split_authority"] = _record(
        payload["source_split_authority"], label="official-SAM source split"
    )
    payload["mask_caches"] = [
        _record(item, label=f"official-SAM source mask cache {index}")
        for index, item in enumerate(caches)
    ]
    return payload


def build_authority(
    inventory: object,
    *,
    inventory_path: str | Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    value = validate_inventory(inventory)
    checkpoint_hashes: set[str] = set()
    images: list[str] = []
    for frame, record in zip(value["source_frames"], value["mask_caches"]):
        cache = torch.load(record["path"], map_location="cpu", weights_only=False)
        metadata = cache.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != 2
            or metadata.get("source")
            != "official_sam3_interactive_grid_multimask_hierarchy"
            or metadata.get("official_decoder") is not True
            or metadata.get("query_free") is not True
            # The inventory owns the immutable frame-ID mapping.  A LERF
            # numeric alias may resolve to an original ``frame_00001.jpg``;
            # bind the frame ID to the sealed cache filename while retaining
            # the decoder's resolved original image path in provenance.
            or Path(str(record["path"])).stem != str(frame)
        ):
            raise ValueError(f"official-SAM source cache differs for frame {frame}")
        checkpoint_hash = str(metadata.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64:
            raise ValueError("official-SAM source cache lacks checkpoint SHA-256")
        checkpoint_hashes.add(checkpoint_hash)
        images.append(str(metadata["image"]))
    if len(checkpoint_hashes) != 1:
        raise ValueError("official-SAM source caches use different checkpoints")
    return {
        "schema": OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA,
        "scene_id": value["scene_id"],
        "official_sam": True,
        "source_rgb_only": True,
        "query_free": True,
        "target_or_evaluation_rgb_opened": False,
        "benchmark_query_opened": False,
        "benchmark_gt_or_metric_opened": False,
        "teacher_artifacts_training_only": True,
        "official_sam_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "source_frames": value["source_frames"],
        "source_split_authority": value["source_split_authority"],
        "source_images": images,
        "source_mask_caches": value["mask_caches"],
        "inventory": {
            "path": str(Path(inventory_path).resolve()),
            "sha256": str(inventory_sha256),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inventory, digest, source = load_json_object(
        args.inventory,
        expected_sha256=args.expected_inventory_sha256,
        label="official-SAM source inventory",
    )
    authority = build_authority(
        inventory,
        inventory_path=source,
        inventory_sha256=digest,
    )
    write_frozen_json(args.output, authority)
    print(
        json.dumps(
            {
                "status": "sealed_source_official_sam_training_authority",
                "output": str(Path(args.output).resolve()),
                "sha256": sha256_file(args.output),
                "scene_id": authority["scene_id"],
                "source_frames": len(authority["source_frames"]),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
