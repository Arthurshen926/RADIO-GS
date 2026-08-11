"""Seal LERF mapping-time official-SAM masks and emit the generic inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.scripts.build_lerf_source_sam_rollout_inventory import SCHEMA
from radio_gs.scripts.build_source_official_sam_authority import INVENTORY_SCHEMA
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


SEAL_SCHEMA = "radio_gs.lerf_source_official_sam_mask_generation_seal.v1"


def build(
    rollout: Mapping[str, Any],
    *,
    rollout_path: Path,
    rollout_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if rollout.get("schema") != SCHEMA:
        raise ValueError("LERF source-SAM rollout schema differs")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("official SAM checkpoint SHA-256 differs")
    frame_ids = [int(value) for value in rollout.get("source_frame_ids", [])]
    source_rows = rollout.get("source_frames")
    if not frame_ids or not isinstance(source_rows, list) or len(source_rows) != len(frame_ids):
        raise ValueError("LERF source-SAM frame axis differs")
    mask_manifest_path = Path(
        rollout["outputs"]["official_sam_mask_root"]["path"]
    ).resolve()
    mask_root = mask_manifest_path.parent
    generation, generation_sha, generation_source = load_json_object(
        mask_manifest_path, label="LERF official-SAM generation manifest"
    )
    reports = generation.get("images")
    if not isinstance(reports, list) or len(reports) != len(frame_ids):
        raise ValueError("LERF official-SAM generation is incomplete")
    by_stem = {Path(str(value.get("image", ""))).stem: value for value in reports}
    if set(by_stem) != {str(value) for value in frame_ids}:
        raise ValueError("LERF official-SAM generation frame IDs differ")
    expected_files = {f"{value}.pt" for value in frame_ids}
    actual_files = {path.name for path in mask_root.glob("*.pt") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("LERF official-SAM cache set differs")
    params = dict(rollout["shared_method"]["official_sam_parameters"])
    caches: list[dict[str, str]] = []
    counts: list[int] = []
    for frame_id, source_row in zip(frame_ids, source_rows):
        alias = Path(str(source_row["alias_path"])).resolve(strict=False)
        # Resolve the symlink target separately, while keeping the numeric
        # alias path as the decoder's immutable input identity.
        alias_lexical = Path(str(source_row["alias_path"])).absolute()
        original = Path(str(source_row["original_path"])).resolve()
        if (
            alias_lexical.stem != str(frame_id)
            or not alias_lexical.is_symlink()
            or alias != original
            or not original.is_file()
        ):
            raise ValueError(f"LERF numeric source alias differs for frame {frame_id}")
        if sha256_file(original) != str(source_row["sha256"]):
            raise ValueError(f"LERF numeric source alias SHA differs for frame {frame_id}")
        report = by_stem[str(frame_id)]
        cache_path = mask_root / f"{frame_id}.pt"
        if (
            Path(str(report.get("image", ""))).absolute() != alias_lexical
            or Path(str(report.get("output", ""))).resolve() != cache_path.resolve()
        ):
            raise ValueError(f"LERF official-SAM report paths differ for frame {frame_id}")
        cache, digest, source = load_torch_mapping(
            cache_path, map_location="cpu", label=f"LERF official-SAM frame {frame_id}"
        )
        metadata = cache.get("metadata")
        score = torch.as_tensor(cache.get("scores"))
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != 2
            or metadata.get("source")
            != "official_sam3_interactive_grid_multimask_hierarchy"
            or metadata.get("official_decoder") is not True
            or metadata.get("query_free") is not True
            or Path(str(metadata.get("image", ""))).resolve() != alias
            or metadata.get("checkpoint_sha256") != checkpoint_sha256
            or int(metadata.get("grid_size", -1)) != int(params["grid_size"])
            or float(metadata.get("minimum_quality", -1)) != float(params["minimum_quality"])
            or float(metadata.get("minimum_area_fraction", -1))
            != float(params["minimum_area_fraction"])
            or float(metadata.get("maximum_area_fraction", -1))
            != float(params["maximum_area_fraction"])
            or float(metadata.get("nms_iou", -1)) != float(params["nms_iou"])
            or float(metadata.get("duplicate_minimum_area_ratio", -1))
            != float(params["duplicate_minimum_area_ratio"])
        ):
            raise ValueError(f"LERF official-SAM cache metadata differs for frame {frame_id}")
        count = int(score.numel())
        if int(report.get("masks", -1)) != count:
            raise ValueError(f"LERF official-SAM cache count differs for frame {frame_id}")
        caches.append({"path": str(source), "sha256": digest})
        counts.append(count)
    generic_inventory = {
        "schema": INVENTORY_SCHEMA,
        "scene_id": str(rollout["scene_id"]),
        "source_frames": [str(value) for value in frame_ids],
        "source_split_authority": {
            "path": str(rollout_path), "sha256": rollout_sha256,
        },
        "target_or_evaluation_frames_excluded": True,
        "benchmark_queries_excluded": True,
        "mask_caches": caches,
    }
    generic_path = Path(rollout["outputs"]["official_sam_inventory"]["path"])
    generic_record = (
        file_record(generic_path)
        if generic_path.is_file()
        else {"path": str(generic_path.resolve()), "sha256": ""}
    )
    seal = {
        "schema": SEAL_SCHEMA,
        "status": "sealed_complete_source_only_official_sam_generation",
        "scene_id": str(rollout["scene_id"]),
        "rollout_inventory": {"path": str(rollout_path), "sha256": rollout_sha256},
        "source_frame_count": len(frame_ids),
        "source_frame_ids": frame_ids,
        "excluded_evaluation_frame_ids": rollout["excluded_evaluation_frame_ids"],
        "official_sam_checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha256,
        },
        "frozen_parameters": params,
        "generation_manifest": {
            "path": str(generation_source), "sha256": generation_sha,
        },
        "generic_inventory": generic_record,
        "summary": {
            "total_masks": sum(counts),
            "empty_frame_ids": [frame for frame, count in zip(frame_ids, counts) if count == 0],
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "query_time_source_or_target_rgb_opened": False,
            "benchmark_query_gt_or_metric_opened": False,
        },
    }
    return generic_inventory, seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-inventory", required=True)
    parser.add_argument("--expected-rollout-inventory-sha256", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, help="Generation seal JSON.")
    args = parser.parse_args()
    rollout, digest, source = load_json_object(
        args.rollout_inventory,
        expected_sha256=args.expected_rollout_inventory_sha256,
        label="LERF source-SAM rollout inventory",
    )
    generic_path = Path(rollout["outputs"]["official_sam_inventory"]["path"])
    if generic_path.exists():
        raise FileExistsError(f"official-SAM generic inventory already exists: {generic_path}")
    generic_inventory, _ = build(
        rollout,
        rollout_path=source,
        rollout_sha256=digest,
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    # The seal contains the generic inventory hash, so persist the inventory
    # first and construct the final seal only after it is immutable.
    write_frozen_json(generic_path, generic_inventory)
    _, seal = build(
        rollout,
        rollout_path=source,
        rollout_sha256=digest,
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    write_frozen_json(args.output, seal)
    print(json.dumps({
        "status": seal["status"], "scene_id": seal["scene_id"],
        "source_frames": seal["source_frame_count"],
        "total_masks": seal["summary"]["total_masks"],
        "generic_inventory": str(generic_path.resolve()),
        "output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
