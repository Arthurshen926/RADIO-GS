"""Seal one paper8 scene's mapping-time official-SAM mask generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.scripts.build_scannet_source_sam_rollout_inventory import SCHEMA
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SEAL_SCHEMA = "radio_gs.scannet_source_official_sam_mask_generation_seal.v1"


def _scene(payload: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("source-SAM rollout inventory schema differs")
    matches = [value for value in payload.get("scenes", []) if value.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError("source-SAM rollout scene does not resolve exactly once")
    return dict(matches[0])


def build(
    rollout: Mapping[str, Any],
    *,
    rollout_path: Path,
    rollout_sha256: str,
    scene_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    scene = _scene(rollout, scene_id)
    validate_file_record(scene["source_frame_list"], label="source frame list")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("official SAM checkpoint SHA-256 differs")
    frame_ids = [int(value) for value in scene["source_frame_ids"]]
    source_paths = [Path(value).resolve() for value in scene["source_rgb_paths"]]
    if len(frame_ids) != len(source_paths) or [path.stem for path in source_paths] != [str(v) for v in frame_ids]:
        raise ValueError("source RGB/frame axis differs")
    mask_root = Path(scene["outputs"]["official_sam_mask_root"]["path"]).resolve()
    manifest_path = mask_root / "manifest.json"
    manifest, manifest_sha, manifest_source = load_json_object(
        manifest_path, label=f"{scene_id} official-SAM generation manifest"
    )
    reports = manifest.get("images")
    if not isinstance(reports, list) or len(reports) != len(frame_ids):
        raise ValueError("official-SAM generation manifest is incomplete")
    by_stem = {Path(str(value.get("image", ""))).stem: value for value in reports}
    if set(by_stem) != {str(value) for value in frame_ids} or len(by_stem) != len(reports):
        raise ValueError("official-SAM generation frame axis differs")
    actual_files = {path.name for path in mask_root.iterdir() if path.is_file() and path.suffix == ".pt"}
    expected_files = {f"{value}.pt" for value in frame_ids}
    if actual_files != expected_files:
        raise ValueError("official-SAM output root contains a missing or undeclared cache")
    params = dict(rollout["shared_method"]["official_sam_parameters"])
    mask_records: list[dict[str, Any]] = []
    total_masks = 0
    empty_frames = []
    for frame_id, image_path in zip(frame_ids, source_paths):
        report = by_stem[str(frame_id)]
        cache_path = mask_root / f"{frame_id}.pt"
        if Path(str(report.get("image", ""))).resolve() != image_path or Path(str(report.get("output", ""))).resolve() != cache_path:
            raise ValueError(f"official-SAM manifest path differs for frame {frame_id}")
        cache, digest, source = load_torch_mapping(
            cache_path, map_location="cpu", label=f"official-SAM cache frame {frame_id}"
        )
        metadata = cache.get("metadata")
        scores = torch.as_tensor(cache.get("scores"))
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != 2
            or metadata.get("source") != "official_sam3_interactive_grid_multimask_hierarchy"
            or metadata.get("official_decoder") is not True
            or metadata.get("query_free") is not True
            or Path(str(metadata.get("image", ""))).resolve() != image_path
            or metadata.get("checkpoint_sha256") != checkpoint_sha256
            or int(metadata.get("grid_size", -1)) != int(params["grid_size"])
            or float(metadata.get("minimum_quality", -1)) != float(params["minimum_quality"])
            or float(metadata.get("minimum_area_fraction", -1)) != float(params["minimum_area_fraction"])
            or float(metadata.get("maximum_area_fraction", -1)) != float(params["maximum_area_fraction"])
            or float(metadata.get("nms_iou", -1)) != float(params["nms_iou"])
            or float(metadata.get("duplicate_minimum_area_ratio", -1)) != float(params["duplicate_minimum_area_ratio"])
        ):
            raise ValueError(f"official-SAM cache metadata differs for frame {frame_id}")
        count = int(scores.numel())
        if int(report.get("masks", -1)) != count:
            raise ValueError(f"official-SAM mask count differs for frame {frame_id}")
        total_masks += count
        if count == 0:
            empty_frames.append(frame_id)
        mask_records.append({"path": str(source), "sha256": digest, "masks": count})
    return {
        "schema": SEAL_SCHEMA,
        "status": "sealed_complete_source_only_official_sam_generation",
        "scene_id": scene_id,
        "rollout_inventory": {"path": str(rollout_path), "sha256": rollout_sha256},
        "source_frame_list": scene["source_frame_list"],
        "source_frame_count": len(frame_ids),
        "source_frame_ids": frame_ids,
        "excluded_frame_ids": scene["excluded_frame_ids"],
        "official_sam_checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha256,
        },
        "frozen_parameters": params,
        "execution_parameters_not_embedded_by_legacy_cache_builder": {
            "resolution": params["resolution"],
            "maximum_masks": params["maximum_masks"],
            "dtype": "bfloat16",
            "authority": "pre_generation_rollout_inventory_and_frozen_command",
        },
        "generation_manifest": {"path": str(manifest_source), "sha256": manifest_sha},
        "mask_caches": mask_records,
        "summary": {"total_masks": total_masks, "empty_frame_ids": empty_frames},
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "query_time_source_or_target_rgb_opened": False,
            "benchmark_query_gt_or_metric_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-inventory", required=True)
    parser.add_argument("--expected-rollout-inventory-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rollout, digest, source = load_json_object(
        args.rollout_inventory,
        expected_sha256=args.expected_rollout_inventory_sha256,
        label="paper8 source-SAM rollout inventory",
    )
    payload = build(
        rollout,
        rollout_path=source,
        rollout_sha256=digest,
        scene_id=args.scene_id,
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    write_frozen_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"], "scene_id": payload["scene_id"],
        "source_frames": payload["source_frame_count"],
        "total_masks": payload["summary"]["total_masks"],
        "output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
