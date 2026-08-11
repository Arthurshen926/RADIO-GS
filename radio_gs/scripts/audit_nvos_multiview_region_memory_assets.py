#!/usr/bin/env python3
"""Seal NVOS source-view assets for target-free multiview region memory.

This is a benchmark adapter and authority audit, not the region-memory method.
The reusable method lives in ``radio_gs.querying.multiview_region_memory``.
The adapter intentionally hashes only manifest-declared training/source RGBs;
evaluation RGB files are never opened or hashed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from radio_gs.querying.multiview_region_memory import method_contract
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


ARTIFACT_TYPE = "nvos_multiview_region_memory_source_inventory_v1"


def _scene(manifest: Mapping[str, Any], scene_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in manifest.get("scenes", [])
        if isinstance(value, Mapping) and value.get("scene_id") == scene_id
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one scene {scene_id!r}")
    return dict(matches[0])


def _lower_sha(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _load_yaml(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    source = path.expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scene config must contain a mapping")
    return value, file_record(source)


def _validate_assignment(
    value: object,
    *,
    view_index: int,
    num_pixels: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "gaussian_ids",
        "pixel_ids",
        "weights",
    }:
        raise ValueError(f"responsibility view {view_index} schema differs")
    gaussian_ids = torch.as_tensor(value["gaussian_ids"]).reshape(-1)
    pixel_ids = torch.as_tensor(value["pixel_ids"]).reshape(-1)
    weights = torch.as_tensor(value["weights"]).float().reshape(-1)
    if (
        gaussian_ids.shape != pixel_ids.shape
        or gaussian_ids.shape != weights.shape
        or gaussian_ids.numel() == 0
        or int(gaussian_ids.min()) < 0
        or int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= int(num_pixels)
        or not bool(torch.isfinite(weights).all())
        or bool(((weights <= 0) | (weights > 1.001)).any())
    ):
        raise ValueError(f"responsibility view {view_index} tensors are malformed")
    if pixel_ids.numel() > 1 and bool((pixel_ids[1:] < pixel_ids[:-1]).any()):
        raise ValueError(f"responsibility view {view_index} pixel order differs")
    result = {
        "assignment_rows": int(gaussian_ids.numel()),
        "visible_gaussian_rows": int(torch.unique(gaussian_ids.long()).numel()),
    }
    return result


def audit_scene(
    manifest: Mapping[str, Any],
    *,
    scene_id: str,
    responsibility_path: str | Path,
    responsibility_sha256: str,
) -> dict[str, Any]:
    scene = _scene(manifest, scene_id)
    if scene.get("target_rgb_policy") != "excluded_from_field_training_and_query":
        raise ValueError(f"{scene_id}: target RGB policy is not strict")
    prompt = scene.get("prompt")
    if (
        not isinstance(prompt, Mapping)
        or prompt.get("type") not in {"positive_negative_scribbles", "reference_binary_mask"}
    ):
        raise ValueError(f"{scene_id}: unsupported reference prompt")
    reference_frame_id = str(prompt.get("frame_id", ""))
    prompt_frames = [str(value) for value in scene.get("prompt_frame_ids", [])]
    evaluation_frames = [str(value) for value in scene.get("evaluation_frame_ids", [])]
    excluded_frames = [str(value) for value in scene.get("excluded_training_frame_ids", [])]
    if (
        prompt_frames != [reference_frame_id]
        or not evaluation_frames
        or set(evaluation_frames) != set(excluded_frames)
        or reference_frame_id in set(evaluation_frames)
    ):
        raise ValueError(f"{scene_id}: prompt/evaluation/exclusion authority differs")

    training = scene.get("training_frames")
    if not isinstance(training, list) or not training:
        raise ValueError(f"{scene_id}: training frame inventory is empty")
    training_by_name: dict[str, dict[str, Any]] = {}
    for row in training:
        if not isinstance(row, Mapping):
            raise ValueError(f"{scene_id}: malformed training row")
        frame_id = str(row.get("frame_id", ""))
        path = Path(str(row.get("rgb_path", ""))).expanduser().absolute()
        if not frame_id or path.stem != frame_id or path.name in training_by_name:
            raise ValueError(f"{scene_id}: ambiguous training RGB identity")
        training_by_name[path.name] = dict(row)
    if reference_frame_id not in {str(row["frame_id"]) for row in training_by_name.values()}:
        raise ValueError(f"{scene_id}: reference frame is not a legal source view")

    known_frames = {
        str(row.get("frame_id")): dict(row)
        for row in scene.get("frames", [])
        if isinstance(row, Mapping)
    }
    if set(evaluation_frames) - set(known_frames):
        raise ValueError(f"{scene_id}: evaluation frame lacks manifest path authority")
    target_names = {
        Path(str(known_frames[frame_id]["rgb_path"])).name
        for frame_id in evaluation_frames
    }
    if target_names.intersection(training_by_name):
        raise ValueError(f"{scene_id}: target RGB leaked into training inventory")

    expected_responsibility_sha = _lower_sha(
        responsibility_sha256,
        label=f"{scene_id} responsibility SHA256",
    )
    responsibility, observed_sha, responsibility_source = load_torch_mapping(
        responsibility_path,
        expected_sha256=expected_responsibility_sha,
        map_location="cpu",
        label=f"{scene_id} source responsibility",
    )
    metadata = responsibility.get("metadata")
    assignments = responsibility.get("assignments")
    if (
        responsibility.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or not isinstance(assignments, list)
        or metadata.get("schema_version") != 1
        or metadata.get("assignment_mode") != "raster_gaussian_top1"
        or metadata.get("registration_weight_mode") != "alpha_depth"
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or list(metadata.get("selected_dataset_indices", []))
        != list(range(len(assignments)))
        or list(metadata.get("selected_frame_indices", []))
        != list(range(len(assignments)))
    ):
        raise ValueError(f"{scene_id}: source responsibility contract differs")
    feature_height = int(metadata.get("feature_height", 0))
    feature_width = int(metadata.get("feature_width", 0))
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError(f"{scene_id}: invalid responsibility feature grid")

    config_path = Path(str(metadata.get("config", ""))).expanduser().resolve(strict=True)
    config, config_record = _load_yaml(config_path)
    frame_manifest_path = (
        Path(str(config.get("feature_dir", ""))).expanduser().resolve(strict=True)
        / "frame_manifest.json"
    )
    frame_manifest, _, frame_manifest_source = load_json_object(
        frame_manifest_path,
        label=f"{scene_id} RADIO frame manifest",
    )
    source_frames = frame_manifest.get("frames")
    if (
        frame_manifest.get("scene") != scene_id
        or frame_manifest.get("image_sort_mode") != "numeric_then_exact_filename"
        or frame_manifest.get("frame_id_mode") != "source_rank"
        or frame_manifest.get("num_frames") != len(assignments)
        or not isinstance(source_frames, list)
        or len(source_frames) != len(assignments)
        or set(frame_manifest.get("excluded_image_names", [])) != target_names
        or list(frame_manifest.get("features", {}).get("backbone", {}).get("grid", []))
        != [feature_height, feature_width]
    ):
        raise ValueError(f"{scene_id}: RADIO frame manifest contract differs")
    source_names = [str(row.get("source_file", "")) for row in source_frames]
    if len(set(source_names)) != len(source_names) or set(source_names) != set(training_by_name):
        raise ValueError(
            f"{scene_id}: assignment-index source set differs from strict training RGB set"
        )

    train_ids_path = Path(str(config.get("train_frame_ids_path", ""))).resolve(strict=True)
    train_ids, _, train_ids_source = load_json_object(
        train_ids_path,
        label=f"{scene_id} train frame ids",
    )
    if train_ids.get("frame_ids") != list(range(len(assignments))):
        raise ValueError(f"{scene_id}: train frame IDs do not bind assignment order")
    camera_map_path = Path(str(config.get("camera_map_path", ""))).resolve(strict=True)
    camera_map, _, camera_map_source = load_json_object(
        camera_map_path,
        label=f"{scene_id} camera map",
    )
    camera_records = camera_map.get("records")
    if camera_map.get("complete_colmap_coverage") is not True or not isinstance(
        camera_records, list
    ):
        raise ValueError(f"{scene_id}: camera map is incomplete")
    camera_by_name = {
        Path(str(row.get("rgb_path", ""))).name: dict(row)
        for row in camera_records
        if isinstance(row, Mapping)
    }
    if not set(source_names).union(target_names).issubset(camera_by_name):
        raise ValueError(f"{scene_id}: camera map omits source or target identity")

    source_records: list[dict[str, Any]] = []
    assignment_rows = 0
    total_source_rgb_bytes = 0
    for view_index, (frame_row, assignment) in enumerate(zip(source_frames, assignments)):
        if (
            frame_row.get("source_rank") != view_index
            or frame_row.get("frame_idx") != view_index
            or frame_row.get("saved_stem") != f"rgb_{view_index}"
        ):
            raise ValueError(f"{scene_id}: source frame order is not canonical")
        source_name = source_names[view_index]
        manifest_row = training_by_name[source_name]
        rgb_path = Path(str(manifest_row["rgb_path"])).expanduser().resolve(strict=True)
        if rgb_path.name in target_names:
            raise ValueError(f"{scene_id}: target RGB reached source hash loop")
        declared_root = Path(str(scene.get("rgb_directory", ""))).resolve(strict=True)
        try:
            rgb_path.relative_to(declared_root)
        except ValueError as error:
            raise ValueError(f"{scene_id}: source RGB escaped declared root") from error
        camera_path = Path(str(camera_by_name[source_name].get("rgb_path", ""))).resolve(
            strict=True
        )
        if camera_path != rgb_path:
            raise ValueError(f"{scene_id}: camera/source RGB path differs")
        assignment_report = _validate_assignment(
            assignment,
            view_index=view_index,
            num_pixels=feature_height * feature_width,
        )
        assignment_rows += assignment_report["assignment_rows"]
        rgb_sha = sha256_file(rgb_path)
        rgb_bytes = int(rgb_path.stat().st_size)
        total_source_rgb_bytes += rgb_bytes
        source_records.append(
            {
                "assignment_view_index": view_index,
                "frame_id": str(manifest_row["frame_id"]),
                "source_file": source_name,
                "rgb_path": str(rgb_path),
                "rgb_sha256": rgb_sha,
                "rgb_bytes": rgb_bytes,
                **assignment_report,
            }
        )

    reference_indices = [
        row["assignment_view_index"]
        for row in source_records
        if row["frame_id"] == reference_frame_id
    ]
    if len(reference_indices) != 1:
        raise ValueError(f"{scene_id}: reference source assignment is ambiguous")
    result = {
        "scene_id": scene_id,
        "prompt_kind": str(prompt["type"]),
        "reference_frame_id": reference_frame_id,
        "reference_assignment_view_index": reference_indices[0],
        "forbidden_target_frame_ids": evaluation_frames,
        "forbidden_target_rgb_names": sorted(target_names),
        "source_view_count": len(source_records),
        "feature_grid_hw": [feature_height, feature_width],
        "source_rgb_total_bytes": total_source_rgb_bytes,
        "assignment_rows_total": assignment_rows,
        "source_view_order_authority": (
            "RADIO frame_manifest numeric_then_exact order, not NVOS manifest training list order"
        ),
        "source_rgb_inventory_sha256": canonical_json_sha256(source_records),
        "source_views": source_records,
        "assets": {
            "responsibility": {
                "path": str(responsibility_source),
                "sha256": observed_sha,
                "bytes": int(responsibility_source.stat().st_size),
                "xyz_sha256": str(metadata.get("xyz_sha256", "")),
                "gaussian_state_sha256": str(metadata.get("gaussian_state_sha256", "")),
                "pose_sha256": str(metadata.get("pose_sha256", "")),
                "intrinsics_sha256": str(metadata.get("intrinsics_sha256", "")),
            },
            "scene_config": config_record,
            "radio_frame_manifest": file_record(frame_manifest_source),
            "train_frame_ids": file_record(train_ids_source),
            "camera_map": file_record(camera_map_source),
        },
        "safety": {
            "target_rgb_content_opened_or_hashed": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "source_rgb_content_hashed": True,
            "source_and_target_sets_disjoint": True,
        },
    }
    # Do not retain one scene's sparse responsibility tensors while auditing
    # the next scene on memory-constrained hosts.
    del assignments, responsibility
    gc.collect()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_sha, manifest_source = load_json_object(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        label="NVOS strict manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("benchmark") != "nvos"
        or manifest.get("protocol", {}).get("target_rgb_at_query") != "forbidden"
        or manifest.get("protocol", {}).get("target_mask_use") != "scoring_only"
    ):
        raise ValueError("NVOS strict manifest protocol differs")
    if not args.scene_binding:
        raise ValueError("at least one scene binding is required")
    scene_ids = [str(value[0]) for value in args.scene_binding]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("scene bindings must be unique")
    scenes = {
        str(scene_id): audit_scene(
            manifest,
            scene_id=str(scene_id),
            responsibility_path=path,
            responsibility_sha256=sha,
        )
        for scene_id, path, sha in args.scene_binding
    }
    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "source_rgb_and_assignment_authority_sealed_before_sam3_or_target_access",
        "manifest": {
            "path": str(manifest_source),
            "sha256": manifest_sha,
        },
        "method_contract": method_contract(),
        "scene_order": scene_ids,
        "scenes": scenes,
        "global_safety": {
            "gpu_used": False,
            "official_sam3_loaded": False,
            "target_rgb_content_opened_or_hashed": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }
    output = write_frozen_json(args.output, payload)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "source_views": {scene: value["source_view_count"] for scene, value in scenes.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--scene-binding",
        action="append",
        nargs=3,
        metavar=("SCENE_ID", "RESPONSIBILITY_PATH", "RESPONSIBILITY_SHA256"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
