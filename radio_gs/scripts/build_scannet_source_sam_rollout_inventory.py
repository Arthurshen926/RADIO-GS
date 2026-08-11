"""Freeze the legal mapping inputs for the ScanNet paper8 source-SAM rollout.

The builder intentionally starts from the immutable Gaussian semantic score
receipts and the adjacent frozen MPR responsibility sidecars.  It never lists
an RGB directory: every legal source image is addressed by the explicit frame
axis in the responsibility artifact, while held-out frames are rejected.
Benchmark labels, queries, predictions, and metrics are neither needed nor
opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCHEMA = "radio_gs.scannet_source_sam_paper8_rollout_inventory.v1"
DEFAULT_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} record differs")
    record = {"path": str(value.get("path", "")), "sha256": str(value.get("sha256", ""))}
    validate_file_record(record, label=label)
    return record


def _safe_output_record(path: Path) -> dict[str, str]:
    if path.exists():
        raise FileExistsError(f"rollout output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(path.resolve()), "no_clobber": True}


def _write_source_frame_list(path: Path, frame_ids: list[int]) -> dict[str, str]:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        wanted = "".join(f"{value}.jpg\n" for value in frame_ids)
        if existing != wanted:
            raise FileExistsError(f"source frame list already exists with different content: {path}")
        return file_record(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}.jpg\n" for value in frame_ids), encoding="utf-8")
    return file_record(path)


def build_scene_record(
    *,
    scene_id: str,
    receipt_path: Path,
    dataset_root: Path,
    rollout_root: Path,
) -> dict[str, Any]:
    receipt, receipt_sha, receipt_source = load_json_object(
        receipt_path, label=f"{scene_id} frozen score-cache receipt"
    )
    if (
        receipt.get("scene_id") != scene_id
        or receipt.get("status") != "complete_immutable_gaussian_semantic_score_cache"
        or receipt.get("method_family") != "canonical_mpr_v3"
    ):
        raise ValueError(f"{scene_id} score-cache receipt contract differs")
    field = _record(receipt.get("canonical_field_source"), label=f"{scene_id} control field")
    graph = _record(receipt.get("support_graph_source"), label=f"{scene_id} support graph")
    mpr = _record(receipt.get("mpr_source"), label=f"{scene_id} raw RADIO MPR cache")
    geometry = _record(receipt.get("geometry_checkpoint"), label=f"{scene_id} geometry checkpoint")
    field_path = Path(field["path"])
    responsibility_path = field_path.parent / "responsibility_heldout4.pt"
    dino_path = field_path.parent / "dino_v3_heldout4.pt"
    sam_adaptor_path = field_path.parent / "sam3_heldout4.pt"
    for path, label in (
        (responsibility_path, "responsibility"),
        (dino_path, "DINO capability target"),
        (sam_adaptor_path, "SAM adaptor capability target"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{scene_id} lacks {label}: {path}")
    responsibility, responsibility_sha, responsibility_source = load_torch_mapping(
        responsibility_path,
        map_location="cpu",
        label=f"{scene_id} frozen responsibility cache",
    )
    metadata = responsibility.get("metadata")
    assignments = responsibility.get("assignments")
    if not isinstance(metadata, Mapping) or not isinstance(assignments, list):
        raise ValueError(f"{scene_id} responsibility structure differs")
    frame_ids = metadata.get("selected_frame_indices")
    excluded = metadata.get("excluded_frame_ids")
    if (
        responsibility.get("schema_version") != 1
        or metadata.get("assignment_mode") != "raster_gaussian_top1"
        or metadata.get("registration_weight_mode") != "alpha_depth"
        or not isinstance(frame_ids, list)
        or not frame_ids
        or not isinstance(excluded, list)
        or len(frame_ids) != len(assignments)
    ):
        raise ValueError(f"{scene_id} responsibility contract differs")
    frame_ids = [int(value) for value in frame_ids]
    excluded = [int(value) for value in excluded]
    if (
        len(frame_ids) != len(set(frame_ids))
        or set(frame_ids).intersection(excluded)
        or any(bool(metadata.get(key, False)) for key in (
            "benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"
        ))
    ):
        raise ValueError(f"{scene_id} source/held-out split is not query free")
    config_path = Path(str(metadata.get("config", ""))).expanduser().resolve()
    checkpoint_path = Path(str(metadata.get("checkpoint", ""))).expanduser().resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"{scene_id} frozen raster-adjoint context is incomplete")
    if checkpoint_path != Path(geometry["path"]).resolve():
        raise ValueError(f"{scene_id} responsibility/receipt geometry checkpoint differs")
    color_root = dataset_root / scene_id / "color"
    if not color_root.is_dir():
        raise FileNotFoundError(f"{scene_id} source color root is absent: {color_root}")
    image_paths = [color_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{scene_id} declared source RGB is absent: {missing[:4]}")
    if any((color_root / f"{frame_id}.jpg") in image_paths for frame_id in excluded):
        raise AssertionError("held-out source exclusion failed")

    scene_root = rollout_root / scene_id
    frame_list = _write_source_frame_list(scene_root / "source_frames.txt", frame_ids)
    mask_root = scene_root / "official_sam3_masks"
    relation_path = scene_root / "scale_ordered_relation_raster_adjoint_fp32.pt"
    inventory_path = scene_root / "official_sam_inventory.json"
    authority_path = scene_root / "official_sam_build_authority.json"
    base_manifest_path = scene_root / "source_only_sam_structure_manifest.json"
    relative_manifest_path = scene_root / "source_only_sam_relative_manifest.json"
    candidate_path = scene_root / "canonical_radio_source_sam_relative_e5_seed0.pth"
    for path in (
        mask_root / "manifest.json",
        relation_path,
        inventory_path,
        authority_path,
        base_manifest_path,
        relative_manifest_path,
        candidate_path,
    ):
        if path.exists():
            raise FileExistsError(f"{scene_id} rollout target already exists: {path}")
    return {
        "scene_id": scene_id,
        "source_frame_count": len(frame_ids),
        "source_frame_ids": frame_ids,
        "excluded_frame_ids": excluded,
        "source_frame_list": frame_list,
        "source_rgb_root": str(color_root.resolve()),
        "source_rgb_paths": [str(path.resolve()) for path in image_paths],
        "control_field": field,
        "raw_radio_mpr_cache": mpr,
        "dino_capability_target": file_record(dino_path),
        "sam_adaptor_capability_target": file_record(sam_adaptor_path),
        "support_graph": graph,
        "responsibility_cache": {
            "path": str(responsibility_source), "sha256": responsibility_sha,
        },
        "raster_adjoint_config": file_record(config_path),
        "geometry_checkpoint": geometry,
        "frozen_score_cache_receipt": {
            "path": str(receipt_source), "sha256": receipt_sha,
        },
        "outputs": {
            "official_sam_mask_root": {"path": str(mask_root.resolve()), "no_clobber": True},
            "relation_cache": _safe_output_record(relation_path),
            "official_sam_inventory": _safe_output_record(inventory_path),
            "official_sam_build_authority": _safe_output_record(authority_path),
            "base_structure_manifest": _safe_output_record(base_manifest_path),
            "relative_structure_manifest": _safe_output_record(relative_manifest_path),
            "candidate_field": _safe_output_record(candidate_path),
        },
        "access_audit": {
            "mapping_source_rgb_declared": True,
            "held_out_rgb_excluded": True,
            "benchmark_query_opened": False,
            "benchmark_labels_or_masks_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    receipt_root = Path(args.receipt_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    scenes = tuple(value for value in str(args.scenes).replace(",", " ").split() if value)
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("paper8 scene axis must be non-empty and unique")
    records = [
        build_scene_record(
            scene_id=scene,
            receipt_path=receipt_root / f"{scene}.pt.receipt.json",
            dataset_root=dataset_root,
            rollout_root=rollout_root,
        )
        for scene in scenes
    ]
    return {
        "schema": SCHEMA,
        "status": "frozen_before_source_sam_generation",
        "scenes": records,
        "shared_method": {
            "persistent_semantic_feature": "canonical_radio_only",
            "mapping_supervision": "official_sam3_scale_matched_relative_no_harm_v2",
            "membership_lifting": "true_alpha_compositing_adjoint",
            "official_sam_parameters": {
                "resolution": 1008,
                "grid_size": 12,
                "minimum_quality": 0.70,
                "minimum_area_fraction": 0.001,
                "maximum_area_fraction": 0.80,
                "nms_iou": 0.85,
                "duplicate_minimum_area_ratio": 0.90,
                "maximum_masks": 0,
            },
            "relation_parameters": {
                "inside_threshold": 0.80,
                "outside_threshold": 0.20,
                "minimum_primitives_per_mask": 3,
                "minimum_stability": 0.0,
                "scale_minimum_radius_m": 0.05,
                "scale_maximum_radius_m": 4.0,
                "scale_bins": 8,
                "vote_storage_dtype": "float32",
            },
            "per_scene_or_per_task_tuning": False,
        },
        "promotion_gate": {
            "all_scene_source_gates_must_pass": True,
            "all_candidate_predictions_sealed_before_benchmark_labels_open": True,
            "paper8_exact_no_rgb_at_query_time": True,
            "six_task_mainline_unchanged_until_joint_no_regression": True,
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "query_time_source_rgb_opened": False,
            "query_time_target_rgb_opened": False,
            "benchmark_query_opened": False,
            "benchmark_labels_or_masks_opened": False,
            "benchmark_predictions_or_metrics_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--scenes", default=" ".join(DEFAULT_SCENES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args)
    write_frozen_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "scenes": {item["scene_id"]: item["source_frame_count"] for item in payload["scenes"]},
        "output": str(Path(args.output).resolve()),
        "sha256": sha256_file(args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
