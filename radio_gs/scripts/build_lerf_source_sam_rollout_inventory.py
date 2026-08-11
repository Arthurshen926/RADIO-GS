"""Freeze legal LERF mapping views for build-time official-SAM supervision.

The output aliases immutable source RGB files to their numeric MPR frame IDs.
This is an execution bridge only: image bytes are not copied or modified, and
the manifest SHA plus every source-image SHA are checked before a symlink is
created.  Evaluation frames excluded by the frozen exact-marginal authority
can therefore never enter the official-SAM teacher by a directory glob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


SCHEMA = "radio_gs.lerf_source_sam_rollout_inventory.v1"


def _output(path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"rollout output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(path.resolve()), "no_clobber": True}


def _checked_record(path: str | Path, expected_sha256: str, *, label: str) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs: {source}")
    return {"path": str(source), "sha256": str(expected_sha256)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    frame_manifest, frame_manifest_sha, frame_manifest_source = load_json_object(
        args.frame_manifest,
        expected_sha256=args.expected_frame_manifest_sha256,
        label="LERF RADIO frame manifest",
    )
    responsibility, responsibility_sha, responsibility_source = load_json_object(
        args.responsibility_authority,
        expected_sha256=args.expected_responsibility_authority_sha256,
        label="LERF exact-marginal responsibility authority",
    )
    if (
        responsibility.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or int(responsibility.get("schema_version", -1)) != 1
    ):
        raise ValueError("LERF responsibility authority schema differs")
    metadata = dict(responsibility.get("metadata", {}))
    frame_ids = [int(value) for value in responsibility.get("frame_indices", [])]
    excluded = [int(value) for value in metadata.get("excluded_frame_ids", [])]
    if (
        not frame_ids
        or frame_ids != [int(value) for value in metadata.get("selected_frame_indices", [])]
        or len(frame_ids) != len(set(frame_ids))
        or set(frame_ids).intersection(excluded)
        or any(bool(metadata.get(key, False)) for key in (
            "benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"
        ))
    ):
        raise ValueError("LERF responsibility source/evaluation split differs")
    if (
        str(frame_manifest_source) != str(Path(metadata.get("feature_frame_manifest", args.frame_manifest)).resolve())
        and metadata.get("feature_frame_manifest")
    ):
        raise ValueError("responsibility and RADIO frame manifest paths differ")
    declared_manifest_sha = str(metadata.get("feature_frame_manifest_sha256", ""))
    if declared_manifest_sha and declared_manifest_sha != frame_manifest_sha:
        raise ValueError("responsibility and RADIO frame manifest hashes differ")

    records = frame_manifest.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("LERF RADIO frame manifest has no frames")
    by_id: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("LERF RADIO frame record differs")
        frame_id = int(record.get("frame_idx", -1))
        if frame_id in by_id:
            raise ValueError("LERF RADIO frame IDs are not unique")
        by_id[frame_id] = record
    if any(frame_id not in by_id for frame_id in frame_ids):
        raise ValueError("exact-marginal frame is absent from RADIO source manifest")

    execution = dict(frame_manifest.get("execution", {}))
    source_root = Path(
        args.source_rgb_root or execution.get("resolved_source_image_dir", "")
    ).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"LERF mapping source RGB root is absent: {source_root}")
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    alias_root = rollout_root / "mapping_source_rgb_numeric"
    if alias_root.exists():
        raise FileExistsError(f"mapping RGB alias root already exists: {alias_root}")
    alias_root.mkdir(parents=True)
    source_rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        record = by_id[frame_id]
        original = (source_root / str(record.get("source_file", ""))).resolve()
        expected = str(record.get("source_sha256", ""))
        if not original.is_file() or sha256_file(original) != expected:
            raise ValueError(f"LERF source RGB SHA-256 differs for frame {frame_id}")
        suffix = original.suffix.lower()
        alias = alias_root / f"{frame_id}{suffix}"
        alias.symlink_to(original)
        if alias.resolve() != original:
            raise AssertionError("LERF source RGB alias differs")
        source_rows.append({
            "frame_id": frame_id,
            "original_path": str(original),
            "alias_path": str(alias),
            "sha256": expected,
        })

    source_list = rollout_root / "source_frames.txt"
    source_list.write_text(
        "".join(f"{row['frame_id']}{Path(row['alias_path']).suffix}\n" for row in source_rows),
        encoding="utf-8",
    )
    relation = rollout_root / "scale_ordered_relation_raster_adjoint_fp32.pt"
    mask_root = rollout_root / "official_sam3_masks"
    return {
        "schema": SCHEMA,
        "status": "frozen_before_source_sam_generation",
        "scene_id": str(args.scene_id),
        "source_frame_count": len(frame_ids),
        "source_frame_ids": frame_ids,
        "excluded_evaluation_frame_ids": excluded,
        "source_frames": source_rows,
        "source_frame_list": file_record(source_list),
        "source_rgb_alias_root": str(alias_root),
        "authorities": {
            "radio_frame_manifest": {
                "path": str(frame_manifest_source), "sha256": frame_manifest_sha,
            },
            "responsibility": {
                "path": str(responsibility_source), "sha256": responsibility_sha,
            },
            "control_field": _checked_record(
                args.control_field, args.expected_control_field_sha256, label="control field"
            ),
            "canonical_radio_cache": _checked_record(
                args.canonical_radio_cache,
                args.expected_canonical_radio_cache_sha256,
                label="canonical RADIO cache",
            ),
            "dino_capability_target": _checked_record(
                args.dino_capability_target,
                args.expected_dino_capability_target_sha256,
                label="DINO capability target",
            ),
            "sam_adaptor_capability_target": _checked_record(
                args.sam_adaptor_capability_target,
                args.expected_sam_adaptor_capability_target_sha256,
                label="SAM adaptor capability target",
            ),
            "support_graph": _checked_record(
                args.support_graph, args.expected_support_graph_sha256, label="support graph"
            ),
            "adjoint_config": _checked_record(
                args.adjoint_config, args.expected_adjoint_config_sha256, label="adjoint config"
            ),
            "geometry_checkpoint": _checked_record(
                args.geometry_checkpoint,
                args.expected_geometry_checkpoint_sha256,
                label="geometry checkpoint",
            ),
        },
        "shared_method": {
            "persistent_semantic_feature": "canonical_radio_only",
            "mapping_supervision": "official_sam3_scale_matched_relative_no_harm_v2",
            "membership_lifting": "true_alpha_compositing_adjoint",
            "official_sam_parameters": {
                "resolution": 1008, "grid_size": 12, "minimum_quality": 0.70,
                "minimum_area_fraction": 0.001, "maximum_area_fraction": 0.80,
                "nms_iou": 0.85, "duplicate_minimum_area_ratio": 0.90,
                "maximum_masks": 0,
            },
            "relation_parameters": {
                "inside_threshold": 0.80, "outside_threshold": 0.20,
                "minimum_primitives_per_mask": 3, "minimum_stability": 0.0,
                "scale_minimum_radius_m": 0.05, "scale_maximum_radius_m": 4.0,
                "scale_bins": 8, "vote_storage_dtype": "float32",
            },
            "per_scene_or_per_task_tuning": False,
        },
        "outputs": {
            "official_sam_mask_root": _output(mask_root / "manifest.json"),
            "relation_cache": _output(relation),
            "official_sam_inventory": _output(rollout_root / "official_sam_inventory.json"),
            "official_sam_build_authority": _output(rollout_root / "official_sam_build_authority.json"),
            "base_structure_manifest": _output(rollout_root / "source_only_sam_structure_manifest.json"),
            "relative_structure_manifest": _output(rollout_root / "source_only_sam_relative_manifest.json"),
            "candidate_field": _output(rollout_root / "canonical_radio_source_sam_relative_e5_seed0.pth"),
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "evaluation_frames_excluded": True,
            "benchmark_query_gt_metric_opened": False,
            "query_time_source_or_target_rgb_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--expected-frame-manifest-sha256", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--expected-responsibility-authority-sha256", required=True)
    parser.add_argument("--source-rgb-root", default="")
    parser.add_argument("--rollout-root", required=True)
    for name in (
        "control-field", "canonical-radio-cache", "dino-capability-target",
        "sam-adaptor-capability-target", "support-graph", "adjoint-config",
        "geometry-checkpoint",
    ):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args)
    write_frozen_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"], "scene_id": payload["scene_id"],
        "source_frames": payload["source_frame_count"],
        "excluded_evaluation_frames": payload["excluded_evaluation_frame_ids"],
        "output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
