#!/usr/bin/env python3
"""Audit legacy-120 versus all-available LERF source-view domains on CPU."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.querying.all_available_source_views import audit_source_view_domain
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


SCHEMA = "radio_gs.lerf_all_available_source_view_domain_audit.v1"


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def audit_scene(entry: Mapping[str, Any]) -> dict[str, Any]:
    if set(entry) != {"scene_id", "feature_manifest", "legacy_responsibility"}:
        raise ValueError("source-view audit scene entry differs")
    scene_id = str(entry["scene_id"])
    feature_record = _record(entry["feature_manifest"], label=f"{scene_id} features")
    legacy_record = _record(
        entry["legacy_responsibility"], label=f"{scene_id} legacy responsibility"
    )
    feature, _, _ = load_json_object(
        feature_record["path"],
        expected_sha256=feature_record["sha256"],
        label=f"{scene_id} feature manifest",
    )
    responsibility, _, _ = load_json_object(
        legacy_record["path"],
        expected_sha256=legacy_record["sha256"],
        label=f"{scene_id} legacy responsibility",
    )
    feature_rows = feature.get("frames") if isinstance(feature, Mapping) else None
    feature_frames = (
        [int(row["frame_idx"]) for row in feature_rows]
        if isinstance(feature_rows, list)
        and all(isinstance(row, Mapping) and "frame_idx" in row for row in feature_rows)
        else None
    )
    metadata = responsibility.get("metadata") if isinstance(responsibility, Mapping) else None
    legacy_frames = responsibility.get("frame_indices") if isinstance(responsibility, Mapping) else None
    excluded = metadata.get("excluded_frame_ids") if isinstance(metadata, Mapping) else None
    if (
        responsibility.get("schema") != SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
        or responsibility.get("schema_version") != 1
        or responsibility.get("formula_sha256") != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or not isinstance(feature_frames, list)
        or not isinstance(legacy_frames, list)
        or not isinstance(excluded, list)
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError(f"{scene_id} source-view audit lineage differs")
    audit = audit_source_view_domain(
        feature_frame_ids=feature_frames,
        excluded_frame_ids=excluded,
        legacy_frame_ids=legacy_frames,
    )
    return {
        "scene_id": scene_id,
        "feature_manifest": feature_record,
        "legacy_responsibility": legacy_record,
        "feature_frame_count": len(audit.feature_frames),
        "excluded_frame_count": len(audit.excluded_frames),
        "all_available_frame_count": len(audit.all_available_frames),
        "legacy_frame_count": len(audit.legacy_frames),
        "omitted_frame_count": len(audit.omitted_frames),
        "legacy_coverage_fraction": audit.legacy_coverage_fraction,
        "legacy_is_all_available": audit.legacy_is_all_available,
        "legacy_is_prefix": audit.legacy_is_prefix,
        "all_available_frame_ids": list(audit.all_available_frames),
        "legacy_frame_ids": list(audit.legacy_frames),
        "omitted_frame_ids": list(audit.omitted_frames),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_record = {
        "path": str(Path(args.input_manifest).expanduser().resolve()),
        "sha256": args.input_manifest_sha256,
    }
    validate_file_record(manifest_record, label="source-view audit input manifest")
    manifest, _, _ = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="source-view audit input manifest",
    )
    scenes = manifest.get("scenes") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(scenes, list)
        or not scenes
        or [row.get("scene_id") for row in scenes]
        != sorted(row.get("scene_id") for row in scenes)
    ):
        raise ValueError("source-view audit manifest scenes differ")
    results = [audit_scene(row) for row in scenes]
    output = Path(args.output).expanduser().resolve()
    if str(output) != args.output or output.exists() or output.is_symlink():
        raise ValueError("source-view audit output must be new canonical absolute")
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_cpu_source_only_view_domain_audit",
        "implementation": file_record(Path(__file__).resolve()),
        "input_manifest": manifest_record,
        "scenes": results,
        "summary": {
            "scene_count": len(results),
            "scenes_with_omitted_available_views": sum(
                int(row["omitted_frame_count"] > 0) for row in results
            ),
            "total_omitted_available_views": sum(
                int(row["omitted_frame_count"]) for row in results
            ),
            "simple_prefix_truncation_detected": any(
                bool(row["legacy_is_prefix"]) and row["omitted_frame_count"] > 0
                for row in results
            ),
            "uniform_or_other_subset_cap_detected": any(
                not bool(row["legacy_is_prefix"]) and row["omitted_frame_count"] > 0
                for row in results
            ),
        },
        "access_audit": {
            "feature_manifests_opened": True,
            "legacy_responsibility_manifests_opened": True,
            "feature_tensors_opened": False,
            "images_opened": False,
            "queries_opened": False,
            "labels_or_masks_opened": False,
            "target_metrics_opened": False,
            "gpu_used": False,
        },
        "metric_execution_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frozen_json(output, payload)
    return {**payload, "output": file_record(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "audit_scene", "materialize"]
