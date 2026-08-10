#!/usr/bin/env python3
"""Select all-available LERF teachers using only frozen source-view LOO."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts import materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as v2
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


SCHEMA = "radio_gs.lerf_source_only_all_available_view_gate.v1"


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    path = Path(result["path"])
    if (
        str(path.expanduser().resolve()) != result["path"]
        or path.is_symlink()
        or not path.is_file()
        or len(result["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in result["sha256"])
    ):
        raise ValueError(f"{label} file record differs")
    return result


def _summary(
    record: dict[str, str], *, candidate: bool
) -> dict[str, Any]:
    payload, _, _ = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="candidate all-available teacher" if candidate else "legacy teacher",
    )
    rows = payload.get("global_rows")
    valid = payload.get("teacher_valid")
    retained = payload.get("retained_view_count")
    audit = payload.get(v2.LOO_AUDIT_FIELD)
    allowed_schema = (
        "radio_gs.lerf_source_teacher_mean_siglip_all_available.v2"
        if candidate
        else "radio_gs.lerf_source_teacher_mean_siglip.v2"
    )
    if (
        payload.get("schema") != allowed_schema
        or not torch.is_tensor(rows)
        or rows.ndim != 1
        or rows.dtype != torch.long
        or not torch.is_tensor(valid)
        or valid.shape != rows.shape
        or valid.dtype != torch.bool
        or not torch.is_tensor(retained)
        or retained.shape != rows.shape
        or retained.dtype != torch.uint8
        or not isinstance(audit, Mapping)
        or payload.get("access_audit", {}).get("target_metrics_opened") is not False
    ):
        raise ValueError("source-only teacher gate payload differs")
    v2.validate_source_only_loo_ceiling_audit(audit)
    baseline_candidate = next(
        (
            item
            for item in audit["candidates"]
            if float(item["maximum_angle_radians"]) == 0.15
        ),
        None,
    )
    if baseline_candidate is None:
        raise ValueError("source-only LOO audit lacks the frozen 0.15 baseline")
    return {
        "record": record,
        "global_rows": rows.detach().long().cpu().contiguous(),
        "teacher_valid": valid.detach().bool().cpu().contiguous(),
        "retained_view_count": retained.detach().to(torch.uint8).cpu().contiguous(),
        "loo_audit": dict(audit),
        "loo_direction_mean_cosine": float(audit["loo_direction_mean_cosine"]),
        "o1_0p15_mean_cosine": float(baseline_candidate["mean_cosine"]),
    }


def compare_scene(entry: Mapping[str, Any]) -> dict[str, Any]:
    if set(entry) != {"scene_id", "legacy_teacher", "candidate_teacher"}:
        raise ValueError("source-only scene gate entry differs")
    scene_id = str(entry["scene_id"])
    legacy = _summary(_record(entry["legacy_teacher"], label=f"{scene_id} legacy"), candidate=False)
    candidate = _summary(
        _record(entry["candidate_teacher"], label=f"{scene_id} candidate"),
        candidate=True,
    )
    if not torch.equal(legacy["global_rows"], candidate["global_rows"]):
        raise ValueError(f"{scene_id} source teacher row axis differs")
    coverage_nonregression = bool(
        (candidate["teacher_valid"] | ~legacy["teacher_valid"]).all()
    )
    retained_nonregression = bool(
        (candidate["retained_view_count"] >= legacy["retained_view_count"]).all()
    )
    loo_delta = (
        candidate["loo_direction_mean_cosine"]
        - legacy["loo_direction_mean_cosine"]
    )
    o1_delta = candidate["o1_0p15_mean_cosine"] - legacy["o1_0p15_mean_cosine"]
    passed = coverage_nonregression and retained_nonregression and loo_delta >= 0.0 and o1_delta >= 0.0
    return {
        "scene_id": scene_id,
        "legacy_teacher": legacy["record"],
        "candidate_teacher": candidate["record"],
        "rows": int(legacy["global_rows"].numel()),
        "legacy_rows_with_teacher": int(legacy["teacher_valid"].sum()),
        "candidate_rows_with_teacher": int(candidate["teacher_valid"].sum()),
        "coverage_nonregression": coverage_nonregression,
        "retained_view_count_elementwise_nonregression": retained_nonregression,
        "legacy_loo_direction_mean_cosine": legacy["loo_direction_mean_cosine"],
        "candidate_loo_direction_mean_cosine": candidate["loo_direction_mean_cosine"],
        "loo_direction_mean_cosine_delta": loo_delta,
        "legacy_o1_0p15_source_mean_cosine": legacy["o1_0p15_mean_cosine"],
        "candidate_o1_0p15_source_mean_cosine": candidate["o1_0p15_mean_cosine"],
        "o1_0p15_source_mean_cosine_delta": o1_delta,
        "source_only_scene_gate_passed": passed,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_record = {
        "path": str(Path(args.input_manifest).expanduser().resolve()),
        "sha256": str(args.input_manifest_sha256),
    }
    validate_file_record(manifest_record, label="source-only gate manifest")
    manifest, _, _ = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="source-only all-available gate manifest",
    )
    scenes = manifest.get("scenes") if isinstance(manifest, Mapping) else None
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("source-only gate manifest scenes differ")
    results = [compare_scene(entry) for entry in scenes]
    output = Path(args.output).expanduser().resolve()
    if str(output) != args.output or output.exists() or output.is_symlink():
        raise ValueError("source-only gate output must be new canonical absolute")
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": (
            "passed_source_only_all_available_view_gate"
            if all(row["source_only_scene_gate_passed"] for row in results)
            else "rejected_source_only_all_available_view_gate"
        ),
        "implementation": file_record(Path(__file__).resolve()),
        "input_manifest": manifest_record,
        "scenes": results,
        "all_scenes_passed": all(
            row["source_only_scene_gate_passed"] for row in results
        ),
        "selection_rule": {
            "coverage_nonregression": "candidate valid includes every legacy-valid row",
            "retained_count_nonregression": "candidate retained-view count is elementwise no smaller",
            "loo_direction_nonregression": "candidate query-free LOO direction mean cosine >= legacy",
            "o1_nonregression": "candidate frozen-0.15 source LOO mean cosine >= legacy",
            "aggregate_compensation_for_scene_regression": False,
            "per_scene_or_per_query_tuning": False,
        },
        "access_audit": {
            "source_teacher_payloads_opened": True,
            "source_loo_audits_opened": True,
            "target_images_opened": False,
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


__all__ = ["SCHEMA", "compare_scene", "materialize"]
