#!/usr/bin/env python3
"""Seal the exact initial-field versus Field-A figurines LERF comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from radio_gs.utils.immutable_artifacts import sha256_file


METRICS = (
    "average_precision",
    "oracle_threshold_iou",
    "positive_negative_score_margin",
    "frozen_formal_miou",
    "frozen_formal_positive_coverage",
    "frozen_formal_selected_purity",
    "target_blind_otsu3_miou",
    "target_blind_otsu3_positive_coverage",
    "target_blind_otsu3_selected_purity",
)
INITIAL_FIELD_SHA = "328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66"
FIELD_A_SHA = "9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067"
READOUT_SHA = "5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb"
RENDERER_SHA = "6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2"
GRAPH_SHA = "abcdd466fbbd726f277b59b137a59ac93b0a2c270a7557fc9916a478a66a1451"
DIRECT_EVALUATOR_SHA = "d186e517c8152e7cbc0f4845d035898097542b8b3ecad134f597af7f12800168"
REGISTRATION_SHA = "7c539fb523c7152446bdc5f28325986a9162baa6c85a5608a66552023aa869c4"


def _load(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError(f"artifact SHA differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON mapping: {path}")
    return value


def _runtime(log_root: Path, stage: str) -> dict:
    telemetry_path = log_root / f"{stage}.telemetry.csv"
    owner_path = log_root / f"{stage}.owner_audit.csv"
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with owner_path.open(newline="", encoding="utf-8") as handle:
        owners = list(csv.DictReader(handle))
    if not rows or not owners or any(row["gpu"] != "1" for row in rows):
        raise ValueError(f"{stage} did not run exclusively on physical GPU1")
    events = [row["event"] for row in rows]
    if events[-1] != "cuda_release_verified_no_compute_owner":
        raise ValueError(f"{stage} did not verify CUDA release")
    unsafe = [
        event
        for event in events
        if "abort" in event or "failed" in event or "foreign_compute_owner" in event
    ]
    foreign = [
        row.get("foreign_owner_pids", "").strip()
        for row in owners
        if row.get("foreign_owner_pids", "").strip()
    ]
    owner_events = [row.get("event") for row in owners]
    if unsafe or foreign or "postexit_owner_clear" not in owner_events:
        raise ValueError(f"{stage} GPU ownership/guard audit failed")
    return {
        "physical_gpu": 1,
        "max_temperature_c": max(int(row["temp_c"]) for row in rows),
        "max_power_w": max(float(row["power_w"]) for row in rows),
        "max_memory_mib": max(int(row["memory_mib"]) for row in rows),
        "soft_pause_events": sum(event.startswith("soft_pause") for event in events),
        "thermal_abort_events": sum("thermal_abort" in event for event in events),
        "foreign_owner_pids": foreign,
        "cuda_release_verified": True,
        "telemetry_sha256": sha256_file(telemetry_path),
        "owner_audit_sha256": sha256_file(owner_path),
    }


def summarize(root: Path) -> dict:
    pool = Path("/mnt/pool/sqy/results/RADIO-GS/output")
    baseline_quality_path = pool / (
        "optimization_20260803/lerf_text_audit/score_quality_v1/figurines/"
        "figurines/lerf_direct_3d_selection_results.json"
    )
    baseline_score_report_path = pool / (
        "optimization_20260802/lerf_native_multiscale_query_scores_v2/figurines.pt.json"
    )
    baseline_descriptor_report_path = pool / (
        "optimization_20260802/lerf_native_multiscale_descriptor_v2/figurines.pt.json"
    )
    quality_path = root / (
        "score_quality/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    fixed_path = root / "fixed/figurines/figurines/lerf_direct_3d_selection_results.json"
    otsu_path = root / "otsu3/figurines/figurines/lerf_direct_3d_selection_results.json"
    score_report_path = root / "query_scores/figurines_field_a.pt.json"
    descriptor_report_path = root / "descriptors/figurines_field_a.pt.json"

    baseline_quality = _load(
        baseline_quality_path,
        "716cb59dc811880e9262d1ab3605ee335b5b054266261ad54c9ddf53654daf35",
    )
    quality = _load(quality_path)
    fixed = _load(fixed_path)
    otsu = _load(otsu_path)
    baseline_score = _load(baseline_score_report_path)["shared_renderer_authority"]
    score = _load(score_report_path)["shared_renderer_authority"]
    baseline_descriptor = _load(baseline_descriptor_report_path)["metadata"]
    descriptor = _load(descriptor_report_path)["metadata"]

    for observed, expected, label in (
        (baseline_score["geometry_axis"]["field_checkpoint_sha256"], INITIAL_FIELD_SHA, "initial field"),
        (score["geometry_axis"]["field_checkpoint_sha256"], FIELD_A_SHA, "Field-A field"),
        (score["descriptor_axis"]["readout_checkpoint_sha256"], READOUT_SHA, "readout"),
        (score["geometry_axis"]["renderer_geometry_checkpoint_sha256"], RENDERER_SHA, "renderer"),
        (descriptor["support_graph_sha256"], GRAPH_SHA, "support graph"),
    ):
        if observed != expected:
            raise ValueError(f"{label} SHA differs")
    for key in ("scale_axis", "query_axis"):
        if baseline_score[key] != score[key]:
            raise ValueError(f"initial/Field-A {key} differs")
    if baseline_descriptor["region_contract_sha256"] != descriptor["region_contract_sha256"]:
        raise ValueError("initial/Field-A SurfaceRegion contract differs")
    if descriptor.get("canonical_radio_source") != "field_decode_only":
        raise ValueError("Field-A descriptor was not decoded from the compact field")
    if any(
        descriptor.get(key) is not False
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("Field-A descriptor was label/query contaminated")
    for result in (fixed, otsu):
        protocol = result["protocol"]
        if protocol["checkpoint_sha256"] != RENDERER_SHA:
            raise ValueError("Field-A result renderer differs")
        if protocol["score_aggregation"] != "none" or protocol["selection_refinement"] != "none":
            raise ValueError("Field-A result added an unregistered support operator")
    if otsu["adaptive_support_diagnostic"]["frozen_evaluator_source_sha256"] != DIRECT_EVALUATOR_SHA:
        raise ValueError("Field-A Otsu evaluator source differs")

    baseline_diagnostic = baseline_quality["score_quality_diagnostic"]
    diagnostic = quality["score_quality_diagnostic"]
    for item in (baseline_diagnostic, diagnostic):
        receipt = item["method_receipt_frozen_before_labels"]
        if receipt["experiment_registration_sha256"] != REGISTRATION_SHA:
            raise ValueError("LERF experiment registration differs")
        if receipt["frozen_evaluator_source_sha256"] != DIRECT_EVALUATOR_SHA:
            raise ValueError("LERF frozen evaluator differs")
    initial = {key: float(baseline_diagnostic["aggregate_object_mean"][key]) for key in METRICS}
    final = {key: float(diagnostic["aggregate_object_mean"][key]) for key in METRICS}
    if int(baseline_diagnostic["objects"]) != 56 or int(diagnostic["objects"]) != 56:
        raise ValueError("initial/Field-A figurines object count differs")

    return {
        "schema_version": "lerf_field_a_figurines_isolated_comparison_v1",
        "status": "complete_post_gate_fixed_protocol_comparison",
        "scene": "figurines",
        "objects": 56,
        "isolation": {
            "new_descriptor_built_from_field_a": True,
            "old_field_descriptor_reused": False,
            "query_conditioned_graph_enabled": False,
            "score_aggregation": "none",
            "selection_refinement": "none",
            "same_surface_region_contract": True,
            "same_text_query_axis": True,
            "same_frozen_evaluator": True,
        },
        "authorities": {
            "initial_field_sha256": INITIAL_FIELD_SHA,
            "field_a_sha256": FIELD_A_SHA,
            "readout_sha256": READOUT_SHA,
            "renderer_sha256": RENDERER_SHA,
            "support_graph_sha256": GRAPH_SHA,
            "frozen_direct_evaluator_sha256": DIRECT_EVALUATOR_SHA,
            "experiment_registration_sha256": REGISTRATION_SHA,
        },
        "metrics": {
            "initial_field": initial,
            "field_a": final,
            "field_a_minus_initial": {key: final[key] - initial[key] for key in METRICS},
        },
        "artifacts": {
            "descriptor": {"path": str((root / "descriptors/figurines_field_a.pt").resolve()), "sha256": score["source_artifacts"]["descriptor_cache"]["sha256"]},
            "query_scores": {"path": str((root / "query_scores/figurines_field_a.pt").resolve()), "sha256": _load(score_report_path)["query_score_cache"]["sha256"]},
            "fixed_result": {"path": str(fixed_path.resolve()), "sha256": sha256_file(fixed_path)},
            "otsu3_result": {"path": str(otsu_path.resolve()), "sha256": sha256_file(otsu_path)},
            "score_quality_result": {"path": str(quality_path.resolve()), "sha256": sha256_file(quality_path)},
            "baseline_score_quality_result": {"path": str(baseline_quality_path.resolve()), "sha256": sha256_file(baseline_quality_path)},
        },
        "runtime": {
            stage: _runtime(root / "logs", stage)
            for stage in ("descriptor", "fixed", "otsu3", "score_quality")
        },
        "claim_boundary": (
            "This post-gate comparison isolates the first frozen Field-A checkpoint "
            "under the unchanged SurfaceRegion/text scorer. AP and oracle IoU are "
            "label-aware diagnostics and did not select the field or readout."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output exists: {output}")
    value = summarize(Path(args.root).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
