#!/usr/bin/env python3
"""Seal the post-freeze Field-A plus frozen LERF query-support combination."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from radio_gs.utils.immutable_artifacts import sha256_file


FIELD_A_SHA = "9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067"
OLD_FIELD_SHA = "328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66"
METHOD_SHA = "e229da73cbcc98b4681cdad698b4035e0a58724bd307e37824cf0ace16bd9319"
EXPECTED_GRAPH_CONFIG = {
    "neighbors": 16,
    "spatial_scale": 2.0,
    "appearance_temperature": 0.1,
    "boundary_temperature": 0.1,
    "normal_temperature": 0.2,
    "covisibility_weight": 0.0,
    "minimum_sigma": 0.0001,
    "affinity_chunk_size": 65536,
    "topology_mode": "symmetric_union",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return value


def _runtime(log_root: Path, stage: str) -> dict:
    telemetry = log_root / f"{stage}.telemetry.csv"
    owner = log_root / f"{stage}.owner_audit.csv"
    with telemetry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with owner.open(newline="", encoding="utf-8") as handle:
        owners = list(csv.DictReader(handle))
    events = [row["event"] for row in rows]
    foreign = [
        row.get("foreign_owner_pids", "").strip()
        for row in owners
        if row.get("foreign_owner_pids", "").strip()
    ]
    if (
        not rows
        or any(row["gpu"] != "1" for row in rows)
        or events[-1] != "cuda_release_verified_no_compute_owner"
        or foreign
        or any("abort" in event or "failed" in event for event in events)
    ):
        raise ValueError(f"{stage} GPU1 audit failed")
    return {
        "physical_gpu": 1,
        "max_temperature_c": max(int(row["temp_c"]) for row in rows),
        "max_power_w": max(float(row["power_w"]) for row in rows),
        "max_memory_mib": max(int(row["memory_mib"]) for row in rows),
        "soft_pause_events": sum(event.startswith("soft_pause") for event in events),
        "thermal_abort_events": sum("thermal_abort" in event for event in events),
        "cuda_release_verified": True,
        "telemetry_sha256": sha256_file(telemetry),
        "owner_audit_sha256": sha256_file(owner),
    }


def summarize(root: Path) -> dict:
    pool = Path("/mnt/pool/sqy/results/RADIO-GS/output")
    old_root = pool / "optimization_20260803/lerf_text_audit"
    old_result_path = old_root / (
        "query_conditioned_support_v1/figurines/figurines/"
        "lerf_direct_3d_selection_results.json"
    )
    old_receipt_path = old_root / (
        "query_conditioned_support_v1/figurines/method_receipt.prelabel.json"
    )
    quality_path = old_root / (
        "score_quality_v1/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    isolated_receipt_path = pool / (
        "optimization_20260803/lerf_field_a_figurines_v1/field_a_vs_initial_receipt.json"
    )
    result_path = root / "figurines/figurines/lerf_direct_3d_selection_results.json"
    receipt_path = root / "figurines/method_receipt.prelabel.json"
    capability_report_path = root / "figurines_field_a_capability.pt.json"
    graph_report_path = root / "figurines_field_a_support_graph_k16.pt.json"
    capability_path = root / "figurines_field_a_capability.pt"
    graph_path = root / "figurines_field_a_support_graph_k16.pt"

    old_result = _load(old_result_path)
    old_receipt = _load(old_receipt_path)
    quality = _load(quality_path)["score_quality_diagnostic"]["aggregate_object_mean"]
    isolated = _load(isolated_receipt_path)
    result = _load(result_path)
    receipt = _load(receipt_path)
    capability = _load(capability_report_path)
    graph = _load(graph_report_path)
    if receipt["method_config_sha256"] != METHOD_SHA or old_receipt[
        "method_config_sha256"
    ] != METHOD_SHA:
        raise ValueError("old/new frozen query-support method differs")
    for key in (
        "implementation_source_sha256",
        "implementation_dependency_source_sha256",
        "frozen_evaluator_source_sha256",
        "experiment_registration_sha256",
    ):
        if receipt[key] != old_receipt[key]:
            raise ValueError(f"old/new query-support implementation differs: {key}")
    if old_receipt["surface_graph"]["field_checkpoint_sha256"] != OLD_FIELD_SHA:
        raise ValueError("old support result does not bind the initial field")
    if receipt["surface_graph"]["field_checkpoint_sha256"] != FIELD_A_SHA:
        raise ValueError("new support result does not bind Field-A")
    if capability["field_checkpoint_sha256"] != FIELD_A_SHA or capability[
        "valid_gaussians"
    ] != 82603:
        raise ValueError("Field-A capability cache authority differs")
    if any(
        capability.get(key) is not False
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("Field-A capability cache is contaminated")
    if graph["num_nodes"] != 82603 or graph["num_edges"] != old_receipt[
        "surface_graph"
    ]["num_edges"]:
        raise ValueError("old/new graph topology size differs")
    for key, expected in EXPECTED_GRAPH_CONFIG.items():
        if graph["graph_config"].get(key) != expected:
            raise ValueError(f"Field-A graph config differs: {key}")
    affinity = graph.get("capability_affinity", graph.get("feature_hash", {}))
    if affinity.get("mode", "signed_hash") != "signed_hash" or affinity.get(
        "output_dim"
    ) != 128:
        raise ValueError("Field-A graph affinity representation differs")
    for key in ("target_rgb_opened", "target_masks_opened", "target_metrics_opened"):
        if receipt.get(key) is not False:
            raise ValueError("query-support prelabel receipt is contaminated")

    new_support = float(result["scene"]["results"]["thr0p6"]["miou"])
    old_support = float(old_result["scene"]["results"]["thr0p6"]["miou"])
    old_otsu = float(quality["target_blind_otsu3_miou"])
    field_a_otsu = float(
        isolated["metrics"]["field_a"]["target_blind_otsu3_miou"]
    )
    metrics = {
        "field_a_query_support": new_support,
        "field_a_otsu3": field_a_otsu,
        "initial_field_query_support": old_support,
        "initial_field_otsu3": old_otsu,
        "field_a_support_minus_field_a_otsu3": new_support - field_a_otsu,
        "field_a_support_minus_initial_support": new_support - old_support,
        "field_a_support_minus_initial_otsu3": new_support - old_otsu,
    }
    return {
        "schema_version": "lerf_field_a_postfreeze_query_support_combination_v1",
        "status": "complete_postfreeze_combination_not_field_selection",
        "scene": "figurines",
        "method_config_sha256": METHOD_SHA,
        "metrics": metrics,
        "interpretation": (
            "The frozen query-support operator still improves Field-A Otsu3, but "
            "the Field-A-derived relation graph does not improve over the initial-"
            "field graph. Field-A trained exact per-node capability fidelity with "
            "relation_weight=0 and therefore did not claim relation preservation."
        ),
        "artifacts": {
            "capability_cache": {"path": str(capability_path.resolve()), "sha256": sha256_file(capability_path)},
            "support_graph": {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)},
            "prelabel_receipt": {"path": str(receipt_path.resolve()), "sha256": sha256_file(receipt_path)},
            "result": {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)},
            "initial_support_result": {"path": str(old_result_path.resolve()), "sha256": sha256_file(old_result_path)},
            "isolated_field_a_receipt": {"path": str(isolated_receipt_path.resolve()), "sha256": sha256_file(isolated_receipt_path)},
        },
        "runtime": {
            stage: _runtime(root / "logs", stage)
            for stage in ("capability", "query_support")
        },
        "claim_boundary": (
            "This combination was requested only after the first Field-A checkpoint "
            "and its isolated LERF result were frozen. It cannot select or tune Field-A."
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
