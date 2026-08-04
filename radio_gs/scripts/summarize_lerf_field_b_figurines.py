#!/usr/bin/env python3
"""Seal the post-gate Field-B figurines LERF isolated/support comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from radio_gs.utils.immutable_artifacts import sha256_file


FIELD_B_SHA = "f9b2d926c7a455acd6fcb2b87e3be9e385cd5cf2357204d0867546da93d9b710"
FIELD_A_SHA = "9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067"
INITIAL_SHA = "328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66"
FIELD_B_GATE_SHA = "0313702dbf7547485d9f8325003ad206e64db57301bd4c9fc46fc162ef4efcf5"
METHOD_SHA = "e229da73cbcc98b4681cdad698b4035e0a58724bd307e37824cf0ace16bd9319"
READOUT_SHA = "5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb"
RENDERER_SHA = "6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2"
GRAPH_SHA = "abcdd466fbbd726f277b59b137a59ac93b0a2c270a7557fc9916a478a66a1451"
DIRECT_EVALUATOR_SHA = "d186e517c8152e7cbc0f4845d035898097542b8b3ecad134f597af7f12800168"
VALA_RESULT_SHA = "1a4a4ce2856b2af7e83d0c157aebf3498d1740ecc5deaefc39403c0daa01fcae"
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


def _load(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError(f"artifact SHA differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON mapping: {path}")
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
        raise ValueError(f"{stage} GPU1 runtime audit failed")
    return {
        "physical_gpu": 1,
        "max_temperature_c": max(int(row["temp_c"]) for row in rows),
        "max_power_w": max(float(row["power_w"]) for row in rows),
        "max_memory_mib": max(int(row["memory_mib"]) for row in rows),
        "soft_pause_events": sum(event.startswith("soft_pause") for event in events),
        "thermal_abort_events": sum("thermal_abort" in event for event in events),
        "foreign_owner_pids": foreign,
        "cuda_release_verified": True,
        "telemetry_sha256": sha256_file(telemetry),
        "owner_audit_sha256": sha256_file(owner),
    }


def summarize(isolated_root: Path, support_root: Path) -> dict:
    pool = Path("/mnt/pool/sqy/results/RADIO-GS/output")
    baseline_quality_path = pool / (
        "optimization_20260803/lerf_text_audit/score_quality_v1/figurines/"
        "figurines/lerf_direct_3d_selection_results.json"
    )
    field_a_quality_path = pool / (
        "optimization_20260803/lerf_field_a_figurines_v1/score_quality/figurines/"
        "figurines/lerf_direct_3d_selection_results.json"
    )
    field_b_quality_path = isolated_root / (
        "score_quality/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    field_b_fixed_path = isolated_root / (
        "fixed/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    field_b_otsu_path = isolated_root / (
        "otsu3/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    descriptor_report_path = isolated_root / "descriptors/figurines_field_b.pt.json"
    score_report_path = isolated_root / "query_scores/figurines_field_b.pt.json"
    field_b_support_path = support_root / (
        "figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    field_b_support_receipt_path = support_root / "figurines/method_receipt.prelabel.json"
    field_a_support_path = pool / (
        "optimization_20260803/lerf_field_a_query_conditioned_support_v1/"
        "figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    field_a_support_receipt_path = pool / (
        "optimization_20260803/lerf_field_a_query_conditioned_support_v1/"
        "figurines/method_receipt.prelabel.json"
    )
    old_support_path = pool / (
        "optimization_20260803/lerf_text_audit/query_conditioned_support_v1/"
        "figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    old_support_receipt_path = pool / (
        "optimization_20260803/lerf_text_audit/query_conditioned_support_v1/"
        "figurines/method_receipt.prelabel.json"
    )
    capability_path = support_root / "figurines_field_b_capability.pt"
    capability_report_path = capability_path.with_suffix(".pt.json")
    graph_path = support_root / "figurines_field_b_support_graph_k16.pt"
    graph_report_path = graph_path.with_suffix(".pt.json")
    gate_path = pool / (
        "optimization_20260803/field_b/figurines_field_b_label_free_gate_receipt.json"
    )
    vala_path = pool / (
        "protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/evaluation/"
        "all_metrics_30000_0.6.json"
    )

    gate = _load(gate_path, FIELD_B_GATE_SHA)
    if gate["gate"]["passed"] is not True or gate["final_field"]["sha256"] != FIELD_B_SHA:
        raise ValueError("Field-B label-free gate authority differs")
    baseline = _load(baseline_quality_path)["score_quality_diagnostic"]
    field_a = _load(field_a_quality_path)["score_quality_diagnostic"]
    field_b = _load(field_b_quality_path)["score_quality_diagnostic"]
    fixed = _load(field_b_fixed_path)
    otsu = _load(field_b_otsu_path)
    descriptor = _load(descriptor_report_path)["metadata"]
    score_report = _load(score_report_path)["shared_renderer_authority"]
    if descriptor.get("canonical_radio_source") != "field_decode_only":
        raise ValueError("Field-B descriptor is not field-decode-only")
    if descriptor.get("support_graph_sha256") != GRAPH_SHA:
        raise ValueError("Field-B isolated readout graph differs")
    if any(
        descriptor.get(key) is not False
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("Field-B descriptor is contaminated")
    authorities = score_report
    if (
        authorities["geometry_axis"]["field_checkpoint_sha256"] != FIELD_B_SHA
        or authorities["descriptor_axis"]["readout_checkpoint_sha256"] != READOUT_SHA
        or authorities["geometry_axis"]["renderer_geometry_checkpoint_sha256"] != RENDERER_SHA
    ):
        raise ValueError("Field-B isolated score authority differs")
    for result in (fixed, otsu):
        protocol = result["protocol"]
        if protocol["checkpoint_sha256"] != RENDERER_SHA:
            raise ValueError("Field-B renderer differs")
        if protocol["score_aggregation"] != "none" or protocol["selection_refinement"] != "none":
            raise ValueError("Field-B isolated result added support")
    if otsu["adaptive_support_diagnostic"]["frozen_evaluator_source_sha256"] != DIRECT_EVALUATOR_SHA:
        raise ValueError("Field-B Otsu evaluator differs")

    isolated_values = {}
    for name, diagnostic in (("initial", baseline), ("field_a", field_a), ("field_b", field_b)):
        if int(diagnostic["objects"]) != 56:
            raise ValueError(f"{name} object count differs")
        isolated_values[name] = {
            key: float(diagnostic["aggregate_object_mean"][key]) for key in METRICS
        }

    support_results = {
        "field_b_support": _load(field_b_support_path),
        "field_a_support": _load(field_a_support_path),
        "old_field_support": _load(old_support_path),
    }
    support_receipts = {
        "field_b_support": _load(field_b_support_receipt_path),
        "field_a_support": _load(field_a_support_receipt_path),
        "old_field_support": _load(old_support_receipt_path),
    }
    for receipt in support_receipts.values():
        if receipt["method_config_sha256"] != METHOD_SHA:
            raise ValueError("query-support method config differs")
        for key in ("target_rgb_opened", "target_masks_opened", "target_metrics_opened"):
            if receipt.get(key) is not False:
                raise ValueError("query-support prelabel receipt is contaminated")
    reference = support_receipts["old_field_support"]
    for receipt in support_receipts.values():
        for key in (
            "implementation_source_sha256",
            "implementation_dependency_source_sha256",
            "frozen_evaluator_source_sha256",
            "experiment_registration_sha256",
        ):
            if receipt[key] != reference[key]:
                raise ValueError(f"query-support implementation differs: {key}")
    expected_fields = {
        "field_b_support": FIELD_B_SHA,
        "field_a_support": FIELD_A_SHA,
        "old_field_support": INITIAL_SHA,
    }
    for name, expected in expected_fields.items():
        if support_receipts[name]["surface_graph"]["field_checkpoint_sha256"] != expected:
            raise ValueError(f"{name} field checkpoint differs")

    capability = _load(capability_report_path)
    graph = _load(graph_report_path)
    if capability["field_checkpoint_sha256"] != FIELD_B_SHA or capability["valid_gaussians"] != 82603:
        raise ValueError("Field-B capability cache differs")
    if any(
        capability.get(key) is not False
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("Field-B capability cache is contaminated")
    if graph["num_nodes"] != 82603 or graph["num_edges"] != 1651042:
        raise ValueError("Field-B graph topology size differs")
    for key, expected in EXPECTED_GRAPH_CONFIG.items():
        if graph["graph_config"].get(key) != expected:
            raise ValueError(f"Field-B graph config differs: {key}")
    affinity = graph.get("capability_affinity", graph.get("feature_hash", {}))
    if affinity.get("mode", "signed_hash") != "signed_hash" or affinity.get("output_dim") != 128:
        raise ValueError("Field-B graph affinity differs")

    vala = _load(vala_path, VALA_RESULT_SHA)
    vala_figurines = float(vala["per_scene"]["figurines"]["mIoU"])
    support_values = {
        name: float(value["scene"]["results"]["thr0p6"]["miou"])
        for name, value in support_results.items()
    }
    support_values["reproduced_vala"] = vala_figurines
    field_b_support = support_values["field_b_support"]
    field_b_otsu = isolated_values["field_b"]["target_blind_otsu3_miou"]
    comparisons = {
        "field_b_support_minus_field_b_otsu3": field_b_support - field_b_otsu,
        "field_b_support_minus_field_a_support": field_b_support - support_values["field_a_support"],
        "field_b_support_minus_old_field_support": field_b_support - support_values["old_field_support"],
        "field_b_support_minus_reproduced_vala": field_b_support - vala_figurines,
    }
    return {
        "schema_version": "lerf_field_b_figurines_post_gate_comparison_v1",
        "status": "complete_fixed_protocol_figurines_only",
        "scene": "figurines",
        "objects": 56,
        "field_b_label_free_gate": {
            "path": str(gate_path.resolve()),
            "sha256": FIELD_B_GATE_SHA,
            "passed": True,
        },
        "isolated_metrics": isolated_values,
        "field_b_isolated_minus_initial": {
            key: isolated_values["field_b"][key] - isolated_values["initial"][key]
            for key in METRICS
        },
        "field_b_isolated_minus_field_a": {
            key: isolated_values["field_b"][key] - isolated_values["field_a"][key]
            for key in METRICS
        },
        "support_metrics": support_values,
        "support_comparisons": comparisons,
        "support_authority": {
            "method_config_sha256": METHOD_SHA,
            "implementation_source_sha256": reference["implementation_source_sha256"],
            "implementation_dependency_source_sha256": reference[
                "implementation_dependency_source_sha256"
            ],
            "nodes": graph["num_nodes"],
            "edges": graph["num_edges"],
            "graph_config": graph["graph_config"],
            "affinity": affinity,
        },
        "artifacts": {
            "field_b_descriptor": {
                "path": str((isolated_root / "descriptors/figurines_field_b.pt").resolve()),
                "sha256": sha256_file(isolated_root / "descriptors/figurines_field_b.pt"),
            },
            "field_b_query_scores": {
                "path": str((isolated_root / "query_scores/figurines_field_b.pt").resolve()),
                "sha256": sha256_file(isolated_root / "query_scores/figurines_field_b.pt"),
            },
            "field_b_capability": {"path": str(capability_path.resolve()), "sha256": sha256_file(capability_path)},
            "field_b_graph": {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)},
            "field_b_support_result": {"path": str(field_b_support_path.resolve()), "sha256": sha256_file(field_b_support_path)},
            "reproduced_vala_result": {"path": str(vala_path.resolve()), "sha256": VALA_RESULT_SHA},
        },
        "runtime": {
            "isolated": {
                stage: _runtime(isolated_root / "logs", stage)
                for stage in ("descriptor", "fixed", "otsu3", "score_quality")
            },
            "support": {
                stage: _runtime(support_root / "logs", stage)
                for stage in ("capability", "query_support")
            },
        },
        "interpretation": (
            "Field-B materially improves query-independent relation fidelity, but the frozen "
            "k16 signed-hash support path converts it into only a very small gain over Field-A "
            "support, remains below old-field support, and leaves a large gap to reproduced VALA."
        ),
        "claim_boundary": (
            "This post-gate evaluation uses only figurines because no Field-B checkpoint exists "
            "for other scenes. No threshold, graph parameter, margin, or loss weight was changed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-root", required=True)
    parser.add_argument("--support-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output exists: {output}")
    value = summarize(
        Path(args.isolated_root).expanduser().resolve(),
        Path(args.support_root).expanduser().resolve(),
    )
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
