#!/usr/bin/env python3
"""Validate the fail-closed five-contract destination baseline gap matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "paper/artifacts/five_contract_destination_compliant_baseline_gap_matrix_20260813.json"
)
EXPECTED_TASKS = ["LERF-2D", "LERF-3D", "NVOS", "SPIn-NeRF", "ScanNet OVS"]
EXPECTED_CONTRACTS = [
    "lerf2d-field-only-four-scene-v1",
    "lerf3d-field-only-four-scene-v1",
    "nvos-ludvig-online-all-view-eight-v1",
    "spin-ludvig-online-full-mask-available-nine-v1",
    "scannet-ovs-paper8-v1",
]
GATE_NAMES = {
    "evaluation_contract",
    "comparator_identity",
    "runtime_compliance_proof",
    "single_compact_feature_field",
    "cold_start_storage",
    "canonical_query_interface",
    "exact_evaluator",
}
GATE_STATUSES = {"pass", "fail", "unproven", "partial"}


class MatrixError(ValueError):
    """Raised when the matrix or one of its evidence bindings has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _resolve_repository_evidence(matrix: Mapping[str, Any]) -> dict[str, bytes]:
    resolved: dict[str, bytes] = {}
    rows = matrix.get("repository_evidence")
    _require(isinstance(rows, list) and rows, "repository_evidence must be non-empty")
    for raw in rows:
        row = _mapping(raw, "repository evidence row")
        evidence_id = str(row.get("id", ""))
        _require(evidence_id and evidence_id not in resolved, "evidence ids must be unique")
        path = (REPO_ROOT / str(row.get("path", ""))).resolve()
        _require(path.is_file(), f"missing evidence: {path}")
        payload = path.read_bytes()
        _require(
            _sha256_bytes(payload) == row.get("sha256"),
            f"SHA-256 drift: {evidence_id}",
        )
        resolved[evidence_id] = payload
    return resolved


def _task_rows(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_rows = matrix.get("task_matrix")
    _require(isinstance(raw_rows, list), "task_matrix must be a list")
    task_order = [str(_mapping(row, "task row").get("task")) for row in raw_rows]
    _require(task_order == EXPECTED_TASKS, "task order or membership drifted")
    rows = {str(row["task"]): _mapping(row, str(row["task"])) for row in raw_rows}
    _require(len(rows) == len(EXPECTED_TASKS), "task names must be unique")
    for task, contract_id in zip(EXPECTED_TASKS, EXPECTED_CONTRACTS):
        row = rows[task]
        _require(row.get("evaluation_contract_id") == contract_id, f"{task}: contract drifted")
        _require(row.get("row_verdict") == "no_eligible_row", f"{task}: row became eligible")
        gates = _mapping(row.get("gate_audit"), f"{task} gate audit")
        _require(set(gates) == GATE_NAMES, f"{task}: gate membership drifted")
        statuses = []
        for gate_name, raw_gate in gates.items():
            gate = _mapping(raw_gate, f"{task} {gate_name}")
            status = str(gate.get("status"))
            _require(status in GATE_STATUSES, f"{task} {gate_name}: invalid status")
            _require(str(gate.get("finding", "")).strip() != "", f"{task} {gate_name}: missing finding")
            statuses.append(status)
        _require(any(status != "pass" for status in statuses), f"{task}: all hard gates passed unexpectedly")
    return rows


def _close(actual: Any, expected: float, label: str) -> None:
    _require(
        math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label} drifted",
    )


def _validate_lerf2d(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["lerf2d_historical_row"]), "LERF-2D source")
    exact = _mapping(source["current_field_exact_coarse"], "LERF-2D exact row")
    metrics = _mapping(exact["scene_macro"], "LERF-2D source metrics")
    observed = _mapping(row["observed_metrics_percent"], "LERF-2D matrix metrics")
    _close(observed["mIoU"], 100.0 * float(metrics["miou"]), "LERF-2D mIoU")
    _close(observed["LocAcc"], 100.0 * float(metrics["loc_acc"]), "LERF-2D LocAcc")
    floor = _mapping(row["mandatory_vpa_floor_percent"], "LERF-2D floor")
    _require(float(observed["mIoU"]) < float(floor["mIoU"]), "LERF-2D mIoU floor failure disappeared")
    _require(float(observed["LocAcc"]) >= float(floor["LocAcc"]), "LERF-2D LocAcc floor pass disappeared")
    diagnostic = _mapping(source["current_field_target_rgb_grabcut_v1"], "LERF-2D diagnostic")
    _require(diagnostic.get("target_rgb_used") is True, "LERF-2D forbidden-RGB diagnostic disappeared")


def _validate_lerf3d(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["lerf3d_historical_row"]), "LERF-3D source")
    results = _mapping(source["same_field_same_cache_causal_results"], "LERF-3D results")
    legacy = _mapping(_mapping(results["scene_macro"], "LERF-3D macro")["legacy"], "LERF-3D legacy")
    observed = _mapping(row["observed_metrics_percent"], "LERF-3D matrix metrics")
    for matrix_key, source_key in (("mIoU", "miou"), ("Acc@0.25", "acc025"), ("Acc@0.50", "acc050")):
        _close(observed[matrix_key], 100.0 * float(legacy[source_key]), f"LERF-3D {matrix_key}")
    floor = _mapping(row["mandatory_vpa_floor_percent"], "LERF-3D floor")
    _require(all(float(observed[key]) < float(value) for key, value in floor.items()), "LERF-3D floor failures disappeared")
    gate = _mapping(source["preregistered_gate"], "LERF-3D readout gate")
    _require(gate.get("overall_passed") is False, "LERF-3D failed readout was promoted")


def _validate_nvos(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["nvos_historical_row"]), "NVOS source")
    scoring = _mapping(source["exact_scoring"], "NVOS scoring")
    observed = _mapping(row["observed_metrics_percent"], "NVOS matrix metrics")
    _close(
        observed["macro_foreground_IoU"],
        100.0 * float(scoring["scene_macro_foreground_iou"]),
        "NVOS IoU",
    )
    _close(
        observed["macro_pixel_accuracy"],
        100.0 * float(scoring["scene_macro_pixel_accuracy"]),
        "NVOS pixel accuracy",
    )
    floor = _mapping(row["mandatory_vpa_floor_percent"], "NVOS floor")
    target = _mapping(row["conditional_sota_target_percent"], "NVOS target")
    _close(floor["macro_foreground_IoU"], 91.25768502741802, "NVOS VPA floor")
    _close(target["macro_foreground_IoU"], 92.4, "NVOS target")
    _require(float(observed["macro_foreground_IoU"]) < float(floor["macro_foreground_IoU"]), "NVOS floor failure disappeared")
    _require("Posthoc" in str(source.get("claim_boundary")), "NVOS posthoc status disappeared")
    query_gate = _mapping(_mapping(row["gate_audit"], "NVOS gates")["canonical_query_interface"], "NVOS query gate")
    _require(query_gate.get("status") == "fail", "NVOS historical query path was upgraded")


def _validate_spin(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["spin_historical_rows"]), "SPIn source")
    metrics = _mapping(source["macro_foreground_iou"], "SPIn source metrics")
    raw_rows = row.get("inspected_context_rows")
    _require(isinstance(raw_rows, list) and len(raw_rows) == 4, "SPIn context-row inventory drifted")
    matrix_rows = {str(_mapping(item, "SPIn context row")["row"]): item for item in raw_rows}
    expected = {
        "previous_canonical_mainline": 100.0 * float(metrics["previous_canonical_mainline"]),
        "full_carrier_sam_branch_without_canonical_fallback": 100.0 * float(metrics["full_carrier_sam_branch_without_canonical_fallback"]),
        "reference_selected_unified_interface": 100.0 * float(metrics["reference_selected_unified_interface"]),
        "reproduced_ludvig_sam": 100.0 * float(metrics["reproduced_ludvig_sam"]),
    }
    _require(set(matrix_rows) == set(expected), "SPIn context-row identities drifted")
    for row_id, value in expected.items():
        _close(matrix_rows[row_id]["macro_foreground_IoU_percent"], value, f"SPIn {row_id}")
    threshold = _mapping(
        row["conditional_sota_target_and_mandatory_vpa_floor_percent"],
        "SPIn target/floor",
    )
    _close(threshold["macro_foreground_IoU"], 93.7200449592385, "SPIn target/floor")
    _require(matrix_rows["reference_selected_unified_interface"]["numeric_status"] == "pass_context_only", "SPIn numerical context pass disappeared")
    method = _mapping(source["method"], "SPIn historical method")
    _require("three target-RGB SAM hypotheses" in str(method.get("proposal_branch")), "SPIn old proposal identity disappeared")
    query_gate = _mapping(_mapping(row["gate_audit"], "SPIn gates")["canonical_query_interface"], "SPIn query gate")
    _require(query_gate.get("status") == "fail", "SPIn historical query path was upgraded")
    _require(row.get("canonical_current_row") is None, "SPIn canonical slot was filled by retrofit")


def _validate_scannet(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["scannet_historical_row"]), "ScanNet source")
    source_metrics = _mapping(source["metrics"], "ScanNet source metrics")
    observed = _mapping(row["observed_metrics_percent"], "ScanNet matrix metrics")
    for split in ("19", "15", "10"):
        split_metrics = _mapping(source_metrics[split], f"ScanNet {split} metrics")
        _close(observed[f"{split}-mIoU"], 100.0 * float(split_metrics["miou"]), f"ScanNet {split} mIoU")
        _close(observed[f"{split}-mAcc"], 100.0 * float(split_metrics["macc"]), f"ScanNet {split} mAcc")
    targets = _mapping(row["exact_sota_target_and_mandatory_vpa_floor_percent"], "ScanNet target")
    _require(all(float(observed[key]) >= float(value) for key, value in targets.items()), "ScanNet numeric pass drifted")
    method = _mapping(source["method_binding"], "ScanNet method")
    _require("semantic_score_cache_contract" in method, "ScanNet score-cache dependency disappeared")
    derivation = _mapping(source["derivation_binding"], "ScanNet derivation")
    _require("mpr_source" in derivation.get("receipt_transitive_sources", []), "ScanNet MPR lineage disappeared")
    gates = _mapping(row["gate_audit"], "ScanNet gates")
    _require(_mapping(gates["exact_evaluator"], "ScanNet evaluator gate").get("status") == "pass", "ScanNet exact evaluator pass drifted")
    _require(_mapping(gates["cold_start_storage"], "ScanNet storage gate").get("status") == "fail", "ScanNet cache violation disappeared")


def _validate_promotion_inputs(matrix: Mapping[str, Any]) -> None:
    promotion = _mapping(matrix.get("promotion_inputs"), "promotion_inputs")
    _require(promotion.get("contract_order") == EXPECTED_CONTRACTS, "promotion contract order drifted")
    seeds = _mapping(promotion.get("seed_policy"), "seed policy")
    _require(seeds.get("stochastic_panel") == [0, 1, 2], "seed panel drifted")
    hard_gates = promotion.get("hard_gates")
    _require(isinstance(hard_gates, list) and len(hard_gates) >= 8, "promotion hard gates are incomplete")
    compiler = _mapping(
        matrix.get("selected_rgb_assisted_compiler_precondition"),
        "selected RGB-assisted compiler precondition",
    )
    _require(compiler.get("method_identity") == "registered2d-sam3-multiview-consensus-v1", "selected compiler drifted")
    _require(compiler.get("pilot_identity") == "registered2d-sam3-multiview-consensus-full8-plus-available9-v1", "compiler pilot drifted")
    _require(compiler.get("current_row_created") is False, "compiler decision was relabelled as a result")
    _require(compiler.get("status") == "not_run", "compiler pilot was relabelled as executed")


def validate(matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = _read_json(matrix_path)
    _require(
        matrix.get("schema") == "radio_gs.five_contract_destination_compliant_baseline_gap_matrix.v1",
        "wrong matrix schema",
    )
    _require(matrix.get("schema_version") == 1, "wrong matrix schema version")
    _require(
        matrix.get("planning_registry_id") == "five-benchmark-evaluation-contract-registry-v1",
        "planning registry drifted",
    )
    _require(
        matrix.get("status") == "complete_no_eligible_joint_development_baseline",
        "matrix status drifted",
    )
    evidence = _resolve_repository_evidence(matrix)
    rows = _task_rows(matrix)
    _validate_lerf2d(rows["LERF-2D"], evidence)
    _validate_lerf3d(rows["LERF-3D"], evidence)
    _validate_nvos(rows["NVOS"], evidence)
    _validate_spin(rows["SPIn-NeRF"], evidence)
    _validate_scannet(rows["ScanNet OVS"], evidence)
    _validate_promotion_inputs(matrix)

    transfer = _mapping(matrix.get("cross_task_gap"), "cross_task_gap")
    _require(
        transfer.get("joint_contract_negative_transfer_status") == "unmeasured",
        "negative transfer was asserted without one paired five-contract candidate",
    )
    resolution = _mapping(matrix.get("resolution"), "resolution")
    _require(resolution.get("joint_development_baseline") is None, "a virtual joint baseline was invented")
    _require(
        resolution.get("verdict") == "no_eligible_joint_development_baseline",
        "resolution verdict drifted",
    )
    _require(resolution.get("eligible_task_row_count") == 0, "eligible task row count drifted")
    _require(resolution.get("historical_row_stitching_permitted") is False, "row stitching was enabled")
    _require(resolution.get("metadata_only_retrofit_permitted") is False, "metadata retrofit was enabled")
    _require(resolution.get("benchmark_rerun_performed") is False, "matrix unexpectedly claims a rerun")
    expected_numeric_context = [
        "SPIn-NeRF: full_carrier_sam_branch_without_canonical_fallback",
        "SPIn-NeRF: reference_selected_unified_interface",
        "ScanNet OVS",
    ]
    _require(
        resolution.get("numerically_passing_context_rows") == expected_numeric_context,
        "numerically passing Context Evidence inventory drifted",
    )
    return {
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": _sha256_bytes(matrix_path.read_bytes()),
        "task_count": len(rows),
        "eligible_task_row_count": 0,
        "numerically_passing_context_rows": expected_numeric_context,
        "selected_rgb_assisted_compiler_status": "not_run",
        "verdict": "no_eligible_joint_development_baseline",
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
