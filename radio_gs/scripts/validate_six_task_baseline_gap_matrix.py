#!/usr/bin/env python3
"""Validate the fail-closed six-task destination baseline gap matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "paper/artifacts/six_task_destination_compliant_baseline_gap_matrix_20260812.json"
)
EXPECTED_TASKS = [
    "LERF-2D",
    "LERF-3D",
    "NVOS",
    "SPIn-NeRF",
    "ScanNet OVS",
    "AGILE3D",
]
GAP_CLASSES = {
    "protocol_incompleteness",
    "implementation_defect",
    "representation_deficit",
    "readout_deficit",
    "multi_task_negative_transfer",
}


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
    _require(
        [str(_mapping(row, "task row").get("task")) for row in raw_rows]
        == EXPECTED_TASKS,
        "task order or membership drifted",
    )
    rows = {str(row["task"]): _mapping(row, str(row["task"])) for row in raw_rows}
    _require(len(rows) == len(EXPECTED_TASKS), "task names must be unique")
    for task, row in rows.items():
        _require(row.get("row_verdict") == "no_eligible_row", f"{task}: row became eligible")
        _require(row.get("runtime_compliance_proof") == "absent", f"{task}: proof status drifted")
        gaps = _mapping(row.get("gap_classification"), f"{task} gap classification")
        _require(set(gaps) == GAP_CLASSES, f"{task}: gap taxonomy is incomplete")
        _require(
            _mapping(gaps["multi_task_negative_transfer"], f"{task} transfer gap").get("status")
            == "unmeasured",
            f"{task}: negative transfer was asserted without a joint comparison",
        )
    return rows


def _close(actual: Any, expected: float, label: str) -> None:
    _require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12), f"{label} drifted")


def _validate_lerf2d(
    row: Mapping[str, Any], evidence: Mapping[str, bytes]
) -> None:
    source = _mapping(json.loads(evidence["lerf2d_current_row"]), "LERF-2D source")
    exact = _mapping(source["current_field_exact_coarse"], "LERF-2D exact row")
    metrics = _mapping(exact["scene_macro"], "LERF-2D source metrics")
    observed = _mapping(row["observed_metrics_percent"], "LERF-2D matrix metrics")
    _close(observed["mIoU"], 100.0 * float(metrics["miou"]), "LERF-2D mIoU")
    _close(observed["LocAcc"], 100.0 * float(metrics["loc_acc"]), "LERF-2D LocAcc")
    diagnostic = _mapping(source["current_field_target_rgb_grabcut_v1"], "LERF-2D diagnostic")
    delta = _mapping(diagnostic["delta_from_current_exact"], "LERF-2D diagnostic delta")
    _close(delta["scene_macro_miou"], 0.011999696645544, "LERF-2D boundary delta")
    _require(diagnostic.get("target_rgb_used") is True, "LERF-2D diagnostic lost its RGB boundary")


def _validate_lerf3d(
    row: Mapping[str, Any], evidence: Mapping[str, bytes]
) -> None:
    source = _mapping(json.loads(evidence["lerf3d_current_row"]), "LERF-3D source")
    results = _mapping(source["same_field_same_cache_causal_results"], "LERF-3D results")
    legacy = _mapping(_mapping(results["scene_macro"], "LERF-3D macro")["legacy"], "LERF-3D legacy")
    observed = _mapping(row["observed_metrics_percent"], "LERF-3D matrix metrics")
    for matrix_key, source_key in (
        ("mIoU", "miou"),
        ("Acc@0.25", "acc025"),
        ("Acc@0.50", "acc050"),
    ):
        _close(observed[matrix_key], 100.0 * float(legacy[source_key]), f"LERF-3D {matrix_key}")
    gate = _mapping(source["preregistered_gate"], "LERF-3D gate")
    _require(gate.get("overall_passed") is False, "LERF-3D failed readout was promoted")


def _validate_nvos(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["nvos_current_row"]), "NVOS source")
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
    _require("INTER_LINEAR" in str(scoring.get("protocol")), "NVOS interpolation mismatch disappeared")
    _require("Posthoc" in str(source.get("claim_boundary")), "NVOS posthoc boundary disappeared")
    aggregate = _mapping(source["aggregate_comparison"], "NVOS aggregate comparison")
    _require(aggregate.get("regressed_vs_compact_scene_count") == 3, "NVOS regression count drifted")


def _validate_spin(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["spin_context_row"]), "SPIn source")
    metrics = _mapping(source["macro_foreground_iou"], "SPIn metrics")
    observed = _mapping(row["observed_metrics_percent"], "SPIn matrix metrics")
    _close(
        observed["macro_foreground_IoU"],
        100.0 * float(metrics["previous_canonical_mainline"]),
        "SPIn field-only IoU",
    )
    method = _mapping(source["method"], "SPIn method")
    _require("target-RGB" in str(method.get("proposal_branch")), "SPIn mixed RGB branch disappeared")
    audit = _mapping(
        json.loads(evidence["spin_comparator_eligibility_audit"]),
        "SPIn comparator audit",
    )
    resolution = _mapping(audit["resolution"], "SPIn comparator resolution")
    _require(resolution.get("numeric_sota_target") is None, "SPIn numeric target must remain absent")
    _require(resolution.get("target_status") == "no_eligible_target", "SPIn target status drifted")
    _require(row.get("numeric_sota_target_percent") is None, "matrix invented a SPIn target")


def _validate_scannet(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    source = _mapping(json.loads(evidence["scannet_ovs_current_row"]), "ScanNet source")
    source_metrics = _mapping(source["metrics"], "ScanNet source metrics")
    observed = _mapping(row["observed_metrics_percent"], "ScanNet matrix metrics")
    for split in ("19", "15", "10"):
        split_metrics = _mapping(source_metrics[split], f"ScanNet {split} metrics")
        _close(observed[f"{split}-mIoU"], 100.0 * float(split_metrics["miou"]), f"ScanNet {split} mIoU")
        _close(observed[f"{split}-mAcc"], 100.0 * float(split_metrics["macc"]), f"ScanNet {split} mAcc")
    method = _mapping(source["method_binding"], "ScanNet method")
    _require("semantic_score_cache_contract" in method, "ScanNet score-cache dependency disappeared")
    derivation = _mapping(source["derivation_binding"], "ScanNet derivation")
    _require("mpr_source" in derivation.get("receipt_transitive_sources", []), "ScanNet MPR lineage disappeared")
    targets = _mapping(row["exact_sota_target_percent"], "ScanNet target")
    _require(all(float(observed[key]) >= float(value) for key, value in targets.items()), "ScanNet numeric pass drifted")


def _validate_agile(row: Mapping[str, Any], evidence: Mapping[str, bytes]) -> None:
    registry = _mapping(yaml.safe_load(evidence["six_task_context_registry"]), "six-task registry")
    tasks = _mapping(registry["tasks"], "six-task registry tasks")
    agile = _mapping(tasks["agile3d"], "AGILE3D registry")
    current_rows = agile.get("current_rows")
    _require(isinstance(current_rows, list) and len(current_rows) == 1, "AGILE3D context row drifted")
    current = _mapping(current_rows[0], "AGILE3D current context row")
    _require(current.get("id") == "canonical_dense20_cellseed_pilot", "AGILE3D pilot identity drifted")
    source_metrics = _mapping(current["metrics"], "AGILE3D source metrics")
    observed = _mapping(row["observed_context_metrics_percent"], "AGILE3D matrix metrics")
    for click in (1, 2, 3, 5, 10):
        _close(
            observed[f"IoU@{click}"],
            100.0 * float(source_metrics[f"iou_at_{click}"]),
            f"AGILE3D IoU@{click}",
        )
    _require(float(observed["IoU@2"]) < float(observed["IoU@1"]), "AGILE3D unstable click trajectory disappeared")
    external = _mapping(row["context_evidence_artifact"], "AGILE3D external artifact")
    _require(external.get("repository_portable") is False, "AGILE3D context artifact was relabelled portable")
    path = REPO_ROOT / str(external.get("path"))
    if path.is_file():
        payload = path.read_bytes()
        _require(_sha256_bytes(payload) == external.get("observed_sha256"), "AGILE3D external artifact drifted")
        raw = _mapping(json.loads(payload), "AGILE3D external result")
        protocol = _mapping(raw["protocol"], "AGILE3D external protocol")
        _require(protocol.get("scenes") == 20 and protocol.get("objects") == 804, "AGILE3D pilot cohort drifted")
        _require(protocol.get("max_clicks") == 20, "AGILE3D pilot click cap drifted")
        support = raw.get("scene_support")
        _require(isinstance(support, list) and support, "AGILE3D scene support missing")
        _require(all("capability_cache" in item and "support_graph" in item for item in support), "AGILE3D cache lineage drifted")


def validate(matrix_path: Path = DEFAULT_MATRIX) -> dict[str, Any]:
    matrix = _read_json(matrix_path)
    _require(
        matrix.get("schema") == "radio_gs.six_task_destination_compliant_baseline_gap_matrix.v1",
        "wrong matrix schema",
    )
    _require(matrix.get("schema_version") == 1, "wrong matrix schema version")
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
    _validate_agile(rows["AGILE3D"], evidence)

    transfer = _mapping(matrix.get("cross_task_gap"), "cross_task_gap")
    _require(
        transfer.get("multi_task_negative_transfer_status") == "unmeasured",
        "negative transfer was asserted without one paired six-task candidate",
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
    _require(resolution.get("rerun_performed") is False, "matrix unexpectedly claims a rerun")
    return {
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": _sha256_bytes(matrix_path.read_bytes()),
        "task_count": len(rows),
        "eligible_task_row_count": 0,
        "numerically_passing_context_rows": resolution.get("numerically_passing_context_rows"),
        "verdict": "no_eligible_joint_development_baseline",
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
