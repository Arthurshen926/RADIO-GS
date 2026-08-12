import copy
import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_five_contract_baseline_gap_matrix import (
    DEFAULT_MATRIX,
    EXPECTED_CONTRACTS,
    MatrixError,
    validate,
)


def _matrix() -> dict:
    return json.loads(DEFAULT_MATRIX.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_matrix_reproduces_empty_five_contract_baseline() -> None:
    receipt = validate()
    assert receipt["verdict"] == "no_eligible_joint_development_baseline"
    assert receipt["task_count"] == 5
    assert receipt["eligible_task_row_count"] == 0
    assert receipt["selected_rgb_assisted_compiler_status"] == "not_run"
    assert receipt["numerically_passing_context_rows"] == [
        "SPIn-NeRF: full_carrier_sam_branch_without_canonical_fallback",
        "SPIn-NeRF: reference_selected_unified_interface",
        "ScanNet OVS",
    ]


def test_matrix_uses_only_the_five_current_contracts() -> None:
    matrix = _matrix()
    assert matrix["promotion_inputs"]["contract_order"] == EXPECTED_CONTRACTS
    assert [row["task"] for row in matrix["task_matrix"]] == [
        "LERF-2D",
        "LERF-3D",
        "NVOS",
        "SPIn-NeRF",
        "ScanNet OVS",
    ]


def test_matrix_rejects_virtual_joint_baseline(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix["resolution"]["joint_development_baseline"] = "stitched-best-of-task"
    with pytest.raises(MatrixError, match="virtual joint baseline"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_numerical_pass_as_eligibility(tmp_path: Path) -> None:
    matrix = _matrix()
    scannet = next(row for row in matrix["task_matrix"] if row["task"] == "ScanNet OVS")
    scannet["row_verdict"] = "eligible"
    with pytest.raises(MatrixError, match="row became eligible"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_spin_metadata_retrofit(tmp_path: Path) -> None:
    matrix = _matrix()
    spin = next(row for row in matrix["task_matrix"] if row["task"] == "SPIn-NeRF")
    spin["canonical_current_row"] = "reference_selected_unified_interface"
    with pytest.raises(MatrixError, match="filled by retrofit"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_compiler_decision_as_executed_result(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix["selected_rgb_assisted_compiler_precondition"]["current_row_created"] = True
    with pytest.raises(MatrixError, match="relabelled as a result"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_seed_panel_drift(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix["promotion_inputs"]["seed_policy"]["stochastic_panel"] = [0]
    with pytest.raises(MatrixError, match="seed panel drifted"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_evidence_hash_drift(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix["repository_evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(MatrixError, match="SHA-256 drift"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_missing_hard_gate(tmp_path: Path) -> None:
    matrix = _matrix()
    lerf2d = next(row for row in matrix["task_matrix"] if row["task"] == "LERF-2D")
    del lerf2d["gate_audit"]["cold_start_storage"]
    with pytest.raises(MatrixError, match="gate membership drifted"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_unearned_negative_transfer_claim(tmp_path: Path) -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["cross_task_gap"]["joint_contract_negative_transfer_status"] = "measured"
    with pytest.raises(MatrixError, match="negative transfer was asserted"):
        validate(_write(tmp_path, matrix))
