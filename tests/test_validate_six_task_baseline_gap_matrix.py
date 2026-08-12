import copy
import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_six_task_baseline_gap_matrix import (
    DEFAULT_MATRIX,
    MatrixError,
    validate,
)


def _matrix() -> dict:
    return json.loads(DEFAULT_MATRIX.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_matrix_reproduces_empty_joint_baseline() -> None:
    receipt = validate()
    assert receipt["verdict"] == "no_eligible_joint_development_baseline"
    assert receipt["task_count"] == 6
    assert receipt["eligible_task_row_count"] == 0
    assert receipt["numerically_passing_context_rows"] == ["ScanNet OVS"]


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


def test_matrix_rejects_invented_spin_target(tmp_path: Path) -> None:
    matrix = _matrix()
    spin = next(row for row in matrix["task_matrix"] if row["task"] == "SPIn-NeRF")
    spin["numeric_sota_target_percent"] = 93.7200449592385
    with pytest.raises(MatrixError, match="invented a SPIn target"):
        validate(_write(tmp_path, matrix))


def test_matrix_rejects_unearned_negative_transfer_claim(tmp_path: Path) -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["cross_task_gap"]["multi_task_negative_transfer_status"] = "measured"
    with pytest.raises(MatrixError, match="negative transfer was asserted"):
        validate(_write(tmp_path, matrix))
