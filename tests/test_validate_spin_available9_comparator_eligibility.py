import copy
import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_spin_available9_comparator_eligibility import (
    AuditError,
    DEFAULT_AUDIT,
    validate,
)


def _audit() -> dict:
    return json.loads(DEFAULT_AUDIT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_audit_reproduces_no_eligible_target() -> None:
    receipt = validate()
    assert receipt["verdict"] == "no_eligible_existing_comparator"
    assert receipt["numeric_sota_target"] is None
    assert receipt["candidate_count"] == 7
    assert receipt["evidence_count"] == 6


def test_audit_rejects_context_number_as_numeric_target(tmp_path: Path) -> None:
    audit = _audit()
    audit["resolution"]["numeric_sota_target"] = 93.7200449592385
    with pytest.raises(AuditError, match="numeric target must remain absent"):
        validate(_write(tmp_path, audit))


def test_audit_rejects_candidate_without_blocking_failure(tmp_path: Path) -> None:
    audit = _audit()
    audit["candidate_audit"][0]["blocking_failures"] = []
    with pytest.raises(AuditError, match="no blocking failure"):
        validate(_write(tmp_path, audit))


def test_audit_rejects_relabelled_radio_gs_row(tmp_path: Path) -> None:
    audit = copy.deepcopy(_audit())
    audit["candidate_audit"][-1]["eligible"] = True
    with pytest.raises(AuditError, match="eligibility is not fail closed"):
        validate(_write(tmp_path, audit))
