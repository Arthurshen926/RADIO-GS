from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts import (
    finalize_surface_readout_weight_interpolation_audit_failure as module,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pool = tmp_path / "pool"
    family = pool / "family"
    family.mkdir(parents=True)
    alias = tmp_path / "output_alias"
    alias.symlink_to(pool, target_is_directory=True)
    monkeypatch.setattr(module, "FORMAL_POOL_ROOT", pool)
    executor = tmp_path / "historical_executor.py"
    executor.write_text("# historical executor\n", encoding="utf-8")
    executor_sha = _sha(executor)
    receipt_lexical = alias / "family" / "audit_opening_receipt.json"
    confirmation_lexical = alias / "family" / "audit90_confirmation.json"
    terminal_lexical = alias / "family" / "audit90_failure_terminal.json"
    receipt = {
        "schema_version": 1,
        "artifact_type": module.OPENING_ARTIFACT_TYPE,
        "status": "one_shot_opening_authorization_committed",
        "opening_count": 1,
        "audit_bank_loads_authorized": 1,
        "selection_validation_completed": True,
        "query_free_recomputation_completed": True,
        "selection": {"path": "/selection", "sha256": "1" * 64},
        "diagnostic": {"path": "/diagnostic", "sha256": "2" * 64},
        "declared_audit_bank": {
            "path": "/must/not/be/resolved/audit.pt",
            "sha256": "3" * 64,
            "manifest_path": "/must/not/be/resolved/audit.json",
            "manifest_sha256": "4" * 64,
            "split": "audit",
            "expected_query_count": 90,
        },
        "intended_confirmation_output": str(confirmation_lexical),
        "implementation": {"path": str(executor), "sha256": executor_sha},
        "implementation_closure": [
            {
                "role": "audit_confirmation_executor",
                "path": str(executor),
                "sha256": executor_sha,
            }
        ],
    }
    receipt_lexical.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        opening_receipt=receipt_lexical,
        opening_receipt_sha256=_sha(receipt_lexical),
        executor_sha256=executor_sha,
        expected_confirmation=confirmation_lexical,
        output=terminal_lexical,
    )
    return args, pool, family


def test_failure_finalizer_closes_family_without_opening_declared_audit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, pool, family = _fixture(tmp_path, monkeypatch)

    payload = module.finalize(args)

    terminal = family / "audit90_failure_terminal.json"
    assert terminal.is_file()
    assert payload["decision"] == "confirmation_failed_family_closed_no_retry"
    assert payload["main_result_eligible"] is False
    assert payload["post_audit_retuning_forbidden"] is True
    assert payload["failure_stage"] == "post_audit_preterminal_receipt_binding"
    assert payload["opening_receipt"]["resolved_path"].startswith(str(pool))
    assert payload["confirmation"]["exists"] is False
    assert payload["audit"]["artifact_accessed_by_finalizer"] is False
    assert payload["audit"]["manifest_accessed_by_finalizer"] is False
    assert json.loads(terminal.read_text(encoding="utf-8")) == payload

    with pytest.raises(ValueError, match="failure terminal output already exists"):
        module.finalize(args)


def test_failure_finalizer_rejects_an_existing_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _, _ = _fixture(tmp_path, monkeypatch)
    Path(args.expected_confirmation).write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="expected confirmation already exists"):
        module.finalize(args)
