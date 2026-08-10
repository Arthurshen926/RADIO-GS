from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.querying.anchor_preserving_transport_checkpoint import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_V21,
    load_anchor_preserving_prompt_head,
    load_anchor_preserving_prompt_head_v21,
)
from radio_gs.querying.registered_evidence_to_unary import RegisteredEvidenceToUnaryV2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, promoted: bool = True) -> tuple[Path, dict]:
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "schema": "radio_gs.anchor_preserving_prompt_transport.clean_gate_result.v1",
                "promotion_gate_passed": promoted,
                "decision": (
                    "eligible_for_one_preregistered_target_sentinel"
                    if promoted
                    else "stop_before_nvos_or_spin_target_metrics"
                ),
            }
        )
    )
    lineage = {
        "transport_contract_sha256": "transport",
        "trainer_sha256": "trainer",
        "preregistration_sha256": "prereg",
        "fit_authority_sha256": "fit",
        "confirmation_authority_sha256": "confirmation",
    }
    head = RegisteredEvidenceToUnaryV2(hidden_dim=32, max_delta_logit=4.0)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "state_dict": head.state_dict(),
            "best_epoch": 1,
            "architecture": {
                "hidden_dim": 32,
                "max_delta_logit": 4.0,
                "fully_observed_tolerance": 1e-5,
            },
            "lineage": lineage,
            "result_path": str(result.resolve()),
            "result_sha256": _sha(result),
        },
        checkpoint,
    )
    return checkpoint, lineage


def _load(checkpoint: Path, lineage: dict):
    return load_anchor_preserving_prompt_head(
        checkpoint,
        expected_checkpoint_sha256=_sha(checkpoint),
        expected_transport_contract_sha256=lineage["transport_contract_sha256"],
        expected_trainer_sha256=lineage["trainer_sha256"],
        expected_preregistration_sha256=lineage["preregistration_sha256"],
        expected_fit_authority_sha256=lineage["fit_authority_sha256"],
        expected_confirmation_authority_sha256=lineage[
            "confirmation_authority_sha256"
        ],
    )


def test_loader_accepts_only_bound_promoted_v2_checkpoint(tmp_path: Path) -> None:
    checkpoint, lineage = _write_fixture(tmp_path)
    head = _load(checkpoint, lineage)
    assert isinstance(head, RegisteredEvidenceToUnaryV2)
    assert head.training is False


def test_loader_rejects_lineage_mismatch(tmp_path: Path) -> None:
    checkpoint, lineage = _write_fixture(tmp_path)
    lineage["trainer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="lineage"):
        _load(checkpoint, lineage)


def test_loader_rejects_unpromoted_source_result(tmp_path: Path) -> None:
    checkpoint, lineage = _write_fixture(tmp_path, promoted=False)
    with pytest.raises(ValueError, match="not promoted"):
        _load(checkpoint, lineage)


def test_v21_loader_rejects_v1_schema_before_state_dict(tmp_path: Path) -> None:
    checkpoint, _lineage = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="V2.1 checkpoint schema"):
        load_anchor_preserving_prompt_head_v21(
            checkpoint,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_transport_contract_sha256="transport",
            expected_trainer_sha256="trainer",
            expected_base_asset_loader_sha256="base",
            expected_preregistration_sha256="prereg",
            expected_fit_authority_sha256="fit",
            expected_confirmation_authority_sha256="confirmation",
            require_promoted_result=False,
        )


def test_v21_schema_constant_is_distinct() -> None:
    assert CHECKPOINT_SCHEMA_V21 != CHECKPOINT_SCHEMA
