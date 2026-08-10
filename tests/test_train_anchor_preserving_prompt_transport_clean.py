from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from radio_gs.querying.registered_evidence_to_unary import RegisteredEvidenceToUnaryV2
from radio_gs.scripts import train_anchor_preserving_prompt_transport_clean as trainer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistered_authority_mismatch_fails_before_scene_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = tmp_path / "fit.json"
    confirmation = tmp_path / "confirmation.json"
    fit.write_text("{}")
    confirmation.write_text("{}")
    prereg = tmp_path / "prereg.json"
    payload = {
        "schema": "radio_gs.anchor_preserving_prompt_transport.clean_gate_preregistration.v1",
        "fit_scene": "scene0001_00",
        "confirmation_scene": "scene0002_00",
        "epochs": 1,
        "authority": {
            "fit_asset_authority": {
                "path": str((tmp_path / "different.json").resolve()),
                "sha256": _sha(fit),
            },
            "confirmation_asset_authority": {
                "path": str(confirmation.resolve()),
                "sha256": _sha(confirmation),
            },
            "implementation": {
                "path": str(Path(trainer.__file__).resolve()),
                "sha256": _sha(Path(trainer.__file__)),
            },
            "transport_contract": {
                "path": str(Path(trainer.transport_module.__file__).resolve()),
                "sha256": _sha(Path(trainer.transport_module.__file__)),
            },
        },
    }
    prereg.write_text(json.dumps(payload))
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("_load_scene must not run")

    monkeypatch.setattr(trainer, "_load_scene", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trainer",
            "--fit-authority",
            str(fit),
            "--confirmation-authority",
            str(confirmation),
            "--preregistration",
            str(prereg),
            "--expected-preregistration-sha256",
            _sha(prereg),
            "--output",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
            "--epochs",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="fit asset authority path"):
        trainer.main()
    assert called is False


def _synthetic_result() -> dict:
    modes = {}
    for mode in ("all", "full_mask", "scribble"):
        modes[mode] = {
            "analytic": {
                "average_precision": 0.50,
                "iou_at_0_5": 0.30,
                "precision_at_0_5": 0.40,
                "area_ratio": 3.0,
            },
            "candidate": {
                "average_precision": 0.55,
                "iou_at_0_5": 0.35,
                "precision_at_0_5": 0.41,
                "area_ratio": 1.5,
            },
        }
    return {
        "macro": modes,
        "records": [{"delta": {"average_precision": -0.004}}],
    }


def test_gate_uses_relative_log_area_improvement_not_old_absolute_interval() -> None:
    gate = trainer._gate(_synthetic_result())
    assert gate["passed"] is True
    assert 1.5 > 1.25
    assert all(
        value["area_log_error_after"] < value["area_log_error_before"]
        for value in gate["per_mode"].values()
    )


def test_checkpoint_payload_has_distinct_schema_and_complete_lineage(tmp_path: Path) -> None:
    head = RegisteredEvidenceToUnaryV2(hidden_dim=32, max_delta_logit=4.0)
    payload = trainer._checkpoint_payload(
        state_dict=head.state_dict(),
        best_epoch=7,
        head=head,
        result_path=tmp_path / "result.json",
        result_sha256="result",
        transport_sha256="transport",
        trainer_sha256="trainer",
        preregistration_sha256="prereg",
        fit_authority_sha256="fit",
        confirmation_authority_sha256="confirmation",
    )
    assert payload["schema"] == "radio_gs.anchor_preserving_prompt_transport.checkpoint.v1"
    assert payload["architecture"] == {
        "hidden_dim": 32,
        "max_delta_logit": 4.0,
        "fully_observed_tolerance": 1e-5,
    }
    assert set(payload["lineage"]) == {
        "transport_contract_sha256",
        "trainer_sha256",
        "preregistration_sha256",
        "fit_authority_sha256",
        "confirmation_authority_sha256",
    }
