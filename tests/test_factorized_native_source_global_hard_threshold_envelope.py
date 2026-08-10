from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import factorized_native_source_global_hard_threshold_envelope as formal
from radio_gs.scripts import diagnose_factorized_native_contrast_v21_hard_threshold_envelope as runner


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def test_exact_grouped_curve_handles_equal_margin_as_one_threshold() -> None:
    margin = torch.tensor([0.3, 0.3, 0.1, -0.2])
    hard = torch.tensor([True, False, True, False])
    threshold, true_positive, predicted = runner.exact_grouped_pr_curve(margin, hard)
    assert torch.equal(threshold, torch.tensor([0.3, 0.1, -0.2]))
    assert torch.equal(true_positive, torch.tensor([1, 2, 2]))
    assert torch.equal(predicted, torch.tensor([2, 3, 4]))


def test_train_selection_uses_only_safe_exact_threshold_and_improves_recall_f1() -> None:
    margin = torch.tensor([0.9, 0.8, 0.1, -0.1, -0.2, -0.3, -0.4])
    hard = torch.tensor([True, False, True, True, False, True, False])
    selected = runner.select_train_threshold(margin, hard)
    assert selected["threshold"] == pytest.approx(-0.3)
    assert selected["validation_contribution"] is False
    assert selected["candidate"]["teacher_positive_recall"] == pytest.approx(1.0)
    assert selected["candidate"]["teacher_positive_f1"] > selected["identity"][
        "teacher_positive_f1"
    ]
    assert all(selected["candidate_checks"].values())


def test_unified_validation_oracle_finds_one_threshold_for_both_scenes(monkeypatch) -> None:
    monkeypatch.setattr(formal, "VALIDATION_SCENES", ("left", "right"))
    margins = {
        "left": torch.tensor([0.8, 0.1, -0.1, -0.4]),
        "right": torch.tensor([0.7, 0.2, -0.2, -0.5]),
    }
    hard = {
        "left": torch.tensor([True, False, True, False]),
        "right": torch.tensor([True, False, True, False]),
    }
    identities = {
        scene: runner.classification_metrics(margins[scene], hard[scene], threshold=0.0)
        for scene in ("left", "right")
    }
    oracle = runner.exact_unified_validation_oracle(margins, hard, identities)
    assert oracle["unified_feasible_threshold_exists"] is True
    assert oracle["unified_feasible_threshold_count"] > 0
    assert oracle["promotion_authorized"] is False
    assert oracle["parameter_export_authorized"] is False
    assert all(
        all(oracle["representative_metrics"][scene]["checks"].values())
        for scene in ("left", "right")
    )


def test_validation_gate_requires_all_four_frozen_checks() -> None:
    identity = {
        "teacher_positive_precision": 0.4,
        "teacher_positive_recall": 0.2,
        "teacher_positive_f1": 0.25,
    }
    candidate = {
        "teacher_positive_precision": 0.31,
        "teacher_positive_recall": 0.4,
        "teacher_positive_f1": 0.34,
    }
    assert all(formal.expected_checks(identity, candidate).values())
    candidate["teacher_positive_precision"] = 0.29
    checks = formal.expected_checks(identity, candidate)
    assert checks["precision_absolute_floor"] is True
    assert checks["precision_identity_retention"] is False


def test_authority_source_gate_is_first(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "authorized_source_only_hard_threshold_envelope",
        "implementation": _record("a"),
        "implementation_dependencies": {
            name: _record("b") for name in formal.IMPLEMENTATION_DEPENDENCIES
        },
        "envelope_contract_sha256": formal.ENVELOPE_CONTRACT_SHA256,
        "source_contrast_v21_result": _record("c"),
        "class_balanced_v2_result": _record("d"),
        "fit_text_bank": _record("e"),
        "canonical_negative_bank": _record("f"),
        "benchmark_exclusion_manifest": _record("1"),
        "envelope_output": str(tmp_path / "result.json"),
        "diagnostic_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": formal.source_access(),
    }
    order: list[str] = []
    monkeypatch.setattr(
        formal,
        "load_json_object",
        lambda *args, **kwargs: (authority, "2" * 64, tmp_path / "authority.json"),
    )

    def source_stop(*args, **kwargs):
        order.append("source")
        raise RuntimeError("source stop")

    def later_open(*args, **kwargs):
        order.append("later")
        raise AssertionError("opened a non-source artifact before source gate")

    monkeypatch.setattr(
        formal.source_formal, "validate_source_contrast_v21_result", source_stop
    )
    monkeypatch.setattr(formal, "validate_file_record", later_open)
    monkeypatch.setattr(formal, "_validate_v2_diagnostic", later_open)
    with pytest.raises(RuntimeError, match="source stop"):
        formal.validate_execution_authority(
            tmp_path / "authority.json", expected_sha256="2" * 64
        )
    assert order == ["source"]


def test_contract_keeps_oracle_non_promotional_and_target_query_closed() -> None:
    contract = formal.envelope_contract()
    assert contract["train_selected_candidate"]["validation_contribution"] is False
    assert contract["validation_oracle"]["promotion_authorized"] is False
    assert contract["validation_oracle"]["parameter_export_authorized"] is False
    assert contract["target_or_query_execution_authorized"] is False
    access = formal.source_access()
    assert access["target_heldout_opened"] is False
    assert access["benchmark_queries_opened"] is False
    assert access["target_metrics_computed"] is False
