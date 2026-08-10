from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import factorized_native_source_global_margin_calibration_v2 as formal
from radio_gs.scripts import calibrate_factorized_native_contrast_v21_global_margin_v2 as runner


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def test_train_soft_positive_and_negative_mass_each_receive_half() -> None:
    target = torch.tensor([0.01, 0.20, 0.70, 0.95], dtype=torch.float64)
    weights = runner.train_mass_weights(target)
    positive_rate = float(target.mean())
    assert weights["positive_weight"] == pytest.approx(0.5 / positive_rate)
    assert weights["negative_weight"] == pytest.approx(0.5 / (1.0 - positive_rate))
    assert weights["weighted_positive_mass_fraction"] == pytest.approx(0.5)
    assert weights["weighted_negative_mass_fraction"] == pytest.approx(0.5)
    assert weights["derived_from_train_only"] is True


def test_class_balanced_newton_recovers_unique_global_positive_slope(monkeypatch) -> None:
    monkeypatch.setattr(formal, "TRAIN_SCENES", ("source",))
    monkeypatch.setattr(formal, "REGIONS_PER_SCENE", 16)
    monkeypatch.setattr(formal, "FIT_QUERY_ROWS", 8)
    margin = torch.linspace(-0.5, 0.5, 128, dtype=torch.float64)
    target = torch.sigmoid(3.5 * margin)
    weights = runner.train_mass_weights(target)
    parameters, audit = runner.fit_global_calibrator(margin, target, weights)
    assert weights["teacher_soft_positive_rate"] == pytest.approx(0.5, abs=1e-12)
    assert parameters["a"] == pytest.approx(3.5, abs=1e-5)
    assert parameters["b"] == pytest.approx(0.0, abs=1e-6)
    assert audit["converged"] is True
    assert (
        audit["final_balanced_soft_binary_cross_entropy"]
        < audit["initial_balanced_soft_binary_cross_entropy"]
    )


def test_balanced_metrics_use_explicit_frozen_train_weights() -> None:
    margin = torch.tensor([-0.2, -0.1, 0.1, 0.2])
    target = torch.tensor([0.1, 0.4, 0.7, 0.9])
    metrics = runner.calibration_metrics(
        margin,
        target,
        a=10.0,
        b=0.0,
        positive_weight=4.0,
        negative_weight=0.25,
        fixed_rank_correlation=0.75,
    )
    probability = torch.sigmoid(10.0 * margin)
    expected_brier = (
        4.0 * target * (1.0 - probability).square()
        + 0.25 * (1.0 - target) * probability.square()
    ).mean()
    assert metrics["balanced_brier"] == pytest.approx(float(expected_brier))
    assert metrics["rank_correlation"] == 0.75


def test_v2_validation_gate_is_fixed_and_requires_every_boundary_check() -> None:
    identity = {
        "balanced_soft_binary_cross_entropy": 0.50,
        "balanced_brier": 0.20,
        "teacher_positive_f1": 0.30,
        "teacher_positive_recall": 0.20,
        "teacher_positive_precision": 0.40,
        "rank_correlation": 0.70,
    }
    candidate = {
        "balanced_soft_binary_cross_entropy": 0.48,
        "balanced_brier": 0.18,
        "teacher_positive_f1": 0.35,
        "teacher_positive_recall": 0.30,
        "teacher_positive_precision": 0.31,
        "rank_correlation": 0.70,
    }
    assert all(formal.expected_validation_checks(identity, candidate).values())
    candidate["teacher_positive_precision"] = 0.29
    checks = formal.expected_validation_checks(identity, candidate)
    assert checks["precision_identity_retention"] is False
    assert checks["precision_absolute_floor"] is True
    candidate["teacher_positive_precision"] = 0.31
    candidate["teacher_positive_recall"] = 0.20
    assert formal.expected_validation_checks(identity, candidate)[
        "hard_recall_strictly_improved"
    ] is False


def test_authority_source_gate_precedes_diagnostic_text_and_code(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "authorized_source_only_class_balanced_global_margin_fit",
        "implementation": _record("a"),
        "implementation_dependencies": {
            name: _record("b") for name in formal.IMPLEMENTATION_DEPENDENCIES
        },
        "calibration_contract_sha256": formal.CALIBRATION_CONTRACT_SHA256,
        "source_contrast_v21_result": _record("c"),
        "unweighted_v1_result": _record("d"),
        "fit_text_bank": _record("e"),
        "canonical_negative_bank": _record("f"),
        "benchmark_exclusion_manifest": _record("1"),
        "calibration_output": str(tmp_path / "result.json"),
        "calibration_authorized": True,
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
        raise AssertionError("non-source record opened before source gate")

    monkeypatch.setattr(
        formal.source_formal, "validate_source_contrast_v21_result", source_stop
    )
    monkeypatch.setattr(formal, "validate_file_record", later_open)
    monkeypatch.setattr(formal, "_validate_unweighted_v1", later_open)
    with pytest.raises(RuntimeError, match="source stop"):
        formal.validate_execution_authority(
            tmp_path / "authority.json", expected_sha256="2" * 64
        )
    assert order == ["source"]


def test_v2_contract_forbids_target_query_and_validation_weight_leakage() -> None:
    contract = formal.calibration_contract()
    assert contract["fit"]["validation_contribution"] is False
    assert contract["fit"]["validation_reuses_fixed_train_weights"] is True
    assert contract["fit"]["scene_parameters"] is False
    assert contract["fit"]["query_parameters"] is False
    assert contract["target_or_query_execution_authorized"] is False
    access = formal.source_access()
    assert access["target_heldout_opened"] is False
    assert access["benchmark_queries_opened"] is False
    assert access["target_metrics_computed"] is False
