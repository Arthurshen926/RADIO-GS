from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import factorized_native_source_global_margin_calibration as formal
from radio_gs.scripts import calibrate_factorized_native_contrast_v21_global_margin as runner


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def test_soft_teacher_probability_is_mean_of_per_view_binary_softmax() -> None:
    teacher = torch.zeros(1, 2, 1536)
    positive = torch.zeros(1, 1536)
    negative = torch.zeros(4, 1536)
    teacher[0, 0, 0] = 1.0
    teacher[0, 1, 1] = 1.0
    positive[0, 0] = 1.0
    negative[:, 1] = 1.0
    mask = torch.ones(1, 2, dtype=torch.bool)
    observed = runner.soft_teacher_probability(teacher, mask, positive, negative)
    expected = 0.5 * (torch.sigmoid(torch.tensor(10.0)) + torch.sigmoid(torch.tensor(-10.0)))
    assert torch.allclose(observed, expected.reshape(1, 1), rtol=0.0, atol=1e-7)


def test_deterministic_newton_recovers_unique_positive_global_parameters(monkeypatch) -> None:
    monkeypatch.setattr(formal, "TRAIN_SCENES", ("source",))
    monkeypatch.setattr(formal, "REGIONS_PER_SCENE", 16)
    monkeypatch.setattr(formal, "FIT_QUERY_ROWS", 8)
    margin = torch.linspace(-0.4, 0.4, 128)
    target = torch.sigmoid(3.5 * margin - 0.4)
    parameters, audit = runner.fit_global_calibrator(margin, target)
    assert audit["converged"] is True
    assert parameters["a"] == pytest.approx(3.5, abs=1e-5)
    assert parameters["b"] == pytest.approx(-0.4, abs=1e-5)
    assert audit["final_soft_binary_cross_entropy"] < audit["initial_soft_binary_cross_entropy"]


def test_validation_gate_requires_all_three_calibration_improvements_and_safe_boundary() -> None:
    identity = {
        "brier": 0.10,
        "soft_binary_cross_entropy": 0.40,
        "mean_absolute_error": 0.20,
        "rank_correlation": 0.6,
        "teacher_positive_precision": 0.8,
        "teacher_positive_recall": 0.8,
    }
    candidate = {
        "brier": 0.08,
        "soft_binary_cross_entropy": 0.35,
        "mean_absolute_error": 0.17,
        "rank_correlation": 0.6,
        "teacher_positive_precision": 0.7,
        "teacher_positive_recall": 0.7,
    }
    assert all(formal.expected_validation_checks(identity, candidate).values())
    candidate["teacher_positive_recall"] = 0.5
    assert formal.expected_validation_checks(identity, candidate)[
        "teacher_positive_recall_not_catastrophic"
    ] is False


def test_authority_validates_source_before_opening_text_records(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_global_margin_fit_4train_2validation",
        "implementation": _record("a"),
        "implementation_dependencies": {
            name: _record("b") for name in formal.IMPLEMENTATION_DEPENDENCIES
        },
        "calibration_contract_sha256": formal.CALIBRATION_CONTRACT_SHA256,
        "source_contrast_v21_result": _record("c"),
        "fit_text_bank": _record("d"),
        "canonical_negative_bank": _record("e"),
        "benchmark_exclusion_manifest": _record("f"),
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
        lambda *args, **kwargs: (authority, "1" * 64, tmp_path / "authority.json"),
    )

    def source_stop(*args, **kwargs):
        order.append("source")
        raise RuntimeError("source stop")

    def later_open(*args, **kwargs):
        order.append("later")
        raise AssertionError("text or implementation opened before source gate")

    monkeypatch.setattr(
        formal.source_formal, "validate_source_contrast_v21_result", source_stop
    )
    monkeypatch.setattr(formal, "validate_file_record", later_open)
    with pytest.raises(RuntimeError, match="source stop"):
        formal.validate_execution_authority(
            tmp_path / "authority.json", expected_sha256="1" * 64
        )
    assert order == ["source"]


def test_contract_closes_every_target_query_metric_path() -> None:
    access = formal.source_access()
    assert access["target_heldout_opened"] is False
    assert access["benchmark_queries_opened"] is False
    assert access["target_descriptor_opened"] is False
    assert access["target_relevance_opened"] is False
    assert access["target_metrics_computed"] is False
    assert formal.calibration_contract()["margin"]["scene_parameters"] is False
    assert formal.calibration_contract()["margin"]["query_parameters"] is False
