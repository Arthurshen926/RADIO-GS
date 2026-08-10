from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.scripts import materialize_lerf_o0_conditional_missing_core_completion_v2 as builder
from radio_gs.scripts.audit_source_same_axis_o0_missing_core_mechanism import FEATURE_NAMES


def _record(name: str) -> dict[str, str]:
    return {"path": f"/{name}", "sha256": name[0] * 64}


def _access() -> dict[str, bool]:
    return {
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
        "benchmark_queries_opened": False,
        "target_metrics_computed": False,
    }


def _model(threshold: float, authority: dict[str, str]) -> dict:
    fold = {
        "location": torch.zeros(6),
        "scale": torch.ones(6),
        "positive_weights": torch.ones(6),
        "bias": torch.tensor(0.0),
    }
    return {
        "schema": "radio_gs.source_multiscene_monotone_missing_core_selector.v2",
        "schema_version": 2,
        "feature_names": [
            "unit_o0_score_positive", "appearance_concentration_positive",
            "boundary_concentration_positive", "negative_log1p_core_spatial_rms_radius",
            "negative_query_selected_scale_index",
            "negative_log1p_full_scalar_source_robust_ood_linf",
        ],
        "source_unit_feature_indices": [0, 14, 15, 17, 9, 18],
        "fold_models": [fold, fold, fold],
        "threshold_inclusive": threshold,
        "target_probability": "minimum_probability_across_three_fold_models",
        "execution_authority": authority,
        "training_provenance": {
            "source_scene_count": 2,
            "source_scene_ids_sha256": "a" * 64,
            "scene_identifier_used_for_balancing_and_folds_only": True,
            "query_identifier_used_as_feature": False,
            "scene_identifier_used_as_feature": False,
        },
    }


def _fixtures(threshold: float = 0.731) -> tuple[dict, dict, dict]:
    records = {name: _record(str(index + 1)) for index, name in enumerate(builder.SOURCE_INPUT_NAMES)}
    authority_record = records["multisource_selector_authority"]
    model_record = records["multisource_selector_model"]
    report_record = records["multisource_selector_report"]
    source_rows = [
        {
            "scene_id": "scene0001_00", "role": "mechanism_train",
            "authority": _record("a"), "report": _record("b"), "unit_table": _record("c"),
        },
        {
            "scene_id": "scene0002_00", "role": "external_source_train",
            "authority": _record("d"), "report": _record("e"), "unit_table": _record("f"),
        },
    ]
    source_authority = {
        "schema": builder.SOURCE_AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": "sealed_after_scene0001_scene0002_source_tables_before_multiscene_fit",
        "source_scenes": ["scene0001_00", "scene0002_00"],
        "source_inputs": source_rows,
        "outputs": {"model": model_record["path"], "report": report_record["path"]},
        "source_access": _access(),
        "source_validation_execution_authorized": False,
        "benchmark_execution_authorized": False,
    }
    source_report = {
        "schema": builder.SOURCE_REPORT_SCHEMA,
        "schema_version": 2,
        "status": "scene0001_scene0002_multiscene_selector_v2_gate_passed",
        "execution_authority": authority_record,
        "model": model_record,
        "gate": {"passed": True},
        "metrics": {"threshold_selection": {"threshold_inclusive": threshold}},
        "source_access": _access(),
        "source_validation_execution_performed": False,
        "target_execution_performed": False,
        "benchmark_execution_authorized": False,
    }
    external_execution = _record("9")
    scene3_authority = {
        "schema": builder.SCENE0003_AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": "scene0003_frozen_multiscene_selector_external_gate_passed",
        "scene_id": "scene0003_00",
        "execution_authority": external_execution,
        "frozen_selector_authority": authority_record,
        "frozen_selector_model": model_record,
        "frozen_selector_report": report_record,
        "unit_table": records["scene0003_pass_unit_table"],
        "validation_report": records["scene0003_pass_report"],
        "validation_outcomes": {"selected": 1},
        "all_formal_gates_passed": True,
        "source_access": _access(),
        "benchmark_execution_authorized": False,
    }
    checks = {"passed": True, "another_gate": True}
    scene3_report = {
        "schema": builder.SCENE0003_REPORT_SCHEMA,
        "schema_version": 2,
        "status": "scene0003_frozen_multiscene_selector_external_gate_passed",
        "execution_authority": external_execution,
        "frozen_selector_model": model_record,
        "unit_table": records["scene0003_pass_unit_table"],
        "frozen_threshold_inclusive": threshold,
        "sample_gate": {"outcomes": {"passed": True}},
        "selector_gate": {"outcomes": {"selected": 1}, "checks": checks},
        "source_access": _access(),
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    payloads = {
        authority_record["path"]: source_authority,
        report_record["path"]: source_report,
        model_record["path"]: _model(threshold, authority_record),
        records["scene0003_pass_authority"]["path"]: scene3_authority,
        records["scene0003_pass_report"]["path"]: scene3_report,
    }
    return records, payloads, external_execution


def test_source_gate_reads_exact_threshold_from_v2_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, payloads, _ = _fixtures(0.731)
    monkeypatch.setattr(builder, "validate_file_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_load_json", lambda record, **kwargs: payloads[record["path"]])
    monkeypatch.setattr(builder, "_load_torch", lambda record, **kwargs: payloads[record["path"]])
    unit = {
        "execution_authority": payloads[records["scene0003_pass_authority"]["path"]]["execution_authority"],
        "features": torch.zeros(1, 37), "hard_labels": torch.tensor([True]),
        "signed_utility": torch.tensor([1.0]), "selector_probability": torch.tensor([0.8]),
        "selected": torch.tensor([True]),
    }
    monkeypatch.setattr(
        builder,
        "_validate_source_unit_table",
        lambda record, **kwargs: unit if kwargs["expected_scene_id"] == "scene0003_00" else {"execution_authority": kwargs["expected_execution_authority"]},
    )
    monkeypatch.setattr(builder, "target_consensus_probability", lambda *args: torch.tensor([0.8]))
    monkeypatch.setattr(builder, "evaluate_selector_gate", lambda **kwargs: ({"selected": 1}, {"passed": True, "another_gate": True}))
    assert not hasattr(builder, "FROZEN_THRESHOLD")
    assert builder.validate_source_gate_v2(records) == 0.731


def test_source_gate_rejects_scene3_report_threshold_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, payloads, _ = _fixtures(0.731)
    payloads[records["scene0003_pass_report"]["path"]]["frozen_threshold_inclusive"] = 0.7
    monkeypatch.setattr(builder, "validate_file_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_load_json", lambda record, **kwargs: payloads[record["path"]])
    monkeypatch.setattr(builder, "_load_torch", lambda record, **kwargs: payloads[record["path"]])
    unit = {
        "execution_authority": payloads[records["scene0003_pass_authority"]["path"]]["execution_authority"],
        "features": torch.zeros(1, 37), "hard_labels": torch.tensor([True]),
        "signed_utility": torch.tensor([1.0]), "selector_probability": torch.tensor([0.8]),
        "selected": torch.tensor([True]),
    }
    monkeypatch.setattr(
        builder,
        "_validate_source_unit_table",
        lambda record, **kwargs: unit if kwargs["expected_scene_id"] == "scene0003_00" else {"execution_authority": kwargs["expected_execution_authority"]},
    )
    monkeypatch.setattr(builder, "target_consensus_probability", lambda *args: torch.tensor([0.8]))
    monkeypatch.setattr(builder, "evaluate_selector_gate", lambda **kwargs: ({"selected": 1}, {"passed": True, "another_gate": True}))
    with pytest.raises(ValueError, match="selector gate differs"):
        builder.validate_source_gate_v2(records)


def test_source_access_missing_false_key_does_not_pass() -> None:
    access = _access()
    del access["benchmark_queries_opened"]
    assert builder._source_access_is_target_blind(access) is False
    assert len(FEATURE_NAMES) == 37
