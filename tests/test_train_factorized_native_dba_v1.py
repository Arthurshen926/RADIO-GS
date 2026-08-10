from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v1 as dba,
)


SHA = "a" * 64


def _scene_row(*, candidate: bool) -> dict[str, float | int]:
    return {
        "eligible_rows": 4096,
        "mean_all_view_cosine": 0.939 if candidate else 0.940,
        "p05_row_mean_all_view_cosine": 0.886 if candidate else 0.890,
        "mean_teacher_centered_residual_cosine": 0.445 if candidate else 0.450,
        "p05_teacher_centered_residual_cosine": 0.142 if candidate else 0.150,
        "student_to_teacher_spread_ratio": 0.84 if candidate else 0.85,
        "teacher_centered_pair_gram_mae": 0.308 if candidate else 0.300,
        "teacher_centered_pair_gram_correlation": 0.59 if candidate else 0.60,
        "absolute_visual_probe_response_mae": 0.038 if candidate else 0.030,
        "absolute_visual_probe_response_correlation": 0.64 if candidate else 0.65,
        "absolute_visual_probe_response_std_ratio": 0.91 if candidate else 0.90,
        "class_balanced_hard_bce": 0.690 if candidate else 0.700,
        "class_balanced_soft_teacher_brier": 0.290 if candidate else 0.300,
        "teacher_positive_rate": 0.002,
        "predicted_positive_rate": 0.006 if candidate else 0.005,
        "teacher_positive_precision": 0.30,
        "teacher_positive_recall": 0.23 if candidate else 0.20,
        "teacher_positive_f1": 0.17 if candidate else 0.15,
        "sampled_teacher_student_margin_rank_correlation": (
            0.396 if candidate else 0.400
        ),
    }


def _validation(*, candidate: bool) -> dict:
    row = _scene_row(candidate=candidate)
    result = {
        "per_scene": {scene: deepcopy(row) for scene in dba.VALIDATION_SCENES},
    }
    for key, value in row.items():
        if isinstance(value, float):
            result[f"macro_{key}"] = value
    return result


def _authority() -> dict:
    record = {"path": "/frozen", "sha256": SHA}
    return {
        "schema": dba.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": dba.SCHEMA_VERSION,
        "status": "authorized_source_only_dba_v1_exact4train_2validation",
        "implementation": record,
        "implementation_dependencies": {
            name: record for name in dba._DEPENDENCY_PATHS
        },
        "training_contract_sha256": dba.TRAINING_CONTRACT_SHA256,
        "source_contrast_v21_result": record,
        "fit_text_bank": record,
        "canonical_negative_bank": record,
        "benchmark_exclusion_manifest": record,
        "training_output": "/frozen/model.pt",
        "training_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": dba.source_access(),
    }


def test_contract_is_exact_64_step_complete_epoch_and_query_closed() -> None:
    contract = dba.training_contract()
    assert contract["optimizer"]["steps"] == 64
    assert contract["optimizer"]["batch_rows_per_scene_per_step"] == 64
    assert contract["optimizer"]["evaluation_steps"] == list(range(0, 65, 8))
    assert contract["objective"]["boundary_auxiliary_weight_on_complete_boundary_loss"] == 0.25
    assert contract["target_query_or_metric_execution_authorized"] is False
    access = dba.source_access()
    assert access["generic_target_blind_text_bank_opened"] is True
    assert access["target_heldout_opened"] is False
    assert access["benchmark_queries_opened"] is False
    assert access["benchmark_labels_opened"] is False


def test_authority_is_fail_closed_for_target_query_and_metric() -> None:
    dba.validate_execution_authority_header(_authority())
    for field in (
        "target_execution_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
    ):
        changed = deepcopy(_authority())
        changed[field] = True
        with pytest.raises(ValueError, match="header differs"):
            dba.validate_execution_authority_header(changed)


def test_promotion_gate_requires_boundary_gain_and_preserves_visual_metrics() -> None:
    baseline = _validation(candidate=False)
    candidate = _validation(candidate=True)
    selected = dba.attach_selection(candidate, baseline)
    assert selected["selection"]["eligible"] is True
    assert all(
        scene["passed"] for scene in selected["selection"]["per_scene"].values()
    )

    excessive = deepcopy(candidate)
    excessive["per_scene"][dba.VALIDATION_SCENES[0]][
        "predicted_positive_rate"
    ] = 0.021
    assert dba.attach_selection(excessive, baseline)["selection"]["eligible"] is False

    damaged = deepcopy(candidate)
    damaged["per_scene"][dba.VALIDATION_SCENES[1]][
        "mean_all_view_cosine"
    ] = 0.937
    assert dba.attach_selection(damaged, baseline)["selection"]["eligible"] is False


def test_step_selection_uses_f1_then_recall_then_bce_then_visual_then_earliest() -> None:
    history = []
    for step in range(0, 65, 8):
        validation = _validation(candidate=step > 0)
        validation["selection"] = {
            "eligible": step in {8, 16, 24},
            "macro_f1_improvement": {8: 0.02, 16: 0.03, 24: 0.03}.get(step, 0.0),
            "macro_recall_improvement": {8: 0.04, 16: 0.03, 24: 0.04}.get(step, 0.0),
            "macro_balanced_hard_bce_improvement": 0.01,
        }
        history.append({"step": step, "validation": validation})
    assert dba.select_step(history) == 24
    history[-1]["step"] = 63
    with pytest.raises(ValueError, match="schedule differs"):
        dba.select_step(history)


def test_boundary_metrics_are_class_balanced_and_use_fixed_rank_axis() -> None:
    torch.manual_seed(4)
    margin = torch.randn(400, dba.FIT_QUERY_ROWS) * 0.02
    teacher = torch.sigmoid(10.0 * (margin + 0.005 * torch.randn_like(margin)))
    observed = dba.boundary_metrics(margin, teacher)
    assert observed["pairs"] == margin.numel()
    assert observed["rank_samples"] == dba.RANK_SAMPLE_CAP
    assert 0.0 <= observed["teacher_positive_precision"] <= 1.0
    assert 0.0 <= observed["teacher_positive_recall"] <= 1.0
    assert observed["sampled_teacher_student_margin_rank_correlation"] > 0.9


def test_synthetic_dry_run_has_descriptor_gradient_and_no_target_access() -> None:
    result = dba.synthetic_dry_run()
    assert result["loss_finite"] is True
    assert result["gradient_finite_and_nonzero"] is True
    assert result["complete_rows_per_scene"] == 4096
    assert result["target_query_or_benchmark_opened"] is False
