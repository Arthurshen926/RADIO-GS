from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.models import source_contrast_preservation as contrast
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as v1,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as v2,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


SHA = "a" * 64


def _teacher(rows: int = 24, width: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    common = F.normalize(torch.randn(width, generator=generator), dim=0) * 3.0
    residual = F.normalize(
        torch.randn(rows, width, generator=generator), dim=-1
    )
    teacher = F.normalize(common[None] + residual, dim=-1)
    center = contrast.fit_equal_scene_teacher_center((teacher[: rows // 2], teacher[rows // 2 :]))
    return teacher, center


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _authority() -> dict:
    return {
        "schema": v2.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_contrast_v2_exact4train_2validation",
        **{name: _record() for name in v2._CODE_RECORD_FIELDS},
        "base_v1_execution_authority": _record(),
        "training_contract_sha256": canonical_json_sha256(v2.training_contract()),
        "authorized_arm": "direction_only",
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": v2.source_access(),
    }


def _scene_metrics(
    *,
    raw: float,
    raw_p05: float,
    residual: float,
    residual_p05: float,
    spread: float,
    gram_mae: float,
    gram_correlation: float,
    probe_mae: float = 0.10,
    probe_correlation: float = 0.40,
    probe_std_ratio: float = 1.0,
) -> dict[str, float | int]:
    return {
        "eligible_rows": 8,
        "mean_all_view_cosine": raw,
        "p05_row_mean_all_view_cosine": raw_p05,
        "mean_teacher_centered_residual_cosine": residual,
        "p05_teacher_centered_residual_cosine": residual_p05,
        "student_to_teacher_spread_ratio": spread,
        "teacher_centered_pair_gram_mae": gram_mae,
        "teacher_centered_pair_gram_correlation": gram_correlation,
        "absolute_visual_probe_response_mae": probe_mae,
        "absolute_visual_probe_response_correlation": probe_correlation,
        "absolute_visual_probe_response_std_ratio": probe_std_ratio,
    }


def _validation(scene: dict[str, float | int]) -> dict:
    per_scene = {name: deepcopy(scene) for name in v2.VALIDATION_SCENES}
    return {
        "macro_mean_all_view_cosine": float(scene["mean_all_view_cosine"]),
        "macro_p05_row_mean_all_view_cosine": float(
            scene["p05_row_mean_all_view_cosine"]
        ),
        "macro_mean_teacher_centered_residual_cosine": float(
            scene["mean_teacher_centered_residual_cosine"]
        ),
        "macro_p05_teacher_centered_residual_cosine": float(
            scene["p05_teacher_centered_residual_cosine"]
        ),
        "macro_student_to_teacher_spread_ratio": float(
            scene["student_to_teacher_spread_ratio"]
        ),
        "macro_teacher_centered_pair_gram_mae": float(
            scene["teacher_centered_pair_gram_mae"]
        ),
        "macro_teacher_centered_pair_gram_correlation": float(
            scene["teacher_centered_pair_gram_correlation"]
        ),
        "macro_absolute_visual_probe_response_mae": float(
            scene["absolute_visual_probe_response_mae"]
        ),
        "macro_absolute_visual_probe_response_correlation": float(
            scene["absolute_visual_probe_response_correlation"]
        ),
        "macro_absolute_visual_probe_response_std_ratio": float(
            scene["absolute_visual_probe_response_std_ratio"]
        ),
        "per_scene": per_scene,
    }


def test_common_component_collapse_has_high_raw_cosine_but_fails_geometry() -> None:
    teacher, center = _teacher()
    views = teacher[:, None]
    mask = torch.ones(teacher.shape[0], 1, dtype=torch.bool)
    collapsed = F.normalize(center, dim=0)[None].repeat(teacher.shape[0], 1)

    bad = contrast.contrast_metrics(collapsed, views, mask, center)
    good = contrast.contrast_metrics(teacher, views, mask, center)

    assert bad["mean_all_view_cosine"] > 0.93
    assert bad["student_to_teacher_spread_ratio"] < 1e-6
    assert bad["teacher_centered_pair_gram_correlation"] == pytest.approx(0.0)
    assert good["mean_teacher_centered_residual_cosine"] > 0.999
    assert good["student_to_teacher_spread_ratio"] == pytest.approx(1.0)
    assert good["teacher_centered_pair_gram_mae"] < 1e-6
    assert good["teacher_centered_pair_gram_correlation"] > 0.999
    assert good["absolute_visual_probe_response_mae"] < 1e-6
    assert good["absolute_visual_probe_response_correlation"] > 0.999
    assert bad["absolute_visual_probe_response_std_ratio"] < 0.75


def test_objective_is_direction_only_scale_invariant_and_has_finite_gradient() -> None:
    teacher, center = _teacher(rows=12)
    views = teacher[:, None].repeat(1, 2, 1)
    mask = torch.ones(12, 2, dtype=torch.bool)
    student = (teacher + 0.03 * torch.roll(teacher, 1, 0)).requires_grad_(True)
    scaled = student.detach() * torch.linspace(0.3, 3.0, 12)[:, None]

    loss, parts = contrast.contrast_preserving_objective(
        student, views, mask, center
    )
    scaled_loss, _ = contrast.contrast_preserving_objective(
        scaled, views, mask, center
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(float(scaled_loss), abs=1e-6)
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())
    assert set(parts) == {
        "raw_all_view_cosine_loss",
        "teacher_centered_residual_cosine_loss",
        "teacher_centered_gram_loss",
        "absolute_visual_probe_calibration_loss",
        "variance_noncollapse_loss",
        "student_spread",
        "teacher_spread",
        "student_to_teacher_spread_ratio",
    }


def test_objective_penalizes_common_center_constant_predictor() -> None:
    teacher, center = _teacher()
    views = teacher[:, None]
    mask = torch.ones(teacher.shape[0], 1, dtype=torch.bool)
    collapsed = F.normalize(center, dim=0)[None].repeat(teacher.shape[0], 1)
    faithful_loss, faithful_parts = contrast.contrast_preserving_objective(
        teacher, views, mask, center
    )
    collapsed_loss, collapsed_parts = contrast.contrast_preserving_objective(
        collapsed, views, mask, center
    )

    assert float(collapsed_loss) > float(faithful_loss) + 0.25
    assert float(collapsed_parts["variance_noncollapse_loss"]) > 0.7
    assert float(faithful_parts["variance_noncollapse_loss"]) == pytest.approx(0.0)


def test_source_gate_accepts_contrast_preservation_and_rejects_raw_only_collapse() -> None:
    baseline = _validation(
        _scene_metrics(
            raw=0.90,
            raw_p05=0.84,
            residual=0.10,
            residual_p05=-0.10,
            spread=0.80,
            gram_mae=0.30,
            gram_correlation=0.25,
            probe_mae=0.12,
            probe_correlation=0.30,
            probe_std_ratio=0.90,
        )
    )
    candidate = _validation(
        _scene_metrics(
            raw=0.905,
            raw_p05=0.85,
            residual=0.13,
            residual_p05=-0.08,
            spread=0.90,
            gram_mae=0.25,
            gram_correlation=0.35,
            probe_mae=0.10,
            probe_correlation=0.40,
            probe_std_ratio=1.0,
        )
    )
    assert v2.attach_selection(candidate, baseline)["selection"]["eligible"] is True

    raw_only = _validation(
        _scene_metrics(
            raw=0.99,
            raw_p05=0.98,
            residual=0.14,
            residual_p05=-0.05,
            spread=0.0,
            gram_mae=0.20,
            gram_correlation=0.0,
            probe_mae=0.08,
            probe_correlation=0.0,
            probe_std_ratio=0.10,
        )
    )
    selected = v2.attach_selection(raw_only, baseline)["selection"]
    assert selected["eligible"] is False
    assert (
        selected["checks"][
            "every_scene_centroid_dispersion_at_least_0p75_teacher"
        ]
        is False
    )
    assert (
        selected["checks"]["every_scene_pair_gram_correlation_at_least_0p20"]
        is False
    )
    assert selected["checks"]["every_scene_absolute_visual_probe_calibrated"] is False


def test_gate_requires_exact_heldout_scenes_and_authority_rejects_target_access() -> None:
    row = _scene_metrics(
        raw=0.9,
        raw_p05=0.8,
        residual=0.1,
        residual_p05=-0.1,
        spread=0.8,
        gram_mae=0.3,
        gram_correlation=0.3,
    )
    complete = _validation(row)
    missing = deepcopy(complete)
    del missing["per_scene"][v2.VALIDATION_SCENES[-1]]
    with pytest.raises(ValueError, match="both heldout source scenes"):
        v2.attach_selection(missing, complete)

    authority = _authority()
    v2.validate_execution_authority(authority)
    target = deepcopy(authority)
    target["benchmark_execution_authorized"] = True
    with pytest.raises(ValueError, match="header differs"):
        v2.validate_execution_authority(target)
    query = deepcopy(authority)
    query["source_access"]["text_queries_opened"] = True
    with pytest.raises(ValueError, match="header differs"):
        v2.validate_execution_authority(query)


def test_v1_trainer_is_byte_frozen_and_v2_reduces_full_validation_frequency() -> None:
    assert file_record(Path(v1.__file__).resolve())["sha256"] == (
        "89770aaa501be87fffbbb562eb38800a71673691b5d1856b1d3296d246fe3aa2"
    )
    contract = v2.training_contract()
    assert contract["legacy_v1_modified"] is False
    assert contract["arm"] == "direction_only"
    assert contract["input"]["query_or_text"] == "prohibited"
    assert contract["input"]["target_or_benchmark"] == "prohibited"
    assert v2.OPTIMIZER_STEPS // v2.EVALUATION_INTERVAL == 12
    assert v2.synthetic_dry_run()["benchmark_opened"] is False
