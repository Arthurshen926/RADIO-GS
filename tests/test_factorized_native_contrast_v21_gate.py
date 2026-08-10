from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as v2,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as v21,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


SHA = "a" * 64


def _pair_row(*, spread: float, mae: float, correlation: float) -> dict[str, float]:
    return {
        "student_to_teacher_spread_ratio": spread,
        "teacher_centered_pair_gram_mae": mae,
        "teacher_centered_pair_gram_correlation": correlation,
    }


def _pair_validation(row: dict[str, float]) -> dict:
    return {
        "per_scene": {scene: deepcopy(row) for scene in v21.VALIDATION_SCENES}
    }


def _complete_row(
    *, spread: float, mae: float, correlation: float
) -> dict[str, float | int]:
    return {
        "eligible_rows": 8,
        "mean_all_view_cosine": 0.905,
        "p05_row_mean_all_view_cosine": 0.85,
        "mean_teacher_centered_residual_cosine": 0.13,
        "p05_teacher_centered_residual_cosine": -0.08,
        "student_centroid_norm": 0.95,
        "teacher_centroid_norm": 0.96,
        "student_spread": 0.08,
        "teacher_spread": 0.08,
        "student_to_teacher_spread_ratio": spread,
        "teacher_centered_pair_gram_mae": mae,
        "teacher_centered_pair_gram_correlation": correlation,
        "teacher_centered_pair_count": 16,
        "absolute_visual_probe_response_mae": 0.10,
        "absolute_visual_probe_response_correlation": 0.40,
        "absolute_visual_probe_response_std_ratio": 1.0,
    }


def _complete_validation(row: dict[str, float | int]) -> dict:
    result = {
        "per_scene": {scene: deepcopy(row) for scene in v21.VALIDATION_SCENES}
    }
    for key in (
        "mean_all_view_cosine",
        "p05_row_mean_all_view_cosine",
        "mean_teacher_centered_residual_cosine",
        "p05_teacher_centered_residual_cosine",
        "student_to_teacher_spread_ratio",
        "teacher_centered_pair_gram_mae",
        "teacher_centered_pair_gram_correlation",
        "absolute_visual_probe_response_mae",
        "absolute_visual_probe_response_correlation",
        "absolute_visual_probe_response_std_ratio",
    ):
        result[f"macro_{key}"] = float(row[key])
    return result


def _authority() -> dict:
    record = {"path": "/frozen", "sha256": SHA}
    return {
        "schema": v21.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_contrast_v21_exact4train_2validation",
        "trainer_implementation": record,
        "base_v2_execution_authority": record,
        "training_contract_sha256": canonical_json_sha256(v21.training_contract()),
        "authorized_arm": "direction_only",
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": v21.source_access(),
    }


def test_collapsed_baseline_uses_absolute_corr_and_strict_mae_not_corr_regression() -> None:
    baseline = _pair_validation(_pair_row(spread=0.04, mae=0.48, correlation=0.83))
    candidate = _pair_validation(_pair_row(spread=0.86, mae=0.28, correlation=0.69))
    passed, audit = v21._conditional_pair_geometry(candidate, baseline)

    assert passed is True
    for scene in v21.VALIDATION_SCENES:
        row = audit[scene]
        assert row["baseline_variance_qualified"] is False
        assert row["comparator"] == "variance_unqualified_strict_mae_improvement"
        assert row["correlation_non_regression_check"] is True
        assert row["candidate_pair_correlation"] < row["baseline_pair_correlation"]
        assert row["passed"] is True


def test_variance_qualified_baseline_still_requires_corr_non_regression() -> None:
    baseline = _pair_validation(_pair_row(spread=0.80, mae=0.30, correlation=0.70))
    candidate = _pair_validation(_pair_row(spread=0.90, mae=0.20, correlation=0.60))
    passed, audit = v21._conditional_pair_geometry(candidate, baseline)

    assert passed is False
    assert all(
        row["comparator"] == "variance_qualified_non_regression"
        and row["correlation_non_regression_check"] is False
        for row in audit.values()
    )


@pytest.mark.parametrize(
    "mae,correlation",
    [(0.48, 0.69), (0.28, 0.19)],
)
def test_collapsed_baseline_requires_strict_mae_improvement_and_absolute_corr(
    mae: float, correlation: float
) -> None:
    baseline = _pair_validation(_pair_row(spread=0.04, mae=0.48, correlation=0.83))
    candidate = _pair_validation(
        _pair_row(spread=0.90, mae=mae, correlation=correlation)
    )
    assert v21._conditional_pair_geometry(candidate, baseline)[0] is False


def test_v21_fixes_only_pair_comparator_and_retains_all_other_v2_checks() -> None:
    baseline_row = _complete_row(spread=0.04, mae=0.48, correlation=0.83)
    baseline_row.update(
        {
            "mean_all_view_cosine": 0.90,
            "p05_row_mean_all_view_cosine": 0.84,
            "mean_teacher_centered_residual_cosine": 0.10,
            "p05_teacher_centered_residual_cosine": -0.10,
            "absolute_visual_probe_response_mae": 0.12,
            "absolute_visual_probe_response_correlation": 0.30,
            "absolute_visual_probe_response_std_ratio": 0.90,
        }
    )
    baseline = _complete_validation(baseline_row)
    candidate = _complete_validation(
        _complete_row(spread=0.86, mae=0.28, correlation=0.69)
    )

    v2_selection = v2.attach_selection(candidate, baseline)["selection"]
    v21_selection = v21.attach_selection(candidate, baseline)["selection"]
    assert v2_selection["eligible"] is False
    assert v2_selection["checks"]["every_scene_pair_geometry_non_regression"] is False
    assert v21_selection["eligible"] is True
    assert (
        v21_selection["checks"][
            "every_scene_pair_geometry_conditional_baseline"
        ]
        is True
    )
    common = set(v2_selection["checks"]) - {
        "every_scene_pair_geometry_non_regression"
    }
    assert {key: v21_selection["checks"][key] for key in common} == {
        key: v2_selection["checks"][key] for key in common
    }


def test_authority_remains_source_only_and_v2_closure_is_frozen() -> None:
    authority = _authority()
    v21.validate_execution_authority(authority)
    target = deepcopy(authority)
    target["benchmark_execution_authorized"] = True
    with pytest.raises(ValueError, match="header differs"):
        v21.validate_execution_authority(target)

    assert file_record(Path(v2.__file__).resolve())["sha256"] == (
        "deb3019af8cb7c2000f001c20b9f60a6d59c0b040c839d0cfa27c93df59963a0"
    )
    objective = Path(v2.contrast.__file__).resolve()
    assert file_record(objective)["sha256"] == (
        "e12f8a333f5e23944a4fc264e997b18f7a8fb33cd621efd2c466bd3e8eb6f571"
    )
    assert v21.training_contract()["contrast_v2_modified"] is False
    assert v21.synthetic_dry_run()["benchmark_opened"] is False
