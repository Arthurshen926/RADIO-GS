from __future__ import annotations

import inspect

import pytest
import torch

from radio_gs.losses.control_referenced_uncertainty_response_risk import (
    compute_control_referenced_exact_hinge_risk,
)
from radio_gs.losses.dual_descriptor_response_risk import (
    DESCRIPTOR_METRIC_NAMES,
    FIT_CONSTRAINT_NAMES,
    UNARY_METRIC_NAMES,
    build_seed0_single_conjunction_gate,
    calibrate_epoch0_gradient_weights,
    compute_dual_descriptor_loss_components,
    compute_dual_descriptor_response_risk,
)


def _fit_checks(value: bool = True) -> dict[str, bool]:
    return {name: value for name in FIT_CONSTRAINT_NAMES}


def _passing_gate_kwargs() -> dict[str, object]:
    return {
        "selected_epoch": 1,
        "dev_normalized_mean_delta": -0.0025,
        "dev_global_cvar10_delta": 0.005,
        "dev_worst_scene_mean_delta": 0.010,
        "dev_worst_scene_cvar10_delta": 0.010,
        "validation_unary_relative_deltas": {
            name: 0.0 for name in UNARY_METRIC_NAMES
        },
        "validation_descriptor_deltas": {
            name: -0.002 for name in DESCRIPTOR_METRIC_NAMES
        },
        "official_token_bitwise_equal": True,
        "official_descriptor_bitwise_equal": True,
        "fit_constraint_checks": _fit_checks(),
        "point_render_max_abs_error": 1e-6,
    }


def test_identity_components_and_risk_are_zero() -> None:
    semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    teacher = semantic.detach().clone().requires_grad_(True)
    views = teacher.detach()[:, None, :].clone().requires_grad_(True)
    view_valid = torch.ones(2, 1, dtype=torch.bool)
    text = torch.eye(2, requires_grad=True)

    structural, all_view, relation, unary = compute_dual_descriptor_loss_components(
        semantic, teacher, views, view_valid, text
    )
    assert structural.item() == pytest.approx(0.0)
    assert all_view.item() == pytest.approx(0.0)
    assert relation.item() == pytest.approx(0.0)
    assert unary.item() == pytest.approx(0.0)

    units = torch.ones(2, 2)
    valid = torch.ones_like(units, dtype=torch.bool)
    total, stats = compute_dual_descriptor_response_risk(
        units,
        valid,
        units,
        valid,
        semantic,
        teacher,
        views,
        view_valid,
        text,
        1.0,
        lambda_unary=0.5,
        lambda_risk=0.25,
    )
    assert total.item() == pytest.approx(0.0)
    assert stats["objective"].item() == pytest.approx(0.0)
    assert (
        stats["constraint_penalty_contract"][
            "mathematical_exact_penalty_guarantee"
        ]
        is False
    )


def test_composite_formula_and_detached_controls() -> None:
    semantic = torch.tensor(
        [[0.9, 0.2], [0.1, 0.8]], dtype=torch.float64, requires_grad=True
    )
    teacher = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    views = torch.tensor(
        [[[1.0, 0.0], [0.8, 0.2]], [[0.0, 1.0], [0.2, 0.8]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    text = torch.eye(2, dtype=torch.float64, requires_grad=True)
    view_valid = torch.ones(2, 2, dtype=torch.bool)
    candidate = torch.tensor(
        [[0.9, 0.95], [1.0, 1.05]], requires_grad=True
    )
    control = torch.ones(2, 2, requires_grad=True)
    valid = torch.ones_like(candidate, dtype=torch.bool)

    structural, all_view, relation, unary = compute_dual_descriptor_loss_components(
        semantic, teacher, views, view_valid, text
    )
    risk, _ = compute_control_referenced_exact_hinge_risk(
        candidate, valid, control, valid, unary, 1.0
    )
    total, stats = compute_dual_descriptor_response_risk(
        candidate,
        valid,
        control,
        valid,
        semantic,
        teacher,
        views,
        view_valid,
        text,
        1.0,
        lambda_unary=0.3,
        lambda_risk=0.2,
    )

    expected = structural + 0.3 * unary + 0.2 * risk
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(stats["structural_loss"], all_view + 0.1 * relation)
    total.backward()
    assert semantic.grad is not None and semantic.grad.abs().sum() > 0
    assert candidate.grad is not None and candidate.grad.abs().sum() > 0
    assert teacher.grad is None
    assert views.grad is None
    assert text.grad is None
    assert control.grad is None


def test_epoch0_gradient_calibration_hits_quarter_structural_norm() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    structural = parameter.square()
    unary = 3.0 * parameter
    risk = parameter

    lambda_unary, lambda_risk, report = calibrate_epoch0_gradient_weights(
        structural, unary, risk, [parameter]
    )

    assert report["structural_gradient_l2"] == pytest.approx(4.0)
    assert report["target_weighted_auxiliary_gradient_l2"] == pytest.approx(1.0)
    assert lambda_unary == pytest.approx(1.0 / 3.0)
    assert lambda_risk == pytest.approx(1.0)
    assert report["selection_kind"].endswith("not_search")


def test_epoch0_gradient_calibration_rejects_degenerate_branch() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    with pytest.raises(ValueError, match="degenerate"):
        calibrate_epoch0_gradient_weights(
            parameter.square(), parameter * 0.0, parameter, [parameter]
        )


def test_seed0_gate_is_one_conjunction_at_all_boundaries() -> None:
    report = build_seed0_single_conjunction_gate(**_passing_gate_kwargs())

    assert report["passed"] is True
    assert report["passed"] is all(report["checks"].values())
    assert report["seed"] == 0
    assert report["conjunction"] == "all(checks.values())"
    assert report["data_boundary"] == {
        "fit_and_frozen_dev_aggregates_only": True,
        "benchmark_targets_or_metrics_used": False,
    }


@pytest.mark.parametrize(
    ("key", "value", "failed_check"),
    [
        ("selected_epoch", 0, "selected_epoch_gt_zero"),
        (
            "dev_normalized_mean_delta",
            -0.0024,
            "dev_normalized_mean_delta_le_negative_0p0025",
        ),
        (
            "dev_global_cvar10_delta",
            0.0051,
            "dev_global_cvar10_delta_le_0p005",
        ),
        (
            "dev_worst_scene_mean_delta",
            0.0101,
            "dev_worst_scene_mean_delta_le_0p010",
        ),
        (
            "dev_worst_scene_cvar10_delta",
            0.0101,
            "dev_worst_scene_cvar10_delta_le_0p010",
        ),
        (
            "official_token_bitwise_equal",
            False,
            "official_outputs_bitwise_equal",
        ),
        (
            "official_descriptor_bitwise_equal",
            False,
            "official_outputs_bitwise_equal",
        ),
        (
            "point_render_max_abs_error",
            1.1e-6,
            "point_render_max_abs_error_le_1e_minus_6",
        ),
    ],
)
def test_seed0_gate_fails_closed_per_scalar_check(
    key: str, value: object, failed_check: str
) -> None:
    kwargs = _passing_gate_kwargs()
    kwargs[key] = value
    report = build_seed0_single_conjunction_gate(**kwargs)
    assert report["passed"] is False
    assert report["checks"][failed_check] is False


def test_seed0_gate_fails_unary_descriptor_and_fit_constraints() -> None:
    cases: list[tuple[str, dict[str, object], str]] = []
    unary = {name: 0.0 for name in UNARY_METRIC_NAMES}
    unary[UNARY_METRIC_NAMES[0]] = 1e-4
    cases.append(
        (
            "unary",
            {"validation_unary_relative_deltas": unary},
            "validation_unary_relative_deltas_le_zero",
        )
    )
    descriptor = {name: 0.0 for name in DESCRIPTOR_METRIC_NAMES}
    descriptor[DESCRIPTOR_METRIC_NAMES[1]] = -0.0021
    cases.append(
        (
            "descriptor",
            {"validation_descriptor_deltas": descriptor},
            "validation_descriptor_deltas_ge_negative_0p002",
        )
    )
    fit = _fit_checks()
    fit[FIT_CONSTRAINT_NAMES[-1]] = False
    cases.append(
        (
            "fit",
            {"fit_constraint_checks": fit},
            "all_fit_constraints_feasible",
        )
    )

    for _label, override, failed_check in cases:
        kwargs = _passing_gate_kwargs()
        kwargs.update(override)
        report = build_seed0_single_conjunction_gate(**kwargs)
        assert report["passed"] is False
        assert report["checks"][failed_check] is False


def test_gate_and_loss_interfaces_have_no_benchmark_target_input() -> None:
    for function in (
        compute_dual_descriptor_loss_components,
        compute_dual_descriptor_response_risk,
        build_seed0_single_conjunction_gate,
    ):
        names = set(inspect.signature(function).parameters)
        assert not any("benchmark" in name or "target" in name for name in names)

    kwargs = _passing_gate_kwargs()
    fit = _fit_checks()
    fit["benchmark_target_score"] = True
    kwargs["fit_constraint_checks"] = fit
    with pytest.raises(ValueError, match="exactly"):
        build_seed0_single_conjunction_gate(**kwargs)
