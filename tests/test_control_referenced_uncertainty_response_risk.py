from __future__ import annotations

import pytest
import torch

from radio_gs.losses.control_referenced_uncertainty_response_risk import (
    compute_control_referenced_exact_hinge_risk,
)


def test_identity_control_has_zero_paired_objective_and_constraints() -> None:
    control = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    candidate = control.clone().requires_grad_(True)
    valid = torch.ones_like(control, dtype=torch.bool)
    unary = torch.tensor(0.25, requires_grad=True)

    loss, stats = compute_control_referenced_exact_hinge_risk(
        candidate,
        valid,
        control,
        valid,
        unary,
        0.25,
    )

    assert loss.item() == pytest.approx(0.0)
    assert stats["global_mean_delta"].item() == pytest.approx(0.0)
    assert stats["global_upper_fractional_cvar_delta"].item() == pytest.approx(0.0)
    assert stats["independent_unary_delta"].item() == pytest.approx(0.0)
    assert all(
        value.item() == pytest.approx(0.0)
        for value in stats["exact_hinge_violations"].values()
    )
    loss.backward()
    assert candidate.grad is not None
    assert candidate.grad.abs().sum().item() > 0.0
    # ReLU uses its zero subgradient at an exactly feasible unary boundary.
    assert unary.grad is not None and unary.grad.item() == pytest.approx(0.0)


def test_exact_hinges_match_preregistered_dimensionless_constraints() -> None:
    control = torch.ones(2, 2)
    candidate = torch.tensor([[0.90, 0.90], [1.02, 1.04]], requires_grad=True)
    valid = torch.ones_like(control, dtype=torch.bool)
    unary = torch.tensor(1.10, requires_grad=True)

    loss, stats = compute_control_referenced_exact_hinge_risk(
        candidate,
        valid,
        control,
        valid,
        unary,
        1.0,
        cvar_tail_fraction=0.5,
        global_cvar_tolerance=0.025,
        worst_scene_mean_tolerance=0.025,
        worst_scene_cvar_tolerance=0.035,
        unary_delta_tolerance=0.05,
    )

    assert stats["global_mean_delta"].item() == pytest.approx(-0.035)
    assert stats["global_upper_fractional_cvar_delta"].item() == pytest.approx(0.03)
    assert stats["worst_scene_mean_delta"].item() == pytest.approx(0.03)
    assert stats["worst_scene_upper_fractional_cvar_delta"].item() == (
        pytest.approx(0.04)
    )
    assert stats["independent_unary_delta"].item() == pytest.approx(0.10)
    violations = stats["exact_hinge_violations"]
    assert violations["global_cvar"].item() == pytest.approx(0.005, abs=1e-6)
    assert violations["worst_scene_mean"].item() == pytest.approx(0.005, abs=1e-6)
    assert violations["worst_scene_cvar"].item() == pytest.approx(0.005, abs=1e-6)
    assert violations["independent_unary"].item() == pytest.approx(0.05, abs=1e-6)
    assert loss.item() == pytest.approx(0.03, abs=1e-6)

    loss.backward()
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
    assert unary.grad is not None and unary.grad.item() > 0.0
    assert control.grad is None


def test_validity_is_aligned_and_invalid_units_have_no_gradient() -> None:
    control = torch.tensor([[1.0, 99.0], [2.0, 4.0]])
    candidate = control.clone().requires_grad_(True)
    valid = torch.tensor([[True, False], [True, True]])

    loss, stats = compute_control_referenced_exact_hinge_risk(
        candidate,
        valid,
        control,
        valid,
        torch.tensor(1.0, requires_grad=True),
        1.0,
    )
    loss.backward()

    assert stats["valid_scene_query_count"].item() == 3
    assert candidate.grad is not None
    assert candidate.grad[0, 1].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cvar_tail_fraction": 0.0},
        {"global_cvar_tolerance": -0.1},
        {"worst_scene_mean_tolerance": -0.1},
        {"worst_scene_cvar_tolerance": -0.1},
        {"exact_penalty_weight": 0.0},
    ],
)
def test_control_referenced_risk_rejects_invalid_contract_scalars(kwargs) -> None:
    units = torch.ones(1, 2)
    valid = torch.ones_like(units, dtype=torch.bool)
    with pytest.raises(ValueError):
        compute_control_referenced_exact_hinge_risk(
            units,
            valid,
            units,
            valid,
            torch.tensor(1.0),
            1.0,
            **kwargs,
        )


def test_control_referenced_risk_rejects_misaligned_validity() -> None:
    units = torch.ones(1, 2)
    candidate_valid = torch.tensor([[True, True]])
    control_valid = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="invalid"):
        compute_control_referenced_exact_hinge_risk(
            units,
            candidate_valid,
            units,
            control_valid,
            torch.tensor(1.0),
            1.0,
        )
