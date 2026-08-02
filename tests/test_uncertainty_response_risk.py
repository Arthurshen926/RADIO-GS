import pytest
import torch

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss,
)
from radio_gs.losses.uncertainty_response_risk import (
    compute_equal_scene_mean_fractional_cvar_risk,
    compute_uncertainty_weighted_pairwise_mean_cvar_risk,
    compute_uncertainty_weighted_scene_query_pairwise_gap_units,
)


def _teacher_fixture():
    teacher = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.75**0.5],
            [0.0, 1.0],
            [-0.5, 0.75**0.5],
        ],
        requires_grad=True,
    )
    views = torch.stack([torch.stack([row, row]) for row in teacher.detach()])
    views.requires_grad_(True)
    mask = torch.ones(4, 2, dtype=torch.bool)
    text = torch.eye(2, requires_grad=True)
    return teacher, views, mask, text


def test_scene_query_units_retain_only_student_gradient():
    teacher, views, mask, text = _teacher_fixture()
    student = teacher.detach().clone()
    student[1] = torch.tensor([0.0, 1.0])
    student.requires_grad_(True)

    units, validity, stats = (
        compute_uncertainty_weighted_scene_query_pairwise_gap_units(
            student,
            teacher,
            views,
            mask,
            text,
            ["a", "a", "b", "b"],
        )
    )

    assert units.shape == (2, 2)
    assert units.requires_grad
    assert validity.shape == units.shape
    assert not validity.requires_grad
    assert stats["scene_query_weight_sum"].requires_grad is False
    units[validity].sum().backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum().item() > 0.0
    assert teacher.grad is None
    assert views.grad is None
    assert text.grad is None


def test_differentiable_units_exactly_match_frozen_v1_selector_units():
    torch.manual_seed(23)
    student = torch.randn(6, 5, requires_grad=True)
    teacher = torch.randn(6, 5)
    views = teacher[:, None, :].repeat(1, 3, 1)
    views[:, 1] += 0.1 * torch.randn_like(views[:, 1])
    views[:, 2] += 0.1 * torch.randn_like(views[:, 2])
    mask = torch.ones(6, 3, dtype=torch.bool)
    text = torch.randn(7, 5)
    scenes = ["a", "a", "a", "b", "b", "b"]

    units, validity, _ = (
        compute_uncertainty_weighted_scene_query_pairwise_gap_units(
            student, teacher, views, mask, text, scenes
        )
    )
    _v1_scalar, v1_stats = (
        compute_scene_wise_uncertainty_weighted_text_response_pairwise_gap_smooth_l1_loss(
            student, teacher, views, mask, text, scenes
        )
    )

    torch.testing.assert_close(units.detach(), v1_stats["scene_query_loss"])
    assert torch.equal(validity, v1_stats["scene_query_valid"])


def test_fractional_cvar_risk_targets_worst_unit_more_strongly():
    units = torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    validity = torch.ones_like(units, dtype=torch.bool)

    risk, stats = compute_equal_scene_mean_fractional_cvar_risk(
        units,
        validity,
        mean_weight=0.5,
        cvar_weight=0.5,
        cvar_tail_fraction=0.375,
    )

    expected_cvar = (4.0 + 0.5 * 3.0) / 1.5
    assert risk.item() == pytest.approx(0.5 * 2.5 + 0.5 * expected_cvar)
    risk.backward()
    # Mean contributes 0.5/4 to every unit. Fractional CVaR contributes
    # 0.5*(2/3) to the worst and 0.5*(1/3) to the second-worst.
    torch.testing.assert_close(
        units.grad,
        torch.tensor([[11.0 / 24.0, 7.0 / 24.0, 1.0 / 8.0, 1.0 / 8.0]]),
    )
    torch.testing.assert_close(
        stats["scene_upper_fractional_cvar"], torch.tensor([expected_cvar])
    )


def test_risk_averages_scenes_equally_despite_different_valid_query_counts():
    units = torch.tensor(
        [[1.0, 3.0, 99.0, 99.0], [2.0, 4.0, 6.0, 8.0]],
        requires_grad=True,
    )
    validity = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )

    risk, stats = compute_equal_scene_mean_fractional_cvar_risk(
        units,
        validity,
        mean_weight=1.0,
        cvar_weight=0.0,
    )

    assert risk.item() == pytest.approx((2.0 + 5.0) / 2.0)
    assert stats["equal_scene_count"].item() == 2
    risk.backward()
    torch.testing.assert_close(
        units.grad,
        torch.tensor(
            [[0.25, 0.25, 0.0, 0.0], [0.125, 0.125, 0.125, 0.125]]
        ),
    )


def test_composed_mean_cvar_risk_is_differentiable_and_teacher_frozen():
    teacher, views, mask, text = _teacher_fixture()
    student = teacher.detach().roll(1, dims=0).requires_grad_(True)

    risk, stats = compute_uncertainty_weighted_pairwise_mean_cvar_risk(
        student,
        teacher,
        views,
        mask,
        text,
        ["a", "a", "b", "b"],
        mean_weight=0.5,
        cvar_weight=0.5,
        cvar_tail_fraction=0.10,
    )

    assert risk.requires_grad
    assert stats["scene_query_unit_loss"].requires_grad is False
    risk.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    assert views.grad is None
    assert text.grad is None


@pytest.mark.parametrize(
    ("mean_weight", "cvar_weight"),
    [(0.6, 0.6), (-0.1, 1.1)],
)
def test_mean_cvar_risk_rejects_invalid_weights(mean_weight, cvar_weight):
    with pytest.raises(ValueError):
        compute_equal_scene_mean_fractional_cvar_risk(
            torch.ones(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            mean_weight=mean_weight,
            cvar_weight=cvar_weight,
        )
