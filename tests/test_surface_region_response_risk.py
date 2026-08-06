from __future__ import annotations

import inspect

import pytest
import torch

from radio_gs.losses.surface_region_response_risk import (
    VISUAL_CONTRAST_BANK_CONTRACT,
    build_train_only_visual_contrast_bank,
    compute_visual_contrast_scene_response_risk,
    compute_visual_contrast_scene_response_units,
)


def _bank_fixture() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    views = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [99.0, 99.0, 99.0]],
            [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0], [99.0, 99.0, 99.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.1, 0.9], [99.0, 99.0, 99.0]],
            [[-1.0, 0.0, 0.0], [-0.9, -0.1, 0.0], [99.0, 99.0, 99.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, True, False]] * 4,
        dtype=torch.bool,
    )
    return views, mask, ["train-a", "train-a", "train-b", "train-b"]


def test_visual_bank_is_fixed_count_deterministic_and_train_only() -> None:
    views, mask, scenes = _bank_fixture()
    bank, provenance = build_train_only_visual_contrast_bank(
        views, mask, scenes, direction_count=5
    )
    repeated, repeated_provenance = build_train_only_visual_contrast_bank(
        views, mask, scenes, direction_count=5
    )

    assert bank.shape == (5, 3)
    assert bank.device.type == "cpu"
    assert bank.dtype == torch.float32
    assert not bank.requires_grad
    torch.testing.assert_close(bank.norm(dim=-1), torch.ones(5))
    assert torch.equal(bank, repeated)
    assert provenance["contract"] == VISUAL_CONTRAST_BANK_CONTRACT
    assert provenance["source_scope"] == (
        "frozen_train_teacher_visual_descriptors_only"
    )
    assert provenance["uses_text_or_vocabulary"] is False
    assert provenance["learned_parameters"] == 0
    assert provenance["random_seed"] is None
    assert provenance["direction_count"] == 5
    assert provenance["bank_sha256_float32"] == repeated_provenance[
        "bank_sha256_float32"
    ]
    assert views.grad is None

    # The public constructor structurally cannot accept vocabulary or split
    # inputs, which prevents accidental validation/benchmark leakage.
    parameters = set(inspect.signature(build_train_only_visual_contrast_bank).parameters)
    assert parameters == {
        "teacher_view_descriptors",
        "teacher_mask",
        "scene_ids",
        "direction_count",
        "eps",
    }


def test_visual_bank_is_invariant_to_row_and_view_permutation() -> None:
    views, mask, scenes = _bank_fixture()
    bank, provenance = build_train_only_visual_contrast_bank(
        views, mask, scenes, direction_count=5
    )
    rows = torch.tensor([2, 0, 3, 1])
    permuted_views = views.detach()[rows][:, [1, 0, 2]]
    permuted_mask = mask[rows][:, [1, 0, 2]]
    permuted_scenes = [scenes[index] for index in rows.tolist()]
    permuted, permuted_provenance = build_train_only_visual_contrast_bank(
        permuted_views,
        permuted_mask,
        permuted_scenes,
        direction_count=5,
    )

    assert torch.equal(bank, permuted)
    assert provenance["normalized_valid_view_multiset_sha256"] == (
        permuted_provenance["normalized_valid_view_multiset_sha256"]
    )
    assert provenance["bank_sha256_float32"] == permuted_provenance[
        "bank_sha256_float32"
    ]


def test_visual_bank_refuses_silent_direction_padding() -> None:
    views = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="only"):
        build_train_only_visual_contrast_bank(
            views, mask, ["train"], direction_count=3
        )


def _risk_fixture() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]
]:
    teacher = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.2, 0.8],
            [-0.8, 0.2, 0.0],
        ]
    )
    teacher_views = teacher[:, None, :].repeat(1, 2, 1)
    mask = torch.ones(6, 2, dtype=torch.bool)
    bank = torch.eye(3)
    scenes = ["a", "a", "b", "b", "b", "b"]
    return teacher, teacher_views, mask, bank, scenes


def test_matching_teacher_has_zero_risk_and_only_student_gradient() -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    views.requires_grad_(True)
    bank.requires_grad_(True)
    student = teacher.clone().requires_grad_(True)

    risk, stats = compute_visual_contrast_scene_response_risk(
        student,
        views,
        mask,
        bank,
        scenes,
        mean_weight=0.5,
        cvar_weight=0.5,
    )

    assert risk.item() == pytest.approx(0.0, abs=1e-8)
    assert stats["scene_direction_unit_loss"].requires_grad is False
    assert stats["scene_direction_valid"].dtype == torch.bool
    risk.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert views.grad is None
    assert bank.grad is None


def test_combined_risk_averages_scenes_equally_despite_region_count() -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    student = teacher.clone()
    student[0] = torch.tensor([0.5, 0.5, 0.0])
    student[2] = torch.tensor([0.3, 0.6, 0.1])
    student.requires_grad_(True)

    kwargs = dict(mean_weight=1.0, cvar_weight=0.0)
    combined, _ = compute_visual_contrast_scene_response_risk(
        student, views, mask, bank, scenes, **kwargs
    )
    risk_a, _ = compute_visual_contrast_scene_response_risk(
        student[:2], views[:2], mask[:2], bank, ["a", "a"], **kwargs
    )
    risk_b, _ = compute_visual_contrast_scene_response_risk(
        student[2:], views[2:], mask[2:], bank, ["b"] * 4, **kwargs
    )

    torch.testing.assert_close(combined, 0.5 * (risk_a + risk_b))
    combined.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_listwise_unit_is_region_mean_with_normalized_uncertainty_weights() -> None:
    # Zero view disagreement gives every region confidence one.  The expected
    # listwise term is therefore exactly the per-region mean Brier divergence.
    teacher = torch.tensor([[1.0, 0.0], [0.4, 0.6], [0.0, 1.0]])
    student = torch.tensor(
        [[0.8, 0.2], [0.8, 0.2], [0.0, 1.0]], requires_grad=True
    )
    views = teacher[:, None, :].repeat(1, 2, 1)
    bank = torch.tensor([[1.0, 0.0]])
    units, valid, stats = compute_visual_contrast_scene_response_units(
        student,
        views,
        torch.ones(3, 2, dtype=torch.bool),
        bank,
        ["scene"] * 3,
        gap_weight=0.0,
        listwise_weight=1.0,
        listwise_temperature=0.25,
    )

    teacher_response = torch.nn.functional.normalize(teacher, dim=-1)[:, 0]
    student_response = torch.nn.functional.normalize(student, dim=-1)[:, 0]
    span = teacher_response.max() - teacher_response.min()
    teacher_logits = (teacher_response - teacher_response.mean()) / span / 0.25
    student_logits = (student_response - student_response.mean()) / span / 0.25
    expected = 0.5 * (
        torch.softmax(student_logits, dim=0)
        - torch.softmax(teacher_logits, dim=0)
    ).square().mean()

    assert valid.tolist() == [[True]]
    torch.testing.assert_close(units[0, 0], expected)
    torch.testing.assert_close(stats["listwise_unit_loss"][0, 0], expected)
    assert stats["region_uncertainty_confidence_mean"].item() == pytest.approx(1.0)


def test_uncertainty_downweights_inconsistent_valid_views_and_ignores_padding() -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    noisy = views.clone()
    noisy[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    padded = torch.cat([noisy, torch.full((6, 1, 3), 999.0)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(6, 1, dtype=torch.bool)], dim=1)
    student = teacher.roll(1, dims=0).requires_grad_(True)

    risk, stats = compute_visual_contrast_scene_response_risk(
        student, noisy, mask, bank, scenes
    )
    padded_risk, padded_stats = compute_visual_contrast_scene_response_risk(
        student, padded, padded_mask, bank, scenes
    )

    assert stats["region_uncertainty_confidence_mean"].item() < 1.0
    assert stats["gap_uncertainty_confidence_mean"].item() < 1.0
    torch.testing.assert_close(risk, padded_risk)
    torch.testing.assert_close(
        stats["scene_direction_unit_loss"],
        padded_stats["scene_direction_unit_loss"],
    )


def test_cvar_component_increases_pressure_on_worst_direction() -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    student = teacher.roll(1, dims=0).requires_grad_(True)
    mean_only, _ = compute_visual_contrast_scene_response_risk(
        student,
        views,
        mask,
        bank,
        scenes,
        mean_weight=1.0,
        cvar_weight=0.0,
        cvar_tail_fraction=0.25,
    )
    mean_cvar, stats = compute_visual_contrast_scene_response_risk(
        student,
        views,
        mask,
        bank,
        scenes,
        mean_weight=0.5,
        cvar_weight=0.5,
        cvar_tail_fraction=0.25,
    )

    assert mean_cvar >= mean_only
    assert stats["scene_upper_fractional_cvar"].shape == (2,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap_weight": 0.7, "listwise_weight": 0.7},
        {"standard_error_multiplier": -1.0},
        {"listwise_temperature": 0.0},
    ],
)
def test_response_risk_rejects_invalid_contract_scalars(kwargs) -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    with pytest.raises(ValueError):
        compute_visual_contrast_scene_response_units(
            teacher, views, mask, bank, scenes, **kwargs
        )


def test_response_risk_requires_two_valid_views_for_uncertainty() -> None:
    teacher, views, mask, bank, scenes = _risk_fixture()
    mask[0, 1] = False
    with pytest.raises(ValueError, match="at least two"):
        compute_visual_contrast_scene_response_risk(
            teacher, views, mask, bank, scenes
        )
