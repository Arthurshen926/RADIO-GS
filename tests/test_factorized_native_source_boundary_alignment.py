from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from radio_gs.losses.factorized_native_source_boundary_alignment import (
    exact_multiview_teacher_probability,
    exact_student_margin,
    source_balanced_boundary_alignment_loss,
)
from radio_gs.querying.unified_query import cosine_relevancy_torch


def _banks() -> tuple[torch.Tensor, torch.Tensor]:
    positive = F.normalize(
        torch.tensor([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]), dim=-1
    )
    negative = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return positive, negative


def test_student_margin_is_exact_frozen_relevance_logit() -> None:
    positive, negative = _banks()
    student = F.normalize(torch.tensor([[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]), dim=-1)
    margin = exact_student_margin(student, positive, negative)
    expected = cosine_relevancy_torch(
        student,
        positive,
        negative,
        logit_scale=10.0,
        assume_normalized=True,
    )
    assert torch.allclose(torch.sigmoid(10.0 * margin), expected, atol=1e-7, rtol=0)


def test_teacher_keeps_negative_max_and_sigmoid_inside_view_average() -> None:
    positive, negative = _banks()
    views = F.normalize(
        torch.tensor(
            [
                [[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]],
                [[-2.0, 1.0, 0.0], [8.0, 1.0, 0.0]],
            ]
        ),
        dim=-1,
    )
    mask = torch.tensor([[True, True], [True, False]])
    observed = exact_multiview_teacher_probability(
        views, mask, positive, negative, query_chunk_rows=1
    )
    positive_score = torch.einsum("bvd,qd->bvq", views, positive)
    negative_score = torch.einsum("bvd,kd->bvk", views, negative).amax(dim=-1)
    per_view = torch.sigmoid(10.0 * (positive_score - negative_score[..., None]))
    expected = (per_view * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    assert torch.allclose(observed, expected, atol=1e-7, rtol=0)

    # This check guards the precise historical aggregation error: nonlinearity
    # and the per-view max cannot be moved outside the view reduction.
    mean_margin_surrogate = torch.sigmoid(
        10.0
        * (
            (positive_score * mask[..., None]).sum(dim=1)
            / mask.sum(dim=1, keepdim=True)
            - (
                (negative_score * mask).sum(dim=1)
                / mask.sum(dim=1)
            )[:, None]
        )
    )
    assert not torch.allclose(observed, mean_margin_surrogate, atol=1e-4, rtol=0)


def test_boundary_loss_has_gradient_that_can_change_margin_signs() -> None:
    positive, negative = _banks()
    teacher = F.normalize(
        torch.tensor(
            [
                [[3.0, 1.0, 0.0]],
                [[-3.0, 1.0, 0.0]],
            ]
        ),
        dim=-1,
    )
    mask = torch.ones(2, 1, dtype=torch.bool)
    student_parameter = torch.nn.Parameter(
        F.normalize(torch.tensor([[-3.0, 1.0, 0.0], [3.0, 1.0, 0.0]]), dim=-1)
    )
    before = exact_student_margin(student_parameter, positive, negative).detach()
    output = source_balanced_boundary_alignment_loss(
        student_parameter, teacher, mask, positive, negative
    )
    output.loss.backward()
    assert student_parameter.grad is not None
    assert bool(torch.isfinite(student_parameter.grad).all())
    assert float(student_parameter.grad.norm()) > 0
    with torch.no_grad():
        student_parameter -= 0.25 * student_parameter.grad
    after = exact_student_margin(student_parameter, positive, negative).detach()
    teacher_probability = exact_multiview_teacher_probability(
        teacher, mask, positive, negative
    )
    target_sign = torch.where(teacher_probability >= 0.5, 1.0, -1.0)
    assert float((after * target_sign).mean()) > float((before * target_sign).mean())


def test_balancing_is_invariant_to_duplicate_easy_negative_units() -> None:
    # The helper is exercised indirectly with one positive and one negative
    # query for each region.  Repeating a teacher-negative region must not
    # change the equal-class primary risk merely through class count.
    positive, negative = _banks()
    teacher = F.normalize(
        torch.tensor([[[3.0, 1.0, 0.0]], [[-3.0, 1.0, 0.0]]]), dim=-1
    )
    student = F.normalize(torch.tensor([[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]]), dim=-1)
    mask = torch.ones(2, 1, dtype=torch.bool)
    base = source_balanced_boundary_alignment_loss(
        student, teacher, mask, positive, negative, soft_fidelity_weight=0.0
    )
    repeated = source_balanced_boundary_alignment_loss(
        torch.cat((student, student[1:].repeat(4, 1))),
        torch.cat((teacher, teacher[1:].repeat(4, 1, 1))),
        torch.ones(6, 1, dtype=torch.bool),
        positive,
        negative,
        soft_fidelity_weight=0.0,
    )
    assert repeated.teacher_negative_pairs > base.teacher_negative_pairs
    assert repeated.balanced_hard_boundary_loss == pytest.approx(
        float(base.balanced_hard_boundary_loss), abs=1e-7
    )


def test_boundary_loss_rejects_single_class_batch() -> None:
    positive, negative = _banks()
    teacher = F.normalize(torch.tensor([[[3.0, 1.0, 0.0]]]), dim=-1)
    student = teacher[:, 0]
    mask = torch.ones(1, 1, dtype=torch.bool)
    # Restrict to one positive query so the failure is explicit instead of
    # silently falling back to the negative-dominated unbalanced objective.
    with pytest.raises(ValueError, match="both teacher classes"):
        source_balanced_boundary_alignment_loss(
            student, teacher, mask, positive[:1], negative
        )
