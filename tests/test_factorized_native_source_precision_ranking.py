from __future__ import annotations

import pytest
import torch

from radio_gs.losses import factorized_native_source_precision_ranking as dba_v2


def _patched_boundary(monkeypatch: pytest.MonkeyPatch, target: torch.Tensor) -> None:
    monkeypatch.setattr(
        dba_v2.dba_v1,
        "exact_student_margin",
        lambda descriptor, positive, negative: descriptor,
    )
    monkeypatch.setattr(
        dba_v2.dba_v1,
        "exact_multiview_teacher_probability",
        lambda *args, **kwargs: target,
    )


def _dummy_inputs(margin: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        margin,
        torch.ones(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1),
        torch.ones(4, 1),
    )


def test_hard_negative_count_is_derived_from_precision_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.full((2, 10), 0.1)
    target[0, 0] = 0.9
    target[1, 0] = 0.8
    _patched_boundary(monkeypatch, target)
    margin = torch.linspace(-0.2, 0.2, 20).reshape(2, 10).requires_grad_(True)
    output = dba_v2.source_precision_constrained_ranking_loss(
        *_dummy_inputs(margin)
    )
    assert dba_v2.HARD_NEGATIVES_PER_POSITIVE == 3
    assert output.teacher_positive_pairs == 2
    assert output.selected_hard_negative_pairs == 6
    assert output.global_order_pairs > 0
    output.loss.backward()
    assert margin.grad is not None
    assert bool(torch.isfinite(margin.grad).all())


def test_easy_negatives_do_not_dilute_zero_boundary_tail_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.full((1, 10), 0.1)
    target[0, 0] = 0.9
    _patched_boundary(monkeypatch, target)
    margin = torch.tensor(
        [[0.05, 0.10, 0.08, 0.06, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80]],
        requires_grad=True,
    )
    output = dba_v2.source_precision_constrained_ranking_loss(
        *_dummy_inputs(margin),
        global_order_weight=0.0,
    )
    output.loss.backward()
    # Three negatives are selected for one positive.  Far easy negatives do
    # not enter the boundary, soft-fidelity, or boundary-rank objectives.
    assert margin.grad is not None
    assert bool((margin.grad[0, 1:4].abs() > 0).all())
    assert torch.equal(margin.grad[0, 4:], torch.zeros_like(margin.grad[0, 4:]))


def test_global_order_loss_moves_student_in_teacher_order() -> None:
    student = torch.tensor([0.2, 0.1, -0.1, -0.2], requires_grad=True)
    teacher = torch.tensor([0.1, 0.3, 0.7, 0.9])
    before = student.detach().clone()
    loss, pairs = dba_v2._global_order_rank(student, teacher)
    loss.backward()
    assert pairs == 2
    with torch.no_grad():
        student -= 0.1 * student.grad
    assert float(student[-1] - student[0]) > float(before[-1] - before[0])


def test_single_teacher_class_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    target = torch.full((2, 10), 0.1)
    _patched_boundary(monkeypatch, target)
    margin = torch.zeros(2, 10)
    with pytest.raises(ValueError, match="both teacher boundary classes"):
        dba_v2.source_precision_constrained_ranking_loss(*_dummy_inputs(margin))
