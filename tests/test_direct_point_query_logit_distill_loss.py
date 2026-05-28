import torch
import torch.nn.functional as F

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_direct_point_query_logit_distill_loss,
    compute_direct_point_query_support_distill_loss,
)


def test_query_logit_distill_is_zero_for_matching_student_teacher():
    text = F.normalize(torch.eye(3), dim=-1)
    teacher = text[[0, 1, 2]]
    student = teacher.clone()

    loss, stats = compute_direct_point_query_logit_distill_loss(
        student,
        teacher,
        text,
        temperature=0.5,
    )

    assert loss.item() < 1e-6
    assert stats["valid_ratio"].item() == 1.0
    assert stats["agreement"].item() == 1.0


def test_query_logit_distill_penalizes_ranking_swaps():
    text = F.normalize(torch.eye(3), dim=-1)
    teacher = text[[0, 1, 2]]
    student = text[[1, 0, 2]]

    loss, stats = compute_direct_point_query_logit_distill_loss(
        student,
        teacher,
        text,
        temperature=0.5,
    )

    assert loss.item() > 0.1
    assert stats["agreement"].item() < 1.0


def test_query_support_distill_is_zero_for_matching_support_distribution():
    text = F.normalize(torch.eye(2), dim=-1)
    teacher = F.normalize(torch.tensor([[3.0, 0.0], [1.0, 0.0], [0.0, 3.0]]), dim=-1)
    student = teacher.clone()

    loss, stats = compute_direct_point_query_support_distill_loss(
        student,
        teacher,
        text,
        temperature=0.25,
    )

    assert loss.item() < 1e-6
    assert stats["valid_ratio"].item() == 1.0
    assert stats["top1_agreement"].item() == 1.0


def test_query_support_distill_penalizes_wrong_primitive_support():
    text = F.normalize(torch.eye(2), dim=-1)
    teacher = F.normalize(torch.tensor([[3.0, 0.0], [1.0, 0.0], [0.0, 3.0]]), dim=-1)
    student = F.normalize(torch.tensor([[0.0, 3.0], [3.0, 0.0], [1.0, 0.0]]), dim=-1)

    loss, stats = compute_direct_point_query_support_distill_loss(
        student,
        teacher,
        text,
        temperature=0.25,
    )

    assert loss.item() > 1.0
    assert stats["top1_agreement"].item() < 1.0


def test_query_support_distill_zscore_amplifies_weak_teacher_support():
    text = F.normalize(torch.eye(2), dim=-1)
    teacher = F.normalize(
        torch.tensor([[1.0, 0.03], [1.0, 0.02], [0.02, 1.0], [0.03, 1.0]]),
        dim=-1,
    )
    student = teacher[[1, 0, 3, 2]]

    plain_loss, _ = compute_direct_point_query_support_distill_loss(
        student,
        teacher,
        text,
        temperature=1.0,
    )
    zscore_loss, _ = compute_direct_point_query_support_distill_loss(
        student,
        teacher,
        text,
        temperature=1.0,
        support_logit_norm="zscore",
    )

    assert zscore_loss.item() > plain_loss.item()


def test_query_support_distill_rejects_unknown_logit_norm():
    text = F.normalize(torch.eye(2), dim=-1)
    teacher = text[[0, 1]]

    try:
        compute_direct_point_query_support_distill_loss(
            teacher,
            teacher,
            text,
            support_logit_norm="bad",
        )
    except ValueError as exc:
        assert "support_logit_norm" in str(exc)
    else:
        raise AssertionError("expected invalid support_logit_norm to raise")
