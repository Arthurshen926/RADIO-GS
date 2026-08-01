import pytest
import torch
import torch.nn.functional as F

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_direct_point_query_logit_distill_loss,
    compute_direct_point_query_support_distill_loss,
    compute_independent_normalized_cosine_response_smooth_l1_loss,
    compute_scene_wise_text_response_profile_ranking_loss,
)


def test_independent_cosine_response_loss_is_strictly_zero_for_matching_descriptors():
    teacher = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 1.0]])
    student = teacher.clone().requires_grad_(True)
    text = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])

    loss = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_independent_cosine_response_loss_is_positive_for_misaligned_descriptors():
    teacher = torch.eye(3)
    student = teacher[[1, 2, 0]]
    text = torch.eye(3)

    loss = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text,
    )

    assert loss.item() > 0.0


def test_independent_cosine_response_loss_is_invariant_to_shared_query_permutation():
    teacher = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 1.0]])
    student = torch.tensor([[0.5, 2.0, -0.5], [-1.0, 1.5, 0.5]])
    text = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 1.0],
            [-1.0, 1.0, 0.5],
            [0.5, -1.0, 2.0],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])

    reference = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text,
    )
    permuted = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text[permutation],
    )

    torch.testing.assert_close(permuted, reference)


def test_independent_cosine_response_loss_keeps_teacher_and_text_bank_frozen():
    student = torch.tensor([[1.0, 0.5]], requires_grad=True)
    teacher = torch.tensor([[0.5, 1.0]], requires_grad=True)
    text = torch.eye(2, requires_grad=True)

    loss = compute_independent_normalized_cosine_response_smooth_l1_loss(
        student,
        teacher,
        text,
    )
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert text.grad is None


@pytest.mark.parametrize(
    ("student", "teacher", "text"),
    [
        (torch.empty(0, 2), torch.empty(0, 2), torch.eye(2)),
        (torch.eye(2), torch.eye(2), torch.empty(0, 2)),
        (torch.empty(2, 0), torch.empty(2, 0), torch.empty(1, 0)),
        (torch.ones(2, 3), torch.ones(3, 3), torch.ones(1, 3)),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(1, 2)),
        (torch.ones(2, 3, 1), torch.ones(2, 3, 1), torch.ones(1, 3)),
        (torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2), torch.ones(1, 2)),
        (torch.ones(1, 2), torch.tensor([[float("inf"), 0.0]]), torch.ones(1, 2)),
        (torch.ones(1, 2), torch.ones(1, 2), torch.tensor([[0.0, float("-inf")]])),
        (
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 2),
            torch.ones(1, 2),
        ),
    ],
)
def test_independent_cosine_response_loss_rejects_invalid_inputs(
    student,
    teacher,
    text,
):
    with pytest.raises((TypeError, ValueError)):
        compute_independent_normalized_cosine_response_smooth_l1_loss(
            student,
            teacher,
            text,
        )


def test_scene_wise_profile_ranking_loss_is_zero_for_matching_descriptors():
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.2],
                [0.7, 0.6, 0.1],
                [0.0, 1.0, 0.3],
                [-0.4, 0.8, 0.2],
            ]
        ),
        dim=-1,
    )
    student = teacher.clone().requires_grad_(True)
    text = torch.eye(3, requires_grad=True)

    loss, stats = compute_scene_wise_text_response_profile_ranking_loss(
        student,
        teacher.requires_grad_(True),
        text,
        scene_ids=["scene_a", "scene_a", "scene_b", "scene_b"],
    )

    assert loss.item() == 0.0
    assert stats["profile_loss"].item() == 0.0
    assert stats["ranking_loss"].item() == 0.0
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert torch.equal(student.grad, torch.zeros_like(student))
    assert teacher.grad is None
    assert text.grad is None


def test_scene_wise_profile_ranking_loss_penalizes_scene_local_order_damage():
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
                [-0.8, 0.6, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        dim=-1,
    )
    text = torch.eye(3)
    scenes = torch.tensor([0, 0, 0, 1, 1, 1])
    matching, _ = compute_scene_wise_text_response_profile_ranking_loss(
        teacher,
        teacher,
        text,
        scenes,
    )
    damaged = teacher.clone()
    damaged[:3] = damaged[torch.tensor([2, 1, 0])]
    loss, stats = compute_scene_wise_text_response_profile_ranking_loss(
        damaged,
        teacher,
        text,
        scenes,
    )

    assert matching.item() == 0.0
    assert loss.item() > matching.item()
    assert stats["profile_loss"].item() > 0.0
    assert stats["ranking_loss"].item() > 0.0
    assert stats["valid_scene_count"].item() == 2


def test_scene_wise_centered_profile_is_invariant_to_scene_query_offset():
    teacher_responses = torch.tensor([-0.6, -0.1, 0.4])
    student_responses = teacher_responses + 0.2
    teacher = torch.stack(
        (teacher_responses, (1.0 - teacher_responses.square()).sqrt()),
        dim=-1,
    )
    student = torch.stack(
        (student_responses, (1.0 - student_responses.square()).sqrt()),
        dim=-1,
    )

    loss, stats = compute_scene_wise_text_response_profile_ranking_loss(
        student,
        teacher,
        torch.tensor([[1.0, 0.0]]),
        scene_ids=["scene"] * 3,
        ranking_weight=0.0,
    )

    assert loss.item() < 1e-12
    assert stats["profile_loss"].item() < 1e-12


def test_scene_wise_loss_handles_teacher_ties_and_single_row_scenes():
    teacher = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    student = teacher.clone()
    student[1] = torch.tensor([0.0, 1.0])
    loss, stats = compute_scene_wise_text_response_profile_ranking_loss(
        student,
        teacher,
        torch.tensor([[1.0, 0.0]]),
        scene_ids=["tie", "tie", "singleton"],
    )

    assert loss.item() > 0.0
    assert stats["profile_loss"].item() == 0.0
    assert stats["ranking_loss"].item() > 0.0
    assert stats["valid_scene_count"].item() == 1
    assert stats["valid_profile_count"].item() == 0

    singleton_student = teacher.clone().requires_grad_(True)
    singleton_loss, singleton_stats = compute_scene_wise_text_response_profile_ranking_loss(
        singleton_student,
        teacher,
        torch.eye(2),
        scene_ids=["a", "b", "c"],
    )
    assert singleton_loss.item() == 0.0
    assert singleton_stats["ranking_unit_count"].item() == 0
    singleton_loss.backward()
    assert torch.equal(singleton_student.grad, torch.zeros_like(singleton_student))


def test_scene_wise_profile_ranking_loss_has_finite_student_gradients():
    torch.manual_seed(7)
    student = torch.randn(7, 5, requires_grad=True)
    teacher = torch.randn(7, 5)
    text = torch.randn(4, 5)
    loss, _ = compute_scene_wise_text_response_profile_ranking_loss(
        student,
        teacher,
        text,
        scene_ids=[0, 0, 0, 1, 1, 1, 1],
    )

    loss.backward()
    assert loss.item() > 0.0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize(
    ("scene_ids", "kwargs"),
    [
        (["only_one"], {}),
        (torch.tensor([0.0, float("nan")]), {}),
        (["a", "a"], {"ranking_temperature": 0.0}),
        (["a", "a"], {"profile_weight": 0.0, "ranking_weight": 0.0}),
    ],
)
def test_scene_wise_profile_ranking_loss_rejects_invalid_grouping_or_parameters(
    scene_ids,
    kwargs,
):
    with pytest.raises((TypeError, ValueError)):
        compute_scene_wise_text_response_profile_ranking_loss(
            torch.eye(2),
            torch.eye(2),
            torch.eye(2),
            scene_ids=scene_ids,
            **kwargs,
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
