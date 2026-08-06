from __future__ import annotations

import torch

from radio_gs.losses.surface_region_codebook_loss import (
    balanced_latent_relation_loss,
    gauge_aware_permutation_set_matching_loss,
    latent_query_max_responses,
    latent_query_responses,
    permutation_set_matching_loss,
    scene_listwise_and_hard_negative_loss,
    uniform_slot_prior_loss,
)


def test_permutation_matching_is_order_invariant_and_differentiable() -> None:
    teacher_token = torch.eye(3, 4)[None]
    teacher_descriptor = torch.eye(3, 5)[None]
    predicted_token = teacher_token[:, [2, 0, 1]].clone().requires_grad_(True)
    predicted_descriptor = teacher_descriptor[:, [2, 0, 1]].clone().requires_grad_(True)
    loss, assignment = permutation_set_matching_loss(
        predicted_token,
        predicted_descriptor,
        teacher_token,
        teacher_descriptor,
        torch.ones(1, 3, dtype=torch.bool),
        token_weight=0.25,
    )
    assert torch.allclose(loss, torch.tensor(0.0))
    assert assignment.tolist() == [[1, 2, 0]]
    loss.backward()
    assert predicted_token.grad is not None
    assert predicted_descriptor.grad is not None


def test_latent_response_respects_mask_and_prior() -> None:
    descriptors = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
    )
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    first = latent_query_responses(
        descriptors,
        text,
        mask=mask,
        temperature=0.05,
    )
    second = latent_query_responses(
        descriptors,
        text,
        mask=mask,
        priors=torch.tensor([[0.9, 0.1, 0.0]]),
        temperature=0.05,
    )
    assert first.shape == (1, 2)
    assert second[0, 0] > first[0, 0]
    assert second[0, 1] < first[0, 1]


def test_hard_max_response_contains_singleton_fallback() -> None:
    fallback = torch.tensor([[[1.0, 0.0]]])
    expanded = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    singleton = latent_query_max_responses(fallback, text)
    candidate = latent_query_max_responses(expanded, text)
    assert torch.equal(singleton, torch.tensor([[1.0, 0.0]]))
    assert bool((candidate >= singleton).all())


def test_gauge_aware_matching_supervises_token_norm() -> None:
    predicted_token = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    teacher_token = torch.tensor([[[2.0, 0.0]]])
    descriptor = torch.tensor([[[1.0, 0.0]]])
    loss, assignment = gauge_aware_permutation_set_matching_loss(
        predicted_token,
        descriptor,
        teacher_token,
        descriptor,
        torch.ones(1, 1, dtype=torch.bool),
        token_direction_weight=0.25,
        token_log_norm_weight=0.25,
    )
    assert assignment.tolist() == [[0]]
    assert loss > 0
    loss.backward()
    assert predicted_token.grad is not None
    assert abs(float(predicted_token.grad[0, 0, 0])) > 0


def test_scene_losses_are_zero_for_identical_responses() -> None:
    teacher = torch.tensor(
        [[0.8, 0.2], [0.4, 0.5], [0.1, 0.9], [0.7, 0.1]]
    )
    listwise, margin = scene_listwise_and_hard_negative_loss(
        teacher,
        teacher,
        ["a", "a", "b", "b"],
    )
    assert abs(float(listwise)) < 1e-6
    assert abs(float(margin)) < 1e-6


def test_uniform_slot_prior_loss_penalizes_collapse() -> None:
    uniform = uniform_slot_prior_loss(torch.full((2, 3), 1.0 / 3.0))
    collapsed = uniform_slot_prior_loss(
        torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]])
    )
    assert abs(float(uniform)) < 1e-6
    assert collapsed > uniform


def test_balanced_relation_is_zero_for_matching_latent_sets() -> None:
    descriptors = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.0, 1.0], [0.2, 0.8]],
            [[-1.0, 0.0], [-0.8, 0.2]],
        ]
    )
    loss = balanced_latent_relation_loss(
        descriptors,
        descriptors,
        torch.ones(3, 2, dtype=torch.bool),
        ["scene", "scene", "scene"],
    )
    assert abs(float(loss)) < 1e-6
