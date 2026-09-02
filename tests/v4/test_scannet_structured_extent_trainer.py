from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from radio_gs.v4.completion import (
    PartialObjectMembership,
    TokenConditionedStructuredExtent,
)
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    DifferentiableProjection,
)
from radio_gs.v4.training.train_scannet_structured_extent import (
    CHECKPOINT_SCHEMA,
    FIXED_ITERATION_COUNT,
    REPORT_SCHEMA,
    _balanced_token_edge_bce,
    _clip_structured_gradient_groups,
    _heldout_present_and_absent_losses,
    _log_full_mass_smooth_l1_loss,
    _object_equal_3d_soft_iou_loss,
    _sample_token_conditioned_edge_pairs,
    _structured_edge_supervision_loss,
)


def _runtime(labels, observed=None, *, token_count=2, edge_index=None):
    labels = torch.as_tensor(labels, dtype=torch.long)
    if observed is None:
        observed = torch.zeros(labels.numel(), dtype=torch.bool)
    partial = PartialObjectMembership.from_oracle_visibility(
        labels,
        torch.as_tensor(observed, dtype=torch.bool),
        token_count=token_count,
    )
    return {
        "labels": labels,
        "partial": partial,
        "edge_index": (
            torch.as_tensor(edge_index, dtype=torch.long)
            if edge_index is not None
            else torch.empty(2, 0, dtype=torch.long)
        ),
        "payload": {"scene_id": "synthetic"},
    }


def test_structured_schemas_are_independent_and_fixed_to_two_iterations():
    assert REPORT_SCHEMA.endswith(".v2")
    assert CHECKPOINT_SCHEMA.endswith(".v2")
    assert FIXED_ITERATION_COUNT == 2
    assert "message_passing" not in REPORT_SCHEMA


def test_object_equal_3d_soft_iou_is_token_macro_and_differentiable():
    runtime = _runtime([0, 0, 1, 1, -1])
    membership = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.2, 0.2],
        ],
        requires_grad=True,
    )
    loss, audit = _object_equal_3d_soft_iou_loss(membership, runtime)
    expected_per_token_iou = 1.5 / (1.7 + 2.0 - 1.5)
    assert audit["target_present_token_count"] == 2
    assert audit["object_equal_soft_iou"] == pytest.approx(expected_per_token_iou)
    assert float(loss) == pytest.approx(1 - expected_per_token_iou)
    loss.backward()
    assert membership.grad is not None
    assert torch.isfinite(membership.grad).all()


def test_log_full_mass_smooth_l1_uses_object_equal_log_targets():
    runtime = _runtime([0, 0, 1, 1, -1])
    predicted = torch.tensor([2.0, 2.0], requires_grad=True)
    loss, audit = _log_full_mass_smooth_l1_loss(predicted, runtime)
    assert float(loss) == 0
    assert audit["target_full_mass_mean"] == 2
    loss.backward()
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_heldout_loss_separates_present_macro_from_continuous_absent_rms_mass():
    runtime = _runtime([0, 1])
    runtime["heldout_projections"] = [
        DifferentiableProjection(
            numerator_element_ids=torch.tensor([0, 1]),
            numerator_pixel_ids=torch.tensor([0, 1]),
            numerator_weights=torch.ones(2),
            denominator=torch.ones(2),
            height=1,
            width=2,
        )
    ]
    runtime["payload"]["heldout_mesh_target_rasters"] = [
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    ]
    membership = torch.tensor(
        [[0.8, 0.2], [0.1, 0.4]], requires_grad=True
    )
    present, absent_rms, absent_mass, audit = (
        _heldout_present_and_absent_losses(membership, runtime)
    )
    assert audit["target_present_token_count"] == 1
    assert audit["target_absent_token_count"] == 1
    assert float(absent_mass) == pytest.approx(float(torch.log1p(torch.tensor(0.6))))
    assert float(absent_rms) == pytest.approx(float(torch.sqrt(torch.tensor(0.1))))
    assert audit["continuous_absent_mean_probability"] == pytest.approx(0.3)
    (present + absent_rms + absent_mass).backward()
    assert membership.grad is not None
    assert float(membership.grad[:, 1].sum()) > 0


def test_token_conditioned_edge_sampling_is_balanced_deterministic_and_target_only():
    edges = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ]
    )
    runtime = _runtime([0, 0, 1, 1, -1], edge_index=edges)
    first = _sample_token_conditioned_edge_pairs(
        runtime, maximum_pairs_per_class=8, seed=31
    )
    repeated = _sample_token_conditioned_edge_pairs(
        runtime, maximum_pairs_per_class=8, seed=31
    )
    assert torch.equal(first["edge_ids"], repeated["edge_ids"])
    assert torch.equal(first["token_ids"], repeated["token_ids"])
    assert torch.equal(first["target"], repeated["target"])
    assert first["audit"]["positive_pair_count"] > 0
    assert first["audit"]["negative_pair_count"] > 0
    source, destination = edges[:, first["edge_ids"]]
    expected = (
        (runtime["labels"][source] == first["token_ids"])
        & (runtime["labels"][destination] == first["token_ids"])
    ).float()
    assert torch.equal(first["target"], expected)


def test_balanced_token_edge_bce_does_not_follow_class_frequency():
    logits = torch.zeros(5, requires_grad=True)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    loss = _balanced_token_edge_bce(logits, target)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(2.0))))
    loss.backward()
    assert float(logits.grad[0]) < 0
    assert bool((logits.grad[1:] > 0).all())


def test_shared_edge_control_uses_query_free_edge_labels_without_token_conflict():
    edges = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ]
    )
    runtime = _runtime([0, 0, 1, 1, -1], edge_index=edges)
    model = TokenConditionedStructuredExtent(
        embedding_dimension=8,
        edge_hidden_dimension=8,
        mode="shared_edge_plus_mass",
    )
    logits = torch.zeros(edges.shape[1], requires_grad=True)
    loss, audit = _structured_edge_supervision_loss(
        model,
        SimpleNamespace(base_edge_logits=logits),
        runtime,
        None,
        device=torch.device("cpu"),
    )
    assert audit["policy"] == "query_free_same_retained_instance_edge_bce"
    assert audit["same_retained_instance_edge_count"] == 4
    assert audit["different_or_null_edge_count"] == 4
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(2.0))))
    loss.backward()
    assert logits.grad is not None


def test_gradient_clipping_keeps_auxiliary_mass_norm_out_of_posterior_scaling():
    first = TokenConditionedStructuredExtent(
        embedding_dimension=8, edge_hidden_dimension=8,
        mode="token_conditioned_edge_plus_mass_bypass",
    )
    second = TokenConditionedStructuredExtent(
        embedding_dimension=8, edge_hidden_dimension=8,
        mode="token_conditioned_edge_plus_mass_bypass",
    )
    second.load_state_dict(first.state_dict())

    def assign(model, mass_scale):
        mass_ids = {
            id(parameter)
            for module in (model.mass_encoder, model.mass_head)
            for parameter in module.parameters()
        }
        for parameter in model.parameters():
            scale = mass_scale if id(parameter) in mass_ids else 1.0
            parameter.grad = torch.full_like(parameter, scale)
        return mass_ids

    first_mass_ids = assign(first, 1.0)
    assign(second, 1e6)
    first_norms = _clip_structured_gradient_groups(first, 0.5)
    second_norms = _clip_structured_gradient_groups(second, 0.5)
    assert float(second_norms["mass"]) > float(first_norms["mass"])
    first_posterior = [
        parameter.grad
        for parameter in first.parameters()
        if id(parameter) not in first_mass_ids
    ]
    second_mass_ids = {
        id(parameter)
        for module in (second.mass_encoder, second.mass_head)
        for parameter in module.parameters()
    }
    second_posterior = [
        parameter.grad
        for parameter in second.parameters()
        if id(parameter) not in second_mass_ids
    ]
    for left, right in zip(first_posterior, second_posterior):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
