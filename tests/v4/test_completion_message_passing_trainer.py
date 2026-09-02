from __future__ import annotations

import pytest
import torch

from radio_gs.v4.completion import PartialObjectMembership
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    DifferentiableProjection,
    _clamp_contract,
    _edge_same_instance_bce,
    _heldout_render_losses,
    _message_passing_bypass_control,
    _posterior_to_membership,
    _render_differentiable,
    _source_tree_receipt,
    _target_absent_mass_comparison,
    _target_absent_prediction_mass,
    _unknown_categorical_loss,
    _validate_protocol_split,
)


def _heldout_statistics(rows):
    return {
        "numeric_dtype": "float64",
        "token_count": len(rows),
        "token_statistics": [
            {
                "token_index": index,
                "prediction_mass": prediction_mass,
                "target_mass": target_mass,
            }
            for index, (prediction_mass, target_mass) in enumerate(rows)
        ],
    }


def _runtime(
    labels: torch.Tensor,
    observed: torch.Tensor,
    *,
    token_count: int,
    eligible: torch.Tensor | None = None,
):
    if eligible is None:
        eligible = torch.ones_like(observed, dtype=torch.bool)
    partial = PartialObjectMembership.from_oracle_visibility(
        labels,
        observed,
        token_count=token_count,
        eligible_elements=eligible,
    )
    return {"labels": labels, "partial": partial}


def test_clamp_contract_distinguishes_observed_token_null_and_ineligible():
    runtime = _runtime(
        torch.tensor([0, 0, -1, 1, -1]),
        torch.tensor([True, False, True, False, False]),
        token_count=2,
        eligible=torch.tensor([True, True, True, True, False]),
    )
    mask, probabilities = _clamp_contract(runtime)
    assert mask.tolist() == [True, False, True, False, True]
    torch.testing.assert_close(probabilities[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(probabilities[2], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(probabilities[4], torch.tensor([0.0, 0.0, 1.0]))
    assert bool((probabilities[mask].sum(-1) == 1).all())
    # Clamp construction consumes the already-materialized partial facts, not
    # integer instance labels as a hidden model input.
    runtime["labels"] = torch.tensor([1, -1, 0, 0, 1])
    shuffled_mask, shuffled_probabilities = _clamp_contract(runtime)
    assert torch.equal(shuffled_mask, mask)
    assert torch.equal(shuffled_probabilities, probabilities)


def test_posterior_to_membership_keeps_exact_facts_and_v10_confidence_cap():
    runtime = _runtime(
        torch.tensor([0, 0, -1, 1, -1]),
        torch.tensor([True, False, True, False, False]),
        token_count=2,
        eligible=torch.tensor([True, True, True, True, False]),
    )
    posterior = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.1, 0.1],
            [0.0, 0.0, 1.0],
            [0.2, 0.7, 0.1],
            [0.0, 0.0, 1.0],
        ]
    )
    membership, null = _posterior_to_membership(
        posterior, runtime, completion_confidence_cap=0.95
    )
    torch.testing.assert_close(membership[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(membership[2], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(membership[1], torch.tensor([0.76, 0.095]))
    assert float(null[1]) == pytest.approx(0.145)
    assert float(null[2]) == 1.0
    assert float(null[4]) == 1.0
    eligible = runtime["partial"].eligible_elements
    torch.testing.assert_close(
        membership[eligible].sum(-1) + null[eligible],
        torch.ones(int(eligible.sum())),
    )


def test_message_passing_bypass_is_an_exact_detached_frozen_unary_copy():
    unary = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]], requires_grad=True
    )
    bypass = _message_passing_bypass_control(unary)
    assert torch.equal(bypass, unary)
    assert bypass.data_ptr() != unary.data_ptr()
    assert not bypass.requires_grad


def test_unknown_categorical_loss_is_class_balanced_and_differentiable():
    runtime = _runtime(
        torch.tensor([0, 0, 0, 1, -1, -1]),
        torch.tensor([True, False, False, False, False, False]),
        token_count=2,
    )
    logits = torch.zeros(6, 3, requires_grad=True)
    posterior = torch.softmax(logits, -1)
    loss, audit = _unknown_categorical_loss(posterior, runtime)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(3.0))))
    assert audit == {
        "unknown_element_count": 5,
        "present_categorical_class_count": 3,
        "unknown_object_element_count": 3,
        "unknown_null_element_count": 2,
    }
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad[0] == 0).all())  # observed fact is not a target loss.


def test_edge_same_instance_bce_treats_background_as_no_instance():
    runtime = _runtime(
        torch.tensor([0, 0, 1, -1]),
        torch.zeros(4, dtype=torch.bool),
        token_count=2,
    )
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 2]])
    logits = torch.zeros(4, requires_grad=True)
    loss, audit = _edge_same_instance_bce(logits, edges, runtime)
    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert audit["same_retained_instance_edge_count"] == 1
    assert audit["different_or_null_edge_count"] == 3
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0]) < 0
    assert bool((logits.grad[1:] > 0).all())


def test_gpu_style_differentiable_render_preserves_invalid_denominator_and_fp_loss():
    projection = DifferentiableProjection(
        numerator_element_ids=torch.tensor([0, 1]),
        numerator_pixel_ids=torch.tensor([0, 1]),
        numerator_weights=torch.ones(2),
        # Pixel one has a second projected but invalid carrier contributor.
        denominator=torch.tensor([1.0, 2.0]),
        height=1,
        width=2,
    )
    membership = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6]], requires_grad=True
    )
    rendered = _render_differentiable(membership, projection)
    torch.testing.assert_close(
        rendered,
        torch.tensor([[[0.8, 0.2], [0.2, 0.3]]]),
    )
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    iou_loss, absent_fp_loss, audit = _heldout_render_losses(
        membership, [projection], [target]
    )
    assert float(iou_loss) > 0
    assert float(absent_fp_loss) == pytest.approx(0.25)
    assert audit["target_present_token_count"] == 1
    assert audit["target_absent_token_count"] == 1
    (iou_loss + absent_fp_loss).backward()
    assert membership.grad is not None
    assert torch.isfinite(membership.grad).all()
    assert float(membership.grad[:, 1].sum()) > 0


def test_target_absent_continuous_mass_detects_reduction_when_support_count_cannot():
    records = [
        {
            "scene_id": "scene_a",
            "heldout_pixel_count": 100,
            "frozen": {
                "soft_iou_sufficient_statistics": {
                    "heldout_2d": _heldout_statistics(
                        [(40.0, 50.0), (10.0, 0.0), (0.0, 0.0)]
                    )
                }
            },
            "final": {
                "soft_iou_sufficient_statistics": {
                    "heldout_2d": _heldout_statistics(
                        [(42.0, 50.0), (4.0, 0.0), (0.0, 0.0)]
                    )
                }
            },
        },
        {
            "scene_id": "scene_b",
            "heldout_pixel_count": 50,
            "frozen": {
                "soft_iou_sufficient_statistics": {
                    "heldout_2d": _heldout_statistics(
                        [(5.0, 0.0), (15.0, 20.0)]
                    )
                }
            },
            "final": {
                "soft_iou_sufficient_statistics": {
                    "heldout_2d": _heldout_statistics(
                        [(2.0, 0.0), (16.0, 20.0)]
                    )
                }
            },
        },
    ]
    frozen = _target_absent_prediction_mass(records, "frozen")
    final = _target_absent_prediction_mass(records, "final")
    assert frozen["total_prediction_mass"] == 15.0
    assert frozen["target_absent_scene_token_count"] == 3
    assert frozen["mean_prediction_mass_per_target_absent_scene_token"] == 5.0
    assert frozen["heldout_pixel_token_normalizer"] == 250
    assert frozen[
        "prediction_mass_per_target_absent_heldout_pixel_token"
    ] == pytest.approx(0.06)
    assert final["total_prediction_mass"] == 6.0
    # Strict-positive support does not need to change for continuous mass to
    # record the scientifically relevant suppression.
    assert frozen["strict_positive_support_scene_token_count"] == 2
    assert final["strict_positive_support_scene_token_count"] == 2
    comparison = _target_absent_mass_comparison(
        {
            "frozen_aligned_pointwise": frozen,
            "message_passing_completion": final,
        }
    )
    normalized = comparison["metrics"][
        "prediction_mass_per_target_absent_heldout_pixel_token"
    ]
    assert normalized["final_minus_frozen"] == pytest.approx(-0.036)
    assert normalized["relative_reduction_fraction"] == pytest.approx(0.6)


def test_protocol_split_keeps_formal_12_4_and_caps_bounded_development():
    train = {f"scene{index:04d}_00" for index in range(12)}
    validation = {f"scene{index:04d}_00" for index in range(12, 16)}
    _validate_protocol_split(
        protocol="formal16",
        training_scene_ids=train,
        validation_scene_ids=validation,
        cohort_training_scene_ids=train,
        cohort_validation_scene_ids=validation,
        epoch_count=40,
        allow_bounded_dev_protocol=False,
    )
    with pytest.raises(ValueError, match="complete frozen 12/4"):
        _validate_protocol_split(
            protocol="formal16",
            training_scene_ids=set(list(train)[:2]),
            validation_scene_ids=set(list(validation)[:1]),
            cohort_training_scene_ids=train,
            cohort_validation_scene_ids=validation,
            epoch_count=2,
            allow_bounded_dev_protocol=False,
        )
    with pytest.raises(PermissionError, match="explicit"):
        _validate_protocol_split(
            protocol="bounded_dev",
            training_scene_ids=set(list(train)[:2]),
            validation_scene_ids=set(list(validation)[:1]),
            cohort_training_scene_ids=train,
            cohort_validation_scene_ids=validation,
            epoch_count=2,
            allow_bounded_dev_protocol=False,
        )
    with pytest.raises(ValueError, match="capped"):
        _validate_protocol_split(
            protocol="bounded_dev",
            training_scene_ids=set(list(train)[:3]),
            validation_scene_ids=set(list(validation)[:1]),
            cohort_training_scene_ids=train,
            cohort_validation_scene_ids=validation,
            epoch_count=2,
            allow_bounded_dev_protocol=True,
        )


def test_explicit_source_tree_receipt_binds_frozen_and_new_implementations():
    receipt = _source_tree_receipt()
    assert len(receipt["source_tree_sha256"]) == 64
    paths = {row["path"] for row in receipt["files"]}
    assert "radio_gs/v4/completion/message_passing.py" in paths
    assert "radio_gs/v4/training/train_scannet_completion_oracle.py" in paths
    assert (
        "radio_gs/v4/training/train_scannet_completion_message_passing.py" in paths
    )
