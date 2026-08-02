from __future__ import annotations

import pytest
import torch

from radio_gs.querying.evidence_scorer import (
    registered_forward_beta_balanced_residual_observation,
)


def _balanced_case(**overrides):
    arguments = {
        "gaussian_ids": torch.tensor([0, 1, 1, 1]),
        "pixel_ids": torch.tensor([0, 1, 2, 3]),
        "contribution_weights": torch.tensor([0.8, 0.8, 0.8, 0.8]),
        "capability_valid": torch.tensor([True, True]),
        "field_prior": torch.tensor([0.6, 0.4]),
        "primitive_reliability": torch.tensor([0.8, 0.5]),
        "primitive_coverage": torch.tensor([0.75, 0.5]),
        "positive_pixel_mask": torch.tensor([True, False, False, False]),
        "negative_pixel_mask": torch.tensor([False, True, True, True]),
        "labeled_pixel_mask": torch.tensor([True, True, True, True]),
        "all_pixel_mask": torch.tensor([True, True, True, True]),
        "anchor_threshold": 0.5,
    }
    arguments.update(overrides)
    return registered_forward_beta_balanced_residual_observation(**arguments)


def test_v2_globally_balances_expected_counts_not_scribble_area() -> None:
    _, diagnostics = _balanced_case()

    assert diagnostics.raw_positive_expected_count.sum().item() == pytest.approx(0.8)
    assert diagnostics.raw_negative_expected_count.sum().item() == pytest.approx(2.4)
    assert diagnostics.positive_expected_count.sum().item() == pytest.approx(1.6)
    assert diagnostics.negative_expected_count.sum().item() == pytest.approx(1.6)
    assert diagnostics.positive_class_balance_scale == pytest.approx(2.0)
    assert diagnostics.negative_class_balance_scale == pytest.approx(2.0 / 3.0)


def test_v2_is_invariant_to_sparse_hit_order() -> None:
    observation, diagnostics = _balanced_case()
    permutation = torch.tensor([3, 0, 2, 1])
    reordered, reordered_diagnostics = _balanced_case(
        gaussian_ids=torch.tensor([0, 1, 1, 1])[permutation],
        pixel_ids=torch.tensor([0, 1, 2, 3])[permutation],
        contribution_weights=torch.tensor([0.8, 0.8, 0.8, 0.8])[permutation],
    )

    torch.testing.assert_close(reordered.values, observation.values)
    torch.testing.assert_close(reordered.confidence, observation.confidence)
    torch.testing.assert_close(
        reordered_diagnostics.fused_probability,
        diagnostics.fused_probability,
    )


def test_v2_prior_precision_is_bounded_and_semantic_is_primary_for_residuals() -> None:
    observation, diagnostics = _balanced_case(
        contribution_weights=torch.tensor([0.1, 0.1, 0.1, 0.1]),
        anchor_threshold=1.0,
    )

    expected_concentration = 1.0 + torch.tensor([0.8, 0.5]).double() * torch.tensor(
        [0.75, 0.5]
    ).double()
    torch.testing.assert_close(
        diagnostics.field_prior_concentration,
        expected_concentration,
    )
    assert bool((diagnostics.field_prior_concentration >= 1.0).all())
    assert bool((diagnostics.field_prior_concentration <= 2.0).all())
    assert bool((diagnostics.residual_evidence_concentration < 1.0).all())
    assert bool((observation.confidence < 0.5).all())
    assert diagnostics.fused_probability[0] > 0.6
    assert diagnostics.fused_probability[0] < 1.0
    assert diagnostics.fused_probability[1] < 0.4
    assert diagnostics.fused_probability[1] > 0.0


def test_v2_extreme_direct_evidence_creates_bipolar_strong_anchors() -> None:
    observation, diagnostics = _balanced_case(anchor_threshold=0.2)

    assert diagnostics.positive_anchor_mask.tolist() == [True, False]
    assert diagnostics.negative_anchor_mask.tolist() == [False, True]
    torch.testing.assert_close(observation.values, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(observation.confidence, torch.ones(2))
    torch.testing.assert_close(
        diagnostics.fused_probability, torch.tensor([1.0, 0.0]).double()
    )


def test_v2_class_balance_does_not_manufacture_anchor_from_tiny_tail() -> None:
    observation, diagnostics = _balanced_case(
        contribution_weights=torch.tensor([0.01, 0.8, 0.8, 0.8]),
        anchor_threshold=0.2,
    )

    assert diagnostics.positive_class_balance_scale > 100.0
    assert diagnostics.positive_anchor_mask.tolist() == [False, False]
    assert diagnostics.negative_anchor_mask.tolist() == [False, True]
    assert observation.values[0].abs().item() < observation.confidence[0].item() + 1e-7
    assert observation.confidence[0].item() < 0.5


def test_v2_sign_swap_is_symmetric() -> None:
    observation, diagnostics = _balanced_case(anchor_threshold=1.0)
    swapped, swapped_diagnostics = _balanced_case(
        field_prior=1.0 - torch.tensor([0.6, 0.4]),
        positive_pixel_mask=torch.tensor([False, True, True, True]),
        negative_pixel_mask=torch.tensor([True, False, False, False]),
        anchor_threshold=1.0,
    )

    torch.testing.assert_close(swapped.values, -observation.values)
    torch.testing.assert_close(swapped.confidence, observation.confidence)
    torch.testing.assert_close(
        swapped_diagnostics.fused_probability,
        1.0 - diagnostics.fused_probability,
    )


def test_v2_no_observable_evidence_exactly_degenerates_to_field() -> None:
    prior = torch.tensor([0.2, 0.8])
    observation, diagnostics = _balanced_case(
        gaussian_ids=torch.empty(0, dtype=torch.long),
        pixel_ids=torch.empty(0, dtype=torch.long),
        contribution_weights=torch.empty(0),
        field_prior=prior,
        anchor_threshold=0.2,
    )

    torch.testing.assert_close(observation.values, torch.zeros(2))
    torch.testing.assert_close(observation.confidence, torch.zeros(2))
    torch.testing.assert_close(diagnostics.fused_probability.float(), prior)
    assert diagnostics.positive_class_balance_scale == 0.0
    assert diagnostics.negative_class_balance_scale == 0.0


def test_v2_fails_closed_when_only_one_class_is_observable() -> None:
    with pytest.raises(ValueError, match="observable positive and negative"):
        _balanced_case(
            gaussian_ids=torch.tensor([0]),
            pixel_ids=torch.tensor([0]),
            contribution_weights=torch.tensor([0.8]),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"primitive_reliability": torch.tensor([0.5])}, "align"),
        ({"primitive_coverage": torch.tensor([0.5, float("nan")])}, "finite"),
        (
            {
                "capability_valid": torch.tensor([True, False]),
                "primitive_reliability": torch.tensor([0.8, 0.5]),
            },
            "invalid primitive rows",
        ),
        ({"anchor_threshold": 0.0}, "anchor_threshold"),
    ],
)
def test_v2_fails_closed_on_malformed_precision_or_anchor_inputs(
    override, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _balanced_case(**override)
