from __future__ import annotations

import pytest
import torch

from radio_gs.querying.evidence_scorer import (
    registered_forward_beta_observation,
)


def _two_pixel_mixture(
    *,
    prior: torch.Tensor | None = None,
    positive: torch.Tensor | None = None,
    negative: torch.Tensor | None = None,
):
    positive = (
        torch.tensor([True, False]) if positive is None else positive.bool()
    )
    negative = (
        torch.tensor([False, True]) if negative is None else negative.bool()
    )
    return registered_forward_beta_observation(
        gaussian_ids=torch.tensor([0, 1, 0, 1]),
        pixel_ids=torch.tensor([0, 0, 1, 1]),
        contribution_weights=torch.tensor([0.25, 0.75, 0.5, 0.5]),
        capability_valid=torch.tensor([True, True]),
        field_prior=(torch.tensor([0.2, 0.8]) if prior is None else prior),
        positive_pixel_mask=positive,
        negative_pixel_mask=negative,
        labeled_pixel_mask=positive | negative,
        all_pixel_mask=torch.ones(2, dtype=torch.bool),
    )


def test_forward_beta_expected_counts_match_one_mixture_e_step() -> None:
    evidence, diagnostics = _two_pixel_mixture()

    expected_positive = torch.tensor(
        [0.25 * 0.2 / 0.65, 0.75 * 0.8 / 0.65],
        dtype=torch.float64,
    )
    expected_negative = torch.tensor(
        [0.5 * 0.8 / 0.5, 0.5 * 0.2 / 0.5],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        diagnostics.positive_expected_count,
        expected_positive,
    )
    torch.testing.assert_close(
        diagnostics.negative_expected_count,
        expected_negative,
    )
    # Each labeled pixel contributes its capability-valid accumulated alpha.
    torch.testing.assert_close(
        diagnostics.positive_expected_count.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.negative_expected_count.sum(),
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.forward_probability_before,
        torch.tensor([0.65, 0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.labeled_coverage,
        torch.ones(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        evidence.confidence,
        torch.ones(2),
    )
    torch.testing.assert_close(
        diagnostics.fused_probability,
        diagnostics.observation_probability,
    )
    assert diagnostics.nll_after < diagnostics.nll_before
    assert diagnostics.protocol_status == (
        "method_primitive_no_independent_protocol_claim"
    )


def test_forward_beta_no_observation_strictly_restores_field_prior() -> None:
    prior = torch.tensor([0.12345679, 0.8765432])
    empty = torch.zeros(2, dtype=torch.bool)
    evidence, diagnostics = registered_forward_beta_observation(
        gaussian_ids=torch.tensor([0, 1]),
        pixel_ids=torch.tensor([0, 1]),
        contribution_weights=torch.tensor([0.7, 0.4]),
        capability_valid=torch.tensor([True, True]),
        field_prior=prior,
        positive_pixel_mask=empty,
        negative_pixel_mask=empty,
        labeled_pixel_mask=empty,
        all_pixel_mask=torch.ones(2, dtype=torch.bool),
    )

    assert torch.equal(evidence.values, torch.zeros_like(evidence.values))
    assert torch.equal(evidence.confidence, torch.zeros_like(evidence.confidence))
    assert torch.equal(diagnostics.labeled_expected_count, torch.zeros(2).double())
    assert torch.equal(diagnostics.labeled_coverage, torch.zeros(2).double())
    assert torch.equal(diagnostics.fused_probability, prior.double())
    assert diagnostics.nll_before == 0.0
    assert diagnostics.nll_after == 0.0
    assert diagnostics.observable_labeled_alpha_mass == 0.0


def test_sparse_scribble_confidence_is_not_multiplied_away_by_coverage() -> None:
    pixel_count = 10
    positive = torch.zeros(pixel_count, dtype=torch.bool)
    positive[0] = True
    negative = torch.zeros_like(positive)
    evidence, diagnostics = registered_forward_beta_observation(
        gaussian_ids=torch.zeros(pixel_count, dtype=torch.long),
        pixel_ids=torch.arange(pixel_count),
        contribution_weights=torch.ones(pixel_count),
        capability_valid=torch.tensor([True]),
        field_prior=torch.tensor([0.2]),
        positive_pixel_mask=positive,
        negative_pixel_mask=negative,
        labeled_pixel_mask=positive,
        all_pixel_mask=torch.ones(pixel_count, dtype=torch.bool),
    )

    torch.testing.assert_close(
        diagnostics.labeled_expected_count,
        torch.tensor([1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.labeled_coverage,
        torch.tensor([0.1], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.beta_confidence,
        torch.tensor([0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.effective_confidence,
        torch.tensor([0.55], dtype=torch.float64),
    )
    assert diagnostics.effective_confidence[0] >= diagnostics.beta_confidence[0]
    # The rejected multiplicative form would have produced only 0.05.
    assert diagnostics.effective_confidence[0] > (
        diagnostics.beta_confidence[0] * diagnostics.labeled_coverage[0]
    )
    torch.testing.assert_close(evidence.values, torch.tensor([0.55]))


def test_full_mask_is_strong_even_from_saturated_adversarial_prior() -> None:
    positive = torch.ones(3, dtype=torch.bool)
    negative = torch.zeros_like(positive)
    evidence, diagnostics = registered_forward_beta_observation(
        gaussian_ids=torch.zeros(3, dtype=torch.long),
        pixel_ids=torch.arange(3),
        contribution_weights=torch.tensor([0.4, 0.8, 1.0]),
        capability_valid=torch.tensor([True]),
        field_prior=torch.tensor([0.0]),
        positive_pixel_mask=positive,
        negative_pixel_mask=negative,
        labeled_pixel_mask=positive,
        all_pixel_mask=torch.ones(3, dtype=torch.bool),
    )

    torch.testing.assert_close(
        diagnostics.positive_expected_count,
        torch.tensor([2.2], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.labeled_coverage,
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(evidence.confidence, torch.ones(1))
    torch.testing.assert_close(evidence.values, torch.ones(1))
    torch.testing.assert_close(
        diagnostics.fused_probability,
        torch.ones(1, dtype=torch.float64),
    )


def test_forward_beta_is_sign_symmetric() -> None:
    positive_evidence, positive_diagnostics = _two_pixel_mixture()
    negative_evidence, negative_diagnostics = _two_pixel_mixture(
        prior=torch.tensor([0.8, 0.2]),
        positive=torch.tensor([False, True]),
        negative=torch.tensor([True, False]),
    )

    torch.testing.assert_close(positive_evidence.values, -negative_evidence.values)
    torch.testing.assert_close(
        positive_evidence.confidence,
        negative_evidence.confidence,
    )
    torch.testing.assert_close(
        positive_diagnostics.positive_expected_count,
        negative_diagnostics.negative_expected_count,
    )
    torch.testing.assert_close(
        positive_diagnostics.negative_expected_count,
        negative_diagnostics.positive_expected_count,
    )
    torch.testing.assert_close(
        positive_diagnostics.fused_probability,
        1.0 - negative_diagnostics.fused_probability,
    )
    torch.testing.assert_close(
        positive_diagnostics.forward_probability_before,
        1.0 - negative_diagnostics.forward_probability_before,
    )
    assert positive_diagnostics.nll_before == pytest.approx(
        negative_diagnostics.nll_before
    )
    assert positive_diagnostics.nll_after == pytest.approx(
        negative_diagnostics.nll_after
    )


def test_capability_invalid_rows_are_isolated_from_forward_and_unary() -> None:
    positive = torch.tensor([True, False])
    negative = torch.tensor([False, True])
    evidence, diagnostics = registered_forward_beta_observation(
        gaussian_ids=torch.tensor([0, 1, 1]),
        pixel_ids=torch.tensor([0, 0, 1]),
        contribution_weights=torch.tensor([0.4, 0.6, 1.0]),
        capability_valid=torch.tensor([True, False]),
        field_prior=torch.tensor([0.3, 0.9]),
        positive_pixel_mask=positive,
        negative_pixel_mask=negative,
        labeled_pixel_mask=positive | negative,
        all_pixel_mask=torch.ones(2, dtype=torch.bool),
    )

    torch.testing.assert_close(evidence.values, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(evidence.confidence, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        diagnostics.visible_contribution_mass,
        torch.tensor([0.4, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.labeled_expected_count,
        torch.tensor([0.4, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics.fused_probability,
        torch.tensor([1.0, 0.9], dtype=torch.float64),
    )
    assert diagnostics.valid_hit_count == 1
    assert diagnostics.observable_labeled_pixel_count == 1
    assert diagnostics.unobservable_labeled_pixel_count == 1


def test_empty_sparse_operator_returns_unobserved_evidence() -> None:
    evidence, diagnostics = registered_forward_beta_observation(
        gaussian_ids=torch.empty(0, dtype=torch.long),
        pixel_ids=torch.empty(0, dtype=torch.long),
        contribution_weights=torch.empty(0),
        capability_valid=torch.tensor([True, True]),
        field_prior=torch.tensor([0.25, 0.75]),
        positive_pixel_mask=torch.tensor([True, False]),
        negative_pixel_mask=torch.tensor([False, True]),
        labeled_pixel_mask=torch.tensor([True, True]),
        all_pixel_mask=torch.tensor([True, True]),
    )

    torch.testing.assert_close(evidence.values, torch.zeros(2))
    torch.testing.assert_close(evidence.confidence, torch.zeros(2))
    torch.testing.assert_close(
        diagnostics.fused_probability,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    assert diagnostics.valid_hit_count == 0
    assert diagnostics.observable_labeled_pixel_count == 0
    assert diagnostics.unobservable_labeled_pixel_count == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"gaussian_ids": torch.tensor([[0]])}, "one-dimensional"),
        ({"gaussian_ids": torch.tensor([0.0])}, "integer dtype"),
        ({"gaussian_ids": torch.tensor([1])}, "gaussian id outside"),
        ({"pixel_ids": torch.tensor([1])}, "pixel id outside"),
        ({"contribution_weights": torch.tensor([float("nan")])}, "finite values"),
        ({"contribution_weights": torch.tensor([-0.1])}, "finite values"),
        ({"contribution_weights": torch.tensor([1.1])}, "finite values"),
        ({"capability_valid": torch.tensor([1])}, "boolean dtype"),
        ({"field_prior": torch.tensor([float("inf")])}, "finite probabilities"),
        ({"field_prior": torch.tensor([1.1])}, "finite probabilities"),
        ({"positive_pixel_mask": torch.tensor([1])}, "boolean dtype"),
        ({"labeled_pixel_mask": torch.tensor([False])}, "must equal"),
        ({"all_pixel_mask": torch.tensor([False])}, "subset"),
    ],
)
def test_forward_beta_rejects_malformed_inputs(override, message) -> None:
    arguments = {
        "gaussian_ids": torch.tensor([0]),
        "pixel_ids": torch.tensor([0]),
        "contribution_weights": torch.tensor([0.5]),
        "capability_valid": torch.tensor([True]),
        "field_prior": torch.tensor([0.5]),
        "positive_pixel_mask": torch.tensor([True]),
        "negative_pixel_mask": torch.tensor([False]),
        "labeled_pixel_mask": torch.tensor([True]),
        "all_pixel_mask": torch.tensor([True]),
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        registered_forward_beta_observation(**arguments)


def test_forward_beta_rejects_overlapping_labels_and_invalid_alpha_mass() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        registered_forward_beta_observation(
            gaussian_ids=torch.tensor([0]),
            pixel_ids=torch.tensor([0]),
            contribution_weights=torch.tensor([0.5]),
            capability_valid=torch.tensor([True]),
            field_prior=torch.tensor([0.5]),
            positive_pixel_mask=torch.tensor([True]),
            negative_pixel_mask=torch.tensor([True]),
            labeled_pixel_mask=torch.tensor([True]),
            all_pixel_mask=torch.tensor([True]),
        )

    with pytest.raises(ValueError, match="must not exceed one"):
        registered_forward_beta_observation(
            gaussian_ids=torch.tensor([0, 1]),
            pixel_ids=torch.tensor([0, 0]),
            contribution_weights=torch.tensor([0.6, 0.6]),
            capability_valid=torch.tensor([True, True]),
            field_prior=torch.tensor([0.5, 0.5]),
            positive_pixel_mask=torch.tensor([True]),
            negative_pixel_mask=torch.tensor([False]),
            labeled_pixel_mask=torch.tensor([True]),
            all_pixel_mask=torch.tensor([True]),
        )


def test_forward_beta_rejects_invalid_mask_shape_and_epsilon() -> None:
    base = {
        "gaussian_ids": torch.tensor([0]),
        "pixel_ids": torch.tensor([0]),
        "contribution_weights": torch.tensor([0.5]),
        "capability_valid": torch.tensor([True]),
        "field_prior": torch.tensor([0.5]),
        "positive_pixel_mask": torch.tensor([True]),
        "negative_pixel_mask": torch.tensor([False]),
        "labeled_pixel_mask": torch.tensor([True]),
        "all_pixel_mask": torch.tensor([True]),
    }
    with pytest.raises(ValueError, match="align as non-empty"):
        registered_forward_beta_observation(
            **{**base, "all_pixel_mask": torch.tensor([True, True])}
        )
    with pytest.raises(ValueError, match="eps must"):
        registered_forward_beta_observation(**base, eps=0.0)
