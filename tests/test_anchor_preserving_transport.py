from __future__ import annotations

import pytest
import torch

from radio_gs.querying.anchor_preserving_transport import (
    apply_anchor_preserving_logit_residual,
    apply_anchor_preserving_probability_proposal,
    method_contract,
    residual_budget,
)


def test_budget_has_exact_anchor_and_proportional_partial_intervention() -> None:
    observation = torch.tensor([1.0, 1.0 - 5e-6, 0.75, 0.0])
    completion = torch.tensor([1.0, 1.0, 0.8, 0.5])
    actual = residual_budget(
        observation,
        completion_confidence=completion,
        fully_observed_tolerance=1e-5,
    )
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.0, 0.2, 0.5]))


def test_logit_transport_is_exact_identity_on_fully_observed_rows() -> None:
    anchor = torch.tensor([0.0, 1.0, 0.2, 0.8])
    output = apply_anchor_preserving_logit_residual(
        anchor,
        torch.tensor([100.0, -100.0, 2.0, -2.0]),
        torch.tensor([1.0, 1.0, 0.5, 0.0]),
        max_abs_logit_residual=3.0,
    )
    assert torch.equal(output.probability[:2], anchor[:2])
    assert torch.count_nonzero(output.applied_logit_residual[:2]) == 0
    assert output.applied_logit_residual[2].item() == pytest.approx(1.0)
    assert output.applied_logit_residual[3].item() == pytest.approx(-2.0)


def test_probability_proposal_respects_global_bound_and_completion_confidence() -> None:
    output = apply_anchor_preserving_probability_proposal(
        torch.full((3,), 0.5),
        torch.tensor([0.999, 0.999, 0.999]),
        torch.zeros(3),
        completion_confidence=torch.tensor([1.0, 0.5, 0.0]),
        max_abs_logit_residual=2.0,
    )
    torch.testing.assert_close(
        output.applied_logit_residual, torch.tensor([2.0, 1.0, 0.0])
    )
    assert output.probability[0] > output.probability[1] > output.probability[2]


def test_inactive_domain_is_bitwise_anchor_even_with_full_budget() -> None:
    anchor = torch.tensor([0.2, 0.8])
    output = apply_anchor_preserving_logit_residual(
        anchor,
        torch.tensor([3.0, -3.0]),
        torch.zeros(2),
        active_domain=torch.tensor([False, True]),
        max_abs_logit_residual=4.0,
    )
    assert torch.equal(output.probability[:1], anchor[:1])
    assert output.residual_gate[0].item() == 0.0
    assert output.probability[1].item() != pytest.approx(anchor[1].item())


def test_complement_and_residual_sign_are_symmetric() -> None:
    anchor = torch.tensor([0.17, 0.31, 0.72])
    residual = torch.tensor([-1.7, 0.4, 2.1])
    confidence = torch.tensor([0.0, 0.35, 0.8])
    positive = apply_anchor_preserving_logit_residual(
        anchor,
        residual,
        confidence,
        max_abs_logit_residual=3.0,
    )
    complement = apply_anchor_preserving_logit_residual(
        1.0 - anchor,
        -residual,
        confidence,
        max_abs_logit_residual=3.0,
    )
    torch.testing.assert_close(
        complement.probability,
        1.0 - positive.probability,
        rtol=1e-6,
        atol=1e-7,
    )


def test_contract_forbids_target_and_connected_selection() -> None:
    contract = method_contract()
    assert contract["fully_observed_policy"] == "exact_identity"
    assert contract["uses_target_rgb_mask_or_metric"] is False
    assert contract["connected_selection"] is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"anchor_probability": torch.tensor([1.1])}, "anchor_probability"),
        ({"observation_confidence": torch.tensor([-0.1])}, "observation_confidence"),
        ({"max_abs_logit_residual": 0.0}, "max_abs_logit_residual"),
    ],
)
def test_transport_fails_closed(kwargs: dict, match: str) -> None:
    inputs = {
        "anchor_probability": torch.tensor([0.5]),
        "proposed_logit_residual": torch.tensor([1.0]),
        "observation_confidence": torch.tensor([0.0]),
        "max_abs_logit_residual": 2.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        apply_anchor_preserving_logit_residual(**inputs)
