from __future__ import annotations

import pytest
import torch

from radio_gs.querying.nvos_local_positive_completion import (
    local_majority_positive_evidence,
    method_contract,
    source_only_loo_diagnostic,
)


def _scribbles(shape=(2, 3)):
    return torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)


def test_fixed_majority_margin_and_no_negative_absence() -> None:
    q = torch.tensor([[0.0, 0.5, 0.6], [0.75, 1.0, 0.4]])
    positive, negative = _scribbles()
    probability, reliability = local_majority_positive_evidence(
        q, positive_scribble=positive, negative_scribble=negative
    )
    assert torch.allclose(
        reliability, torch.tensor([[0.0, 0.0, 0.2], [0.5, 1.0, 0.0]])
    )
    assert torch.equal(
        probability, torch.tensor([[0.5, 0.5, 1.0], [1.0, 1.0, 0.5]])
    )
    assert method_contract()["parameter_sweep"] is False


def test_raw_signed_scribbles_override_completion() -> None:
    q = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    positive = torch.tensor([[False, True], [False, False]])
    negative = torch.tensor([[True, False], [False, False]])
    probability, reliability = local_majority_positive_evidence(
        q, positive_scribble=positive, negative_scribble=negative
    )
    assert probability[0, 0].item() == 0.0
    assert reliability[0, 0].item() == 1.0
    assert probability[0, 1].item() == 1.0
    assert reliability[0, 1].item() == 1.0


def test_scribble_conflict_fails_closed() -> None:
    q = torch.full((2, 2), 0.75)
    positive = torch.tensor([[True, False], [False, False]])
    with pytest.raises(ValueError, match="overlap"):
        local_majority_positive_evidence(
            q, positive_scribble=positive, negative_scribble=positive
        )


def test_source_only_loo_excludes_scribbles_and_is_finite() -> None:
    masks = torch.zeros((10, 2, 3), dtype=torch.bool)
    masks[:, 0, 0] = True
    masks[:8, 0, 1] = True
    masks[:6, 0, 2] = True
    masks[:2, 1, 0] = True
    positive = torch.zeros((2, 3), dtype=torch.bool)
    negative = torch.zeros((2, 3), dtype=torch.bool)
    positive[1, 1] = True
    negative[1, 2] = True
    diagnostic = source_only_loo_diagnostic(
        masks, positive_scribble=positive, negative_scribble=negative
    )
    assert diagnostic["full_fit"]["evaluable_non_scribble_pixels"] == 4
    assert diagnostic["full_fit"]["nonproposal_completion_confidence_mass"] == 0.0
    assert len(diagnostic["per_trial"]) == 10
    assert diagnostic["safety"]["absence_used_as_negative_evidence"] is False
    assert 0.0 <= diagnostic["summary"]["pooled_confidence_weighted_precision"] <= 1.0


@pytest.mark.parametrize(
    "masks",
    [
        torch.zeros((9, 2, 2), dtype=torch.bool),
        torch.zeros((10, 0, 2), dtype=torch.bool),
        torch.zeros((10, 2, 2), dtype=torch.float32),
    ],
)
def test_invalid_trial_authority_fails_closed(masks: torch.Tensor) -> None:
    positive, negative = _scribbles((2, 2))
    with pytest.raises(ValueError, match="trial_masks"):
        source_only_loo_diagnostic(
            masks, positive_scribble=positive, negative_scribble=negative
        )
