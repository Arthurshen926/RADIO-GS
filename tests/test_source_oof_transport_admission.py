from __future__ import annotations

import pytest
import torch

from radio_gs.querying.source_oof_transport_admission import (
    DirectionalAdmissionCalibration,
    apply_source_oof_directional_admission,
    directional_completion_confidence,
    fit_conservative_directional_admission,
    method_contract,
)


def test_counterfactual_fit_rejects_harmful_expansion_and_accepts_contraction() -> None:
    anchor = torch.full((12,), 0.5)
    proposal = torch.tensor([0.9, 0.8, 0.2, 0.1] * 3)
    # Expansion points are background and contraction points are background:
    # only contraction can improve on the neutral anchor.
    positive = torch.zeros(12)
    negative = torch.ones(12)
    calibration = fit_conservative_directional_admission(
        anchor,
        proposal,
        positive,
        negative,
        torch.ones(12, dtype=torch.bool),
        torch.arange(12) // 4,
    )
    assert calibration.expansion == 0.0
    assert calibration.contraction == 1.0
    assert calibration.leave_one_fold_expansion == (0.0, 0.0, 0.0)
    assert calibration.leave_one_fold_contraction == (1.0, 1.0, 1.0)


def test_lower_envelope_uses_most_conservative_leave_one_fold_fit() -> None:
    anchor = torch.full((9,), 0.5)
    proposal = torch.full((9,), 0.8)
    # One fold contradicts expansion.  Every leave-one-fold coefficient is
    # recorded and deployment uses their exact minimum.
    positive = torch.tensor([1.0] * 6 + [0.0] * 3)
    negative = 1.0 - positive
    result = fit_conservative_directional_admission(
        anchor,
        proposal,
        positive,
        negative,
        torch.ones(9, dtype=torch.bool),
        torch.arange(9) // 3,
    )
    assert result.expansion == min(result.leave_one_fold_expansion)
    assert result.leave_one_fold_expansion[2] == 1.0
    assert result.leave_one_fold_expansion[0] < 1.0


def test_directional_confidence_and_transport_preserve_anchors_exactly() -> None:
    calibration = DirectionalAdmissionCalibration(
        expansion=0.0,
        contraction=1.0,
        leave_one_fold_expansion=(0.0, 0.0, 0.0),
        leave_one_fold_contraction=(1.0, 1.0, 1.0),
        folds=(0, 1, 2),
        eligible_rows=3,
    )
    anchor = torch.tensor([0.5, 0.5, 0.8, 0.3])
    proposal = torch.tensor([0.9, 0.1, 0.1, 0.9])
    confidence = directional_completion_confidence(anchor, proposal, calibration)
    torch.testing.assert_close(confidence, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    output = apply_source_oof_directional_admission(
        anchor,
        proposal,
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        calibration,
    )
    assert output.probability[0].item() == anchor[0].item()
    assert output.probability[1].item() == pytest.approx(proposal[1].item())
    assert output.probability[2].item() == anchor[2].item()
    assert output.probability[3].item() == anchor[3].item()


def test_contract_forbids_target_feedback_and_connected_selection() -> None:
    contract = method_contract()
    assert contract["fold_aggregation"] == "minimum_leave_one_fold_coefficient"
    assert contract["uses_target_rgb_mask_or_metric"] is False
    assert contract["connected_selection"] is False


def test_fit_fails_closed_on_insufficient_folds_or_bad_weights() -> None:
    inputs = dict(
        anchor_probability=torch.full((6,), 0.5),
        proposal_probability=torch.linspace(0.1, 0.9, 6),
        positive_weight=torch.ones(6),
        negative_weight=torch.ones(6),
        eligible=torch.ones(6, dtype=torch.bool),
        fold_ids=torch.arange(6) // 3,
    )
    with pytest.raises(ValueError, match="at least three folds"):
        fit_conservative_directional_admission(**inputs)
    inputs["fold_ids"] = torch.arange(6) // 2
    inputs["negative_weight"] = torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fit_conservative_directional_admission(**inputs)
