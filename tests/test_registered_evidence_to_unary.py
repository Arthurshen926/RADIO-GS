from __future__ import annotations

import pytest
import torch

from radio_gs.querying.registered_evidence_to_unary import (
    FEATURE_NAMES,
    RegisteredEvidenceToUnaryV1,
    RegisteredEvidenceToUnaryV2,
    build_registered_evidence_features,
)


def _features(**overrides):
    rows = 4
    values = {
        "foreground_mass": torch.tensor([2.0, 0.0, 0.0, 1.0]),
        "background_mass": torch.tensor([0.0, 3.0, 0.0, 1.0]),
        "visible_mass": torch.tensor([2.0, 3.0, 4.0, 4.0]),
        "dino_margin": torch.tensor([0.2, -0.2, 0.1, 0.3]),
        "sam_margin": torch.tensor([0.1, -0.1, 0.2, -0.1]),
        "directional_dispersion": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "log_amplitude_std": torch.tensor([0.2, 0.3, 0.4, 0.5]),
        "observation_evidence": torch.tensor([3.0, 2.0, 1.0, 0.0]),
        "visibility_purity_value": torch.tensor([0.9, 0.8, 0.7, 0.6]),
        "visibility_purity_known": torch.tensor([True, True, False, False]),
        "capability_valid": torch.tensor([True, True, True, False]),
        "source_view_support": torch.tensor([1.0, 1.0, 0.0, 1.0]),
    }
    values.update(overrides)
    return build_registered_evidence_features(**values)


def test_feature_builder_preserves_full_mask_and_unlabeled_semantic_prior() -> None:
    features = _features()
    assert features.values.shape == (4, len(FEATURE_NAMES))
    assert features.analytic_probability[0].item() == pytest.approx(1.0 - 1e-6)
    assert features.analytic_probability[1].item() == pytest.approx(1e-6)
    assert features.labeled_coverage[2].item() == 0.0
    assert features.analytic_probability[2].item() == pytest.approx(
        torch.sigmoid(torch.tensor(1.2)).item()
    )
    # Invalid capability plus half-labeled, balanced evidence remains neutral.
    assert features.analytic_probability[3].item() == pytest.approx(0.5)


def test_zero_initialized_head_is_exact_graph_off_analytic_unary() -> None:
    features = _features()
    head = RegisteredEvidenceToUnaryV1(hidden_dim=8)
    output = head(features)
    assert torch.equal(output.foreground_probability, features.analytic_probability)
    assert torch.count_nonzero(output.bounded_logit_residual) == 0
    assert torch.equal(output.abstention, 1.0 - output.confidence)


def test_head_update_is_globally_bounded_and_has_no_row_coupling() -> None:
    features = _features()
    torch.manual_seed(7)
    head = RegisteredEvidenceToUnaryV1(hidden_dim=8, max_delta_logit=2.0)
    with torch.no_grad():
        head.output.weight.normal_()
        head.output.bias.normal_()
    output = head(features)
    assert bool((output.bounded_logit_residual.abs() <= 2.0).all())
    single = type(features)(
        values=features.values[:1],
        analytic_probability=features.analytic_probability[:1],
        registered_probability=features.registered_probability[:1],
        labeled_coverage=features.labeled_coverage[:1],
        capability_valid=features.capability_valid[:1],
    )
    assert torch.equal(
        head(single).foreground_probability,
        output.foreground_probability[:1],
    )


def test_v2_nonzero_head_preserves_complete_evidence_and_gates_partial_rows() -> None:
    features = _features()
    head = RegisteredEvidenceToUnaryV2(hidden_dim=8, max_delta_logit=2.0)
    with torch.no_grad():
        head.output.bias.copy_(torch.tensor([1.0, 10.0]))
    output = head(features)

    # Rows 0 and 1 have complete positive/background source evidence.
    assert torch.equal(
        output.foreground_probability[:2],
        features.analytic_probability[:2],
    )
    assert torch.count_nonzero(output.bounded_logit_residual[:2]) == 0

    # Row 2 is unobserved and row 3 is half observed.  With a constant raw
    # residual, the latter receives exactly half the residual budget.
    assert output.bounded_logit_residual[2].item() > 0
    assert output.bounded_logit_residual[3].item() == pytest.approx(
        0.5 * output.bounded_logit_residual[2].item()
    )
    assert output.foreground_probability[2].item() != pytest.approx(
        features.analytic_probability[2].item()
    )


def test_v2_zero_initialized_head_is_exact_analytic_unary() -> None:
    features = _features()
    output = RegisteredEvidenceToUnaryV2(hidden_dim=8)(features)
    assert torch.equal(output.foreground_probability, features.analytic_probability)
    assert torch.count_nonzero(output.bounded_logit_residual) == 0


def test_feature_builder_fails_closed_on_overlapping_prompt_mass() -> None:
    with pytest.raises(ValueError, match="exceeds source visible"):
        _features(
            foreground_mass=torch.tensor([2.0, 0.0, 0.0, 1.0]),
            background_mass=torch.tensor([1.0, 3.0, 0.0, 1.0]),
        )
