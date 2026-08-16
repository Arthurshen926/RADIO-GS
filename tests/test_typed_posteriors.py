from __future__ import annotations

import torch

from radio_gs.querying.typed_posteriors import (
    CategoricalPosteriorV2,
    MarginalCategoricalPosteriorV2,
    TextPosteriorV2,
    validate_reliability_state,
)


def _reliability() -> torch.Tensor:
    return torch.tensor(
        [
            [0.9, 0.1, 0.1, 0.9, 0.9],
            [0.8, 0.2, 0.2, 0.8, 0.7],
            [0.4, 0.6, 0.8, 0.3, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


def test_text_posterior_is_zero_initialized_primitive_identity() -> None:
    semantic = torch.tensor([[2.0], [0.4], [-1.0], [5.0]])
    valid = torch.tensor([True, True, True, False])
    head = TextPosteriorV2(extent_feature_dim=2)
    result = head(
        semantic,
        reliability=_reliability(),
        valid=valid,
        region_logit=torch.tensor([[0.2], [1.0], [2.0], [0.0]]),
        extent_features=torch.ones(4, 1, 2),
    )
    assert torch.equal(result.logits[:3], semantic[:3])
    assert torch.equal(result.probability[:3], torch.sigmoid(semantic[:3]))
    assert torch.equal(result.probability[3], torch.zeros(1))


def test_text_posterior_nonempty_guard_is_fixed_and_target_blind() -> None:
    head = TextPosteriorV2(extent_feature_dim=0)
    result = head(
        torch.tensor([[-4.0, -5.0], [-2.0, -3.0], [-6.0, -1.0]]),
        reliability=_reliability()[:3],
        valid=torch.ones(3, dtype=torch.bool),
    )
    selected = result.select(threshold=0.8, ensure_nonempty=True)
    assert selected.sum(dim=0).tolist() == [1, 1]
    assert selected[:, 0].tolist() == [False, True, False]
    assert selected[:, 1].tolist() == [False, False, True]


def test_text_posterior_post_spatial_initialization_is_exact_identity() -> None:
    head = TextPosteriorV2(extent_feature_dim=2)
    base = torch.tensor([[0.0, 0.2], [0.6, 1.0], [0.9, 0.4], [0.8, 0.7]])
    valid = torch.tensor([True, True, True, False])
    result = head.forward_post_spatial(
        base,
        reliability=_reliability(),
        valid=valid,
        region_probability=torch.ones_like(base),
        extent_features=torch.randn(4, 2, 2),
    )
    assert torch.equal(result.probability[:3], base[:3])
    assert torch.equal(result.probability[3], torch.zeros(2))


def test_text_posterior_post_spatial_can_recover_clipped_zero() -> None:
    head = TextPosteriorV2(extent_feature_dim=0)
    with torch.no_grad():
        head.extent_residual[-1].bias.fill_(2.0)
    result = head.forward_post_spatial(
        torch.tensor([[0.0], [1.0]]),
        reliability=_reliability()[:2],
        valid=torch.ones(2, dtype=torch.bool),
    )
    assert result.probability[0, 0] > 0.6
    assert result.probability[1, 0] == 1.0


def test_text_posterior_post_spatial_zero_init_keeps_residual_gradient() -> None:
    head = TextPosteriorV2(extent_feature_dim=0)
    result = head.forward_post_spatial(
        torch.tensor([[0.2], [0.6]]),
        reliability=_reliability()[:2],
        valid=torch.ones(2, dtype=torch.bool),
    )
    result.probability.sum().backward()
    assert head.extent_residual[-1].bias.grad is not None
    assert head.extent_residual[-1].bias.grad.abs() > 0


def test_text_posterior_post_spatial_residual_scale_is_global_shrinkage() -> None:
    head = TextPosteriorV2(extent_feature_dim=0)
    with torch.no_grad():
        head.extent_residual[-1].bias.fill_(0.5)
    base = torch.tensor([[0.2]])
    full = head.forward_post_spatial(
        base,
        reliability=_reliability()[:1],
        valid=torch.ones(1, dtype=torch.bool),
    ).probability
    half = head.forward_post_spatial(
        base,
        reliability=_reliability()[:1],
        valid=torch.ones(1, dtype=torch.bool),
        residual_scale=0.5,
    ).probability
    assert torch.allclose(half - base, 0.5 * (full - base))


def test_categorical_posterior_is_mutually_exclusive_and_can_abstain() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.2, 0.1], [2.0, 2.1], [4.0, 1.0]])
    valid = torch.tensor([True, True, True, False])
    head = CategoricalPosteriorV2(num_classes=2)
    baseline = head(logits, reliability=_reliability(), valid=valid)
    assert baseline.prediction.tolist() == [0, 0, 1, -1]
    with torch.no_grad():
        head.background_bias.fill_(2.0)
        head.background_reliability.weight.zero_()
        head.background_reliability.weight[0, 4] = -6.0
    calibrated = head(logits, reliability=_reliability(), valid=valid)
    assert calibrated.prediction[0].item() == 0
    assert calibrated.prediction[1].item() == 0
    assert calibrated.prediction[2].item() == -1
    assert calibrated.prediction[3].item() == -1
    assert torch.allclose(calibrated.probability.sum(dim=-1), torch.ones(4))


def test_reliability_contract_rejects_resultant_dispersion_drift() -> None:
    reliability = _reliability()[:3].clone()
    reliability[1, 1] = 0.7
    try:
        validate_reliability_state(reliability, torch.ones(3, dtype=torch.bool))
    except ValueError as error:
        assert "dispersion" in str(error)
    else:
        raise AssertionError("inconsistent reliability state was accepted")


def test_categorical_active_subset_returns_global_class_indices() -> None:
    head = CategoricalPosteriorV2(num_classes=4)
    result = head(
        torch.tensor([[9.0, 2.0, 1.0, 3.0]]),
        reliability=_reliability()[:1],
        valid=torch.ones(1, dtype=torch.bool),
        active_class_indices=torch.tensor([1, 3]),
    )
    assert result.prediction.tolist() == [3]


def test_marginal_categorical_zero_init_is_exact_argmax_identity() -> None:
    logits = torch.tensor(
        [[0.4, 0.3, 0.1], [0.1, 0.5, 0.2], [0.2, 0.1, 0.8], [9.0, 1.0, 0.0]]
    )
    valid = torch.tensor([True, True, True, False])
    head = MarginalCategoricalPosteriorV2(num_classes=3)
    result = head(logits, valid=valid)
    assert torch.equal(result.logits[:3], logits[:3])
    assert result.prediction.tolist() == [0, 1, 2, -1]


def test_marginal_categorical_rule_changes_unseen_class_without_class_parameters() -> None:
    logits = torch.tensor(
        [[0.8, 0.74, 0.2], [0.79, 0.70, 0.1], [0.78, 0.69, 0.3]]
    )
    head = MarginalCategoricalPosteriorV2(num_classes=3)
    with torch.no_grad():
        head.centering_parameter.fill_(4.0)
    result = head(logits, valid=torch.ones(3, dtype=torch.bool))
    assert result.prediction[0].item() == 1
    assert head.state_dict().keys() == {"centering_parameter", "scaling_parameter"}


def test_query_valid_fallback_may_have_unknown_zero_reliability() -> None:
    reliability, valid = validate_reliability_state(
        torch.zeros(2, 5), torch.ones(2, dtype=torch.bool)
    )
    assert valid.all()
    assert not reliability.any()
