from __future__ import annotations

import torch

from radio_gs.interfaces.source_monotone_missing_core_selector import (
    SELECTOR_FEATURE_NAMES,
    fit_monotone_additive_logistic,
    oriented_selector_features,
    select_largest_safe_oof_threshold,
    selector_probability,
    tie_invariant_average_precision,
)


def _unit_features(rows: int = 12) -> torch.Tensor:
    result = torch.zeros(rows, 37)
    axis = torch.linspace(0.1, 0.9, rows)
    result[:, 0] = axis
    result[:, 14] = axis
    result[:, 15] = axis
    result[:, 17] = torch.linspace(0.9, 0.1, rows)
    result[:, 9] = torch.arange(rows) % 3
    result[:, 18] = torch.linspace(3.0, 0.1, rows)
    return result


def test_oriented_selector_features_has_preregistered_directions() -> None:
    unit = _unit_features(3)
    oriented = oriented_selector_features(unit)
    assert oriented.shape == (3, len(SELECTOR_FEATURE_NAMES))
    assert torch.all(oriented[1:, :3] > oriented[:-1, :3])
    assert torch.all(oriented[1:, 3] > oriented[:-1, 3])
    assert oriented[1, 4] < oriented[0, 4]
    assert torch.all(oriented[1:, 5] > oriented[:-1, 5])


def test_fitted_selector_is_monotone_in_every_oriented_channel() -> None:
    torch.manual_seed(0)
    feature = torch.randn(60, len(SELECTOR_FEATURE_NAMES), dtype=torch.float64)
    labels = feature.sum(dim=1) > 0
    groups = torch.arange(60) // 3
    model = fit_monotone_additive_logistic(
        feature, labels, groups, maximum_iterations=25
    )
    baseline = torch.zeros(1, len(SELECTOR_FEATURE_NAMES))
    p0 = selector_probability(model, baseline)
    for index in range(len(SELECTOR_FEATURE_NAMES)):
        improved = baseline.clone()
        improved[0, index] = 1.0
        assert float(selector_probability(model, improved)) >= float(p0)
    assert bool((model.positive_weights >= 0.0).all())


def test_threshold_selects_largest_safe_tied_population() -> None:
    probability = torch.cat((torch.full((280,), 0.9), torch.full((120,), 0.4)))
    labels = torch.cat(
        (
            torch.cat((torch.ones(260), torch.zeros(20))),
            torch.cat((torch.ones(20), torch.zeros(100))),
        )
    ).bool()
    utility = torch.where(labels, torch.ones(400), -torch.ones(400))
    result = select_largest_safe_oof_threshold(
        probability, labels, utility, minimum_selected=256
    )
    assert result["selected"] == 280
    assert result["hard_positive"] == 260
    assert result["threshold_inclusive"] == probability[0]


def test_average_precision_is_invariant_within_ties() -> None:
    score = torch.tensor([0.9, 0.9, 0.3, 0.3])
    first = torch.tensor([True, False, True, False])
    second = torch.tensor([False, True, False, True])
    assert tie_invariant_average_precision(score, first) == tie_invariant_average_precision(
        score, second
    )
