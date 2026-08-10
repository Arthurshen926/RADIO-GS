from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.finalize_source_only_graph_positive_utility_fix4b import (
    _combine_signed,
    marginal_weighted_signed_utility,
)


def test_marginal_weighted_signed_utility_uses_gain_and_label_sign() -> None:
    result = marginal_weighted_signed_utility(
        gains=torch.tensor([[0.8, 0.2, 0.1]]),
        marginal_primitives=torch.tensor([[256, 128, 64]]),
        labels=torch.tensor([[True, False, True]]),
        selected=torch.tensor([[True, True, True]]),
        novel_mass_reference=256.0,
    )
    assert result["signed_gain_sum"] == pytest.approx(0.7)
    assert result["marginal_weight_sum"] == pytest.approx(1.75)
    assert result["marginal_weighted_signed_utility"] == pytest.approx(0.4)


def test_combine_signed_weights_scenes_by_selected_marginal_mass() -> None:
    combined = _combine_signed(
        [
            {
                "selected_count": 1,
                "positive_count": 1,
                "negative_count": 0,
                "signed_gain_sum": 0.8,
                "marginal_weight_sum": 1.0,
                "marginal_weighted_signed_utility": 0.8,
            },
            {
                "selected_count": 1,
                "positive_count": 0,
                "negative_count": 1,
                "signed_gain_sum": -0.2,
                "marginal_weight_sum": 0.5,
                "marginal_weighted_signed_utility": -0.4,
            },
        ]
    )
    assert combined["signed_gain_sum"] == pytest.approx(0.6)
    assert combined["marginal_weight_sum"] == pytest.approx(1.5)
    assert combined["marginal_weighted_signed_utility"] == pytest.approx(0.4)


def test_signed_utility_rejects_zero_marginal_selected_row() -> None:
    with pytest.raises(ValueError, match="inputs"):
        marginal_weighted_signed_utility(
            gains=torch.tensor([[0.1]]),
            marginal_primitives=torch.tensor([[0]]),
            labels=torch.tensor([[True]]),
            selected=torch.tensor([[True]]),
            novel_mass_reference=256.0,
        )

