from __future__ import annotations

import pytest
import torch

from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES,
    RegionCoMembershipV1,
    seed_connected_instance_filter,
)


def test_zero_initialized_head_is_epoch_zero_half_probability_and_trains() -> None:
    model = RegionCoMembershipV1(torch.zeros(15), torch.ones(15))
    features = torch.randn(8, len(PAIR_FEATURE_NAMES))
    logits = model(features)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(model.probability(features), torch.full((8,), 0.5))
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0] * 4)
    ).backward()
    assert model.logit.weight.grad is not None
    assert float(model.logit.weight.grad.abs().sum()) > 0


def test_seed_connected_filter_keeps_only_thresholded_seed_components() -> None:
    pairs = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    probability = torch.tensor([0.9, 0.8, 0.2, 0.95])
    selected = seed_connected_instance_filter(
        region_count=5,
        pair_indices=pairs,
        pair_probabilities=probability,
        seed_region_indices=[0],
        threshold=0.5,
    )
    assert selected.tolist() == [True, True, True, False, False]
    selected_two = seed_connected_instance_filter(
        region_count=5,
        pair_indices=pairs,
        pair_probabilities=probability,
        seed_region_indices=[0, 4],
        threshold=0.5,
    )
    assert selected_two.tolist() == [True, True, True, True, True]


def test_seed_connected_filter_rejects_noncanonical_pairs() -> None:
    with pytest.raises(ValueError, match="canonical pair order"):
        seed_connected_instance_filter(
            region_count=2,
            pair_indices=torch.tensor([[1], [0]]),
            pair_probabilities=torch.tensor([0.9]),
            seed_region_indices=[0],
            threshold=0.5,
        )
