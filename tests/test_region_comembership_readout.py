from __future__ import annotations

import torch

from radio_gs.querying.region_comembership_readout import (
    connected_region_union_from_o0,
)


def test_highest_o0_seed_tie_break_and_connected_union_are_fixed() -> None:
    result = connected_region_union_from_o0(
        region_o0_scores=torch.tensor([[0.9], [0.9], [0.2], [0.1]]),
        pair_indices=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        pair_probabilities=torch.tensor([0.8, 0.7, 0.95]),
        probability_threshold=0.75,
        region_rows=torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]]),
        token_mask=torch.ones(4, 2, dtype=torch.bool),
        num_primitives=6,
    )
    assert result.seed_region_indices.tolist() == [0]
    assert result.selected_region_masks[:, 0].tolist() == [True, True, False, False]
    assert result.primitive_membership[:, 0].tolist() == [1, 1, 1, 0, 0, 0]


def test_false_positive_bridge_exposes_transitive_closure() -> None:
    result = connected_region_union_from_o0(
        region_o0_scores=torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        pair_indices=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        pair_probabilities=torch.tensor([0.9, 0.9, 0.9]),
        probability_threshold=0.8,
        region_rows=torch.arange(4)[:, None],
        token_mask=torch.ones(4, 1, dtype=torch.bool),
        num_primitives=4,
    )
    assert bool(result.selected_region_masks.all())
