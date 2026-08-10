from __future__ import annotations

import pytest
import torch

from radio_gs.querying.bounded_region_comembership_query_readout import (
    bounded_region_union_from_o0,
)


def _inputs() -> dict:
    return {
        "region_o0_scores": torch.tensor(
            [[0.9, 0.1], [0.2, 0.8], [0.3, 0.4], [0.1, 0.2]]
        ),
        "pair_indices": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "pair_probabilities": torch.tensor([0.8, 0.9, 0.7]),
        "probability_threshold": 0.75,
        "region_rows": torch.tensor([[0, 1], [2, 3], [3, 4], [5, 6]]),
        "token_mask": torch.ones(4, 2, dtype=torch.bool),
        "num_primitives": 7,
    }


def test_bounded_query_readout_applies_one_global_k_to_each_query() -> None:
    result = bounded_region_union_from_o0(
        **_inputs(), method="maximum_product", maximum_regions=2
    )
    assert result.seed_region_indices.tolist() == [0, 1]
    assert result.selected_region_masks.sum(dim=0).tolist() == [2, 2]
    assert torch.nonzero(result.primitive_membership[:, 0]).flatten().tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert torch.nonzero(result.primitive_membership[:, 1]).flatten().tolist() == [
        2,
        3,
        4,
    ]


def test_singleton_rule_never_opens_graph_edges() -> None:
    inputs = _inputs()
    inputs["pair_probabilities"] = torch.ones(3)
    result = bounded_region_union_from_o0(
        **inputs, method="maximum_product", maximum_regions=1
    )
    assert result.selected_region_masks.sum(dim=0).tolist() == [1, 1]


def test_singleton_rule_still_rejects_corrupt_inference() -> None:
    inputs = _inputs()
    inputs["pair_probabilities"] = torch.tensor([float("nan")] * 3)
    with pytest.raises(ValueError, match="graph"):
        bounded_region_union_from_o0(
            **inputs, method="maximum_product", maximum_regions=1
        )


def test_bounded_query_readout_rejects_unregistered_target_rule() -> None:
    with pytest.raises(ValueError, match="inputs"):
        bounded_region_union_from_o0(
            **_inputs(), method="widest_path", maximum_regions=8
        )
