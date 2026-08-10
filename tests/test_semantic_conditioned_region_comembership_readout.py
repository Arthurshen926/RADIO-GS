from __future__ import annotations

import pytest
import torch

from radio_gs.querying.semantic_conditioned_region_comembership_readout import (
    semantic_conditioned_bounded_region_union,
)


def _inputs() -> dict:
    return {
        "region_absolute_relevance": torch.tensor(
            [
                [0.90, 0.20],
                [0.70, 0.80],
                [0.49, 0.75],
                [0.95, 0.40],
            ]
        ),
        "pair_indices": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "pair_probabilities": torch.tensor([0.90, 0.95, 0.95]),
        "probability_threshold": 0.80,
        "method": "maximum_product",
        "maximum_regions": 4,
        "region_rows": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
        "token_mask": torch.ones(4, 2, dtype=torch.bool),
        "num_primitives": 8,
    }


def test_absolute_relevance_owns_seed_and_blocks_negative_bridge() -> None:
    result = semantic_conditioned_bounded_region_union(**_inputs())
    # Query zero starts at row three, but row two is below the exact 0.5
    # positive/negative boundary and therefore cannot bridge into rows one/zero.
    assert result.seed_region_indices.tolist() == [3, 1]
    assert torch.where(result.selected_region_masks[:, 0])[0].tolist() == [3]
    # Query one retains the connected positive rows one and two.
    assert torch.where(result.selected_region_masks[:, 1])[0].tolist() == [1, 2]


def test_below_boundary_argmax_is_retained_as_singleton() -> None:
    inputs = _inputs()
    inputs["region_absolute_relevance"] = torch.tensor([[0.10], [0.20], [0.30], [0.40]])
    result = semantic_conditioned_bounded_region_union(**inputs)
    assert result.seed_region_indices.tolist() == [3]
    assert torch.where(result.selected_region_masks[:, 0])[0].tolist() == [3]


def test_equal_relevance_uses_lowest_canonical_row_tie_break() -> None:
    inputs = _inputs()
    inputs["maximum_regions"] = 1
    inputs["region_absolute_relevance"] = torch.tensor([[0.80], [0.90], [0.90], [0.10]])
    result = semantic_conditioned_bounded_region_union(**inputs)
    assert result.seed_region_indices.tolist() == [1]


def test_semantic_filter_is_applied_before_multipoint_selection() -> None:
    inputs = _inputs()
    inputs["method"] = "multipoint_consistency"
    inputs["region_absolute_relevance"] = torch.tensor([[0.90], [0.70], [0.49], [0.95]])
    result = semantic_conditioned_bounded_region_union(**inputs)
    assert torch.where(result.selected_region_masks[:, 0])[0].tolist() == [3]


def test_rejects_non_probability_relevance() -> None:
    inputs = _inputs()
    inputs["region_absolute_relevance"][0, 0] = 1.01
    with pytest.raises(ValueError, match="semantic-conditioned"):
        semantic_conditioned_bounded_region_union(**inputs)


def test_singleton_still_validates_graph_authority() -> None:
    inputs = _inputs()
    inputs["maximum_regions"] = 1
    inputs["pair_probabilities"][0] = float("nan")
    with pytest.raises(ValueError, match="graph"):
        semantic_conditioned_bounded_region_union(**inputs)
