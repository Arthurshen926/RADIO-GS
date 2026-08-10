from __future__ import annotations

import pytest
import torch

from radio_gs.querying.multi_region_union_readout import (
    MultiRegionUnionConfig,
    greedy_connected_expected_mass_union_readout,
    greedy_novelty_union_readout,
)


def test_greedy_union_prefers_new_coverage_over_redundant_region() -> None:
    probability = torch.tensor([[0.90], [0.89], [0.80]])
    rows = torch.tensor([[0, 1], [0, 1], [2, 3]], dtype=torch.int32)
    core = torch.ones_like(rows, dtype=torch.bool)
    result = greedy_novelty_union_readout(
        probability,
        rows,
        core,
        num_primitives=5,
        config=MultiRegionUnionConfig(maximum_regions=2, candidate_chunk_rows=2),
    )
    assert result.selected_region_indices == ((0, 2),)
    assert result.selected_marginal_core_rows == ((2, 2),)
    assert torch.equal(
        result.primitive_membership[:, 0],
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
    )


def test_evidence_gate_and_query_axes_are_independent() -> None:
    probability = torch.tensor([[0.70, 0.59], [0.61, 0.95]])
    rows = torch.tensor([[0, -1], [1, -1]], dtype=torch.int64)
    core = torch.tensor([[True, False], [True, False]])
    result = greedy_novelty_union_readout(probability, rows, core, num_primitives=3)
    assert result.selected_region_indices == ((0, 1), (1,))
    assert torch.equal(
        result.primitive_membership,
        torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
    )


def test_equal_utility_uses_smaller_immutable_candidate_index() -> None:
    probability = torch.tensor([[0.8], [0.8]])
    rows = torch.tensor([[0], [1]], dtype=torch.int64)
    core = torch.ones_like(rows, dtype=torch.bool)
    result = greedy_novelty_union_readout(
        probability,
        rows,
        core,
        num_primitives=2,
        config=MultiRegionUnionConfig(maximum_regions=1, candidate_chunk_rows=1),
    )
    assert result.selected_region_indices == ((0,),)


@pytest.mark.parametrize(
    ("probability", "rows", "core", "message"),
    [
        (torch.tensor([[1.1]]), torch.tensor([[0]]), torch.tensor([[True]]), "[0,1]"),
        (
            torch.tensor([[0.7]]),
            torch.tensor([[2]]),
            torch.tensor([[True]]),
            "out-of-range",
        ),
        (
            torch.tensor([[0.7]]),
            torch.tensor([[-1]]),
            torch.tensor([[True]]),
            "semantic core",
        ),
    ],
)
def test_invalid_authority_fails_closed(probability, rows, core, message) -> None:
    with pytest.raises(ValueError, match=message):
        greedy_novelty_union_readout(probability, rows, core, num_primitives=2)


def test_connected_v2_seed_uses_expected_supported_mass() -> None:
    probability = torch.tensor([[0.99], [0.70], [0.69]])
    rows = torch.tensor([[0, -1], [1, 2], [3, 4]], dtype=torch.int64)
    core = rows >= 0
    result = greedy_connected_expected_mass_union_readout(
        probability,
        rows,
        core,
        torch.tensor([[2], [3]], dtype=torch.int64),
        num_primitives=5,
        config=MultiRegionUnionConfig(maximum_regions=1),
    )
    assert result.selected_region_indices == ((1,),)
    assert result.selected_marginal_core_rows == ((2,),)


def test_connected_v2_rejects_disconnected_higher_probability() -> None:
    probability = torch.tensor([[0.90], [0.99], [0.70]])
    rows = torch.tensor([[0, 1, 6], [4, 5, -1], [2, 3, -1]], dtype=torch.int64)
    core = rows >= 0
    result = greedy_connected_expected_mass_union_readout(
        probability,
        rows,
        core,
        torch.tensor([[1], [2]], dtype=torch.int64),
        num_primitives=7,
        config=MultiRegionUnionConfig(maximum_regions=2, candidate_chunk_rows=1),
    )
    assert result.selected_region_indices == ((0, 2),)
    assert torch.equal(
        result.primitive_membership[:, 0],
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    )


def test_connected_v2_core_overlap_is_connected_without_edge() -> None:
    probability = torch.tensor([[0.90], [0.80], [0.70]])
    rows = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.int64)
    core = torch.ones_like(rows, dtype=torch.bool)
    result = greedy_connected_expected_mass_union_readout(
        probability,
        rows,
        core,
        torch.empty((2, 0), dtype=torch.int64),
        num_primitives=5,
        config=MultiRegionUnionConfig(maximum_regions=3),
    )
    assert result.selected_region_indices == ((0, 1),)
    assert result.selected_marginal_core_rows == ((2, 1),)


def test_connected_v2_rejects_invalid_support_edge() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        greedy_connected_expected_mass_union_readout(
            torch.tensor([[0.9]]),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[True]]),
            torch.tensor([[0], [1]], dtype=torch.int64),
            num_primitives=1,
        )
