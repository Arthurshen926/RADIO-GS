from __future__ import annotations

import pytest
import torch

from radio_gs.interfaces.lerf_marginal_preserving_copula_residual import (
    marginal_preserving_copula_residual,
    marginal_preserving_primitive_query_scores,
)


def test_exact_marginal_threshold_counts_and_rank_budget() -> None:
    accepted = torch.tensor(
        [[0.05, 0.91, 0.21, 0.61, 0.42, 0.79, 0.32, 0.71, 0.11, 0.52]]
    )
    candidate = -accepted
    result = marginal_preserving_copula_residual(
        accepted,
        candidate,
        torch.ones(10, dtype=torch.bool),
        strength=1.0,
        maximum_rank_fraction=0.4,
    )
    assert result.marginal_exact
    assert result.block_size == 4
    assert result.maximum_rank_displacement <= 3
    assert torch.equal(torch.sort(result.scores).values, torch.sort(accepted).values)
    for threshold in (0.0, 0.1, 0.5, 0.6, 0.8, 1.0):
        assert int((result.scores >= threshold).sum()) == int(
            (accepted >= threshold).sum()
        )


def test_half_block_offset_allows_safe_top_decile_membership_change() -> None:
    accepted = torch.arange(100, dtype=torch.float32)[None]
    candidate = accepted.clone()
    # Reverse only the accepted ranks around the top-decile boundary.
    candidate[0, 85:95] = candidate[0, 85:95].flip(0)
    result = marginal_preserving_copula_residual(
        accepted,
        candidate,
        torch.ones(100, dtype=torch.bool),
        strength=1.0,
        maximum_rank_fraction=0.1,
    )
    assert result.maximum_rank_displacement <= 9
    assert torch.equal(torch.sort(result.scores, dim=-1).values, accepted)
    assert not torch.equal(result.scores >= 90, accepted >= 90)
    assert int((result.scores >= 90).sum()) == 10


def test_zero_strength_and_singleton_blocks_are_bitwise_fail_closed() -> None:
    generator = torch.Generator().manual_seed(7)
    accepted = torch.randn(3, 41, generator=generator)
    candidate = torch.randn(3, 41, generator=generator)
    valid = torch.rand(41, generator=generator) > 0.2
    zero = marginal_preserving_copula_residual(
        accepted,
        candidate,
        valid,
        strength=0.0,
        maximum_rank_fraction=1.0,
    )
    singleton = marginal_preserving_copula_residual(
        accepted,
        candidate,
        valid,
        strength=1.0,
        maximum_rank_fraction=1.0 / 1000.0,
    )
    assert torch.equal(zero.scores, accepted)
    assert torch.equal(singleton.scores, accepted)


def test_invalid_items_are_immutable_and_reliability_can_clamp() -> None:
    accepted = torch.tensor([[0.2, 0.4, 99.0, 0.6, 0.8]])
    candidate = torch.tensor([[0.8, 0.6, -99.0, 0.4, 0.2]])
    valid = torch.tensor([True, True, False, True, True])
    result = marginal_preserving_copula_residual(
        accepted,
        candidate,
        valid,
        reliability=torch.zeros(5),
        strength=1.0,
        maximum_rank_fraction=1.0,
    )
    assert torch.equal(result.scores, accepted)
    assert result.scores[0, 2].item() == 99.0


def test_single_zero_reliability_item_is_a_fixed_rank_barrier() -> None:
    accepted = torch.arange(9, dtype=torch.float32)[None]
    candidate = accepted.flip(-1)
    reliability = torch.ones(9)
    reliability[4] = 0.0
    result = marginal_preserving_copula_residual(
        accepted,
        candidate,
        torch.ones(9, dtype=torch.bool),
        reliability=reliability,
        strength=1.0,
        maximum_rank_fraction=1.0,
    )
    assert result.scores[0, 4].item() == accepted[0, 4].item()
    assert bool((result.scores[0, :4] < accepted[0, 4]).all())
    assert bool((result.scores[0, 5:] > accepted[0, 4]).all())


@pytest.mark.parametrize(
    "strength,rank_fraction",
    [(-0.1, 0.1), (1.1, 0.1), (0.5, -0.1), (0.5, 1.1)],
)
def test_invalid_policy_is_rejected(strength: float, rank_fraction: float) -> None:
    with pytest.raises(ValueError):
        marginal_preserving_copula_residual(
            torch.ones(1, 4),
            torch.ones(1, 4),
            torch.ones(4, dtype=torch.bool),
            strength=strength,
            maximum_rank_fraction=rank_fraction,
        )


def test_formal_primitive_query_adapter_is_explicit_nq() -> None:
    accepted = torch.tensor(
        [
            [0.1, 0.8],
            [0.2, 0.7],
            [0.3, 0.6],
            [0.4, 0.5],
            [0.5, 0.4],
            [0.6, 0.3],
        ]
    )
    candidate = accepted.flip(0)
    valid = torch.ones(6, dtype=torch.bool)
    result = marginal_preserving_primitive_query_scores(
        accepted,
        candidate,
        valid,
        strength=1.0,
        maximum_rank_fraction=1.0,
    )
    assert result.scores.shape == accepted.shape
    for query in range(accepted.shape[1]):
        assert torch.equal(
            torch.sort(result.scores[:, query]).values,
            torch.sort(accepted[:, query]).values,
        )
    with pytest.raises(ValueError, match=r"\[N,Q\]"):
        marginal_preserving_primitive_query_scores(
            accepted.T,
            candidate.T,
            valid,
            strength=1.0,
            maximum_rank_fraction=1.0,
        )
