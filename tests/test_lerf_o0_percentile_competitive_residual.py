from __future__ import annotations

import json

import pytest
import torch

from radio_gs.interfaces import lerf_o0_percentile_competitive_residual as competitive


def _run(
    o0: torch.Tensor,
    proposed: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> competitive.PercentileCompetitiveResidualResult:
    return competitive.percentile_competitive_residual(
        o0_logits=o0,
        proposed_residual_logits=proposed,
        primitive_valid_mask=(
            torch.ones(o0.shape[0], dtype=torch.bool) if valid is None else valid
        ),
        selection_probability_threshold=0.6,
        maximum_residual_logit=0.4,
    )


def test_contract_never_compares_raw_cross_query_scores_or_adds_parameters() -> None:
    contract = competitive.allocation_contract()
    serialized = json.dumps(contract, sort_keys=True).lower()
    assert contract["cross_query_raw_score_comparison"] is False
    assert contract["query_conditioned_parameters"] is False
    assert contract["scene_conditioned_parameters"] is False
    assert contract["target_metrics_used"] is False
    assert "midrank" in serialized
    assert "minmax" not in serialized


def test_only_threshold_reachable_proposals_can_change_o0() -> None:
    threshold_logit = torch.logit(torch.tensor(0.6)).item()
    o0 = torch.tensor(
        [[threshold_logit - 0.2], [threshold_logit - 0.5], [threshold_logit + 0.1]],
        dtype=torch.float32,
    )
    proposed = torch.full_like(o0, 0.3)
    result = _run(o0, proposed)
    assert result.activation_eligible_mask[:, 0].tolist() == [True, False, False]
    assert result.allocation_mask[:, 0].tolist() == [True, False, False]
    assert torch.equal(result.fused_logits[1:].view(torch.int32), o0[1:].view(torch.int32))
    assert result.fused_logits[0, 0] >= threshold_logit


def test_single_query_and_nonoverlapping_proposals_are_retained_exactly() -> None:
    o0 = torch.tensor(
        [[0.3, -2.0], [-2.0, 0.3], [0.0, 0.0]], dtype=torch.float32
    )
    proposed = torch.tensor(
        [[0.2, 0.0], [0.0, 0.2], [0.0, 0.0]], dtype=torch.float32
    )
    result = _run(o0, proposed)
    assert not bool(result.competition_mask.any())
    assert torch.equal(result.allocated_residual_logits, proposed)
    assert torch.equal(result.fused_logits, o0 + proposed)


def test_ambiguous_overlap_uses_within_query_rank_not_raw_magnitude() -> None:
    # At primitive zero q0 has the larger raw logit, but it is ordinary within q0;
    # q1 is its query-specific maximum and therefore wins the semantic allocation.
    o0 = torch.tensor(
        [[0.39, 0.35], [0.40, -2.0], [0.38, -3.0]], dtype=torch.float32
    )
    proposed = torch.tensor(
        [[0.03, 0.06], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )
    result = _run(o0, proposed)
    assert bool(result.competition_mask[0])
    assert result.within_query_midranks[0, 0] == pytest.approx(0.5)
    assert result.within_query_midranks[0, 1] == pytest.approx(1.0)
    assert result.allocation_mask[0].tolist() == [False, True]
    assert torch.equal(result.fused_logits[0, 0].view(torch.int32), o0[0, 0].view(torch.int32))


def test_exact_rank_ties_are_retained_without_query_index_tie_break() -> None:
    o0 = torch.tensor([[0.35, 0.35], [-1.0, -1.0]], dtype=torch.float32)
    proposed = torch.tensor([[0.1, 0.1], [0.0, 0.0]], dtype=torch.float32)
    result = _run(o0, proposed)
    assert result.allocation_mask[0].tolist() == [True, True]


def test_query_permutation_is_exactly_equivariant() -> None:
    o0 = torch.tensor(
        [[0.39, 0.35, 0.36], [0.40, -2.0, 0.1], [0.38, -3.0, 0.2]],
        dtype=torch.float32,
    )
    proposed = torch.tensor(
        [[0.03, 0.06, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    original = _run(o0, proposed)
    order = torch.tensor([2, 0, 1])
    permuted = _run(o0[:, order], proposed[:, order])
    assert torch.equal(permuted.within_query_midranks, original.within_query_midranks[:, order])
    assert torch.equal(permuted.allocation_mask, original.allocation_mask[:, order])
    assert torch.equal(permuted.fused_logits, original.fused_logits[:, order])


def test_allocation_mask_is_invariant_to_strict_monotone_query_reparameterization() -> None:
    # The transform is applied within each query.  Threshold reachability is held
    # fixed here so this isolates the advertised semantic rank allocation.
    o0 = torch.tensor(
        [[0.39, 0.35], [0.40, -2.0], [0.38, -3.0]], dtype=torch.float32
    )
    proposed = torch.tensor(
        [[0.03, 0.06], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )
    original = _run(o0, proposed)
    transformed = o0.clone()
    transformed[:, 0] = 0.4054651 - (0.4054651 - transformed[:, 0]) * 0.5
    transformed[:, 1] = 0.4054651 - (0.4054651 - transformed[:, 1]) * 0.01
    transformed_proposed = torch.zeros_like(proposed)
    transformed_proposed[0] = 0.4054651 - transformed[0] + 1e-4
    remapped = _run(transformed, transformed_proposed)
    assert torch.equal(remapped.within_query_midranks, original.within_query_midranks)
    assert torch.equal(remapped.allocation_mask, original.allocation_mask)


def test_invalid_rows_and_suppressed_competitors_are_bitwise_o0() -> None:
    o0 = torch.tensor([[0.35, 0.36], [0.35, 0.36]], dtype=torch.float32)
    proposed = torch.full_like(o0, 0.1)
    result = _run(o0, proposed, valid=torch.tensor([True, False]))
    assert torch.equal(result.fused_logits[1].view(torch.int32), o0[1].view(torch.int32))
    assert torch.count_nonzero(result.allocated_residual_logits[1]) == 0


@pytest.mark.parametrize(
    ("proposed", "maximum"),
    [
        (torch.tensor([[-0.1]], dtype=torch.float32), 0.4),
        (torch.tensor([[0.5]], dtype=torch.float32), 0.4),
    ],
)
def test_invalid_residual_proposals_fail_closed(
    proposed: torch.Tensor, maximum: float
) -> None:
    with pytest.raises(ValueError, match="inputs differ"):
        competitive.percentile_competitive_residual(
            o0_logits=torch.zeros_like(proposed),
            proposed_residual_logits=proposed,
            primitive_valid_mask=torch.tensor([True]),
            selection_probability_threshold=0.6,
            maximum_residual_logit=maximum,
        )
