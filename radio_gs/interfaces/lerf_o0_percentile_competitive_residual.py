"""Parameter-free semantic allocation for overlapping LERF graph residuals.

The region graph is deliberately query agnostic.  Consequently, an overlap
between residual proposals for two text queries is evidence for geometric
support, but is not evidence that both semantic labels should be expanded.
This interface resolves only those overlaps that can change the frozen O0
decision.  It compares within-query empirical midranks, never raw scores from
different independently normalized queries.

The allocator is a conservative layer after a bounded, non-negative residual
proposal.  A proposal is activation eligible only when it can move an O0 logit
from below to at least the fixed selection threshold.  At a primitive with
multiple eligible proposals, all queries attaining the largest within-query
midrank survive (ties are retained).  Single-query proposals are unchanged.
Everything else is bitwise O0.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_o0_percentile_competitive_residual.v1"


def allocation_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "canonical_capability": "frozen_o0_primitive_logit",
        "input": "bounded_nonnegative_graph_residual_proposal",
        "activation_eligibility": (
            "o0_below_fixed_threshold_and_o0_plus_proposed_residual_at_or_above_"
            "fixed_threshold"
        ),
        "semantic_evidence": (
            "within_query_empirical_midrank_over_valid_primitives"
        ),
        "cross_query_raw_score_comparison": False,
        "competition_scope": (
            "only_same_primitive_with_multiple_activation_eligible_query_proposals"
        ),
        "winner": "maximum_within_query_midrank_with_all_exact_ties_retained",
        "single_proposal": "retained_exactly",
        "fusion": "o0_logit_plus_allocated_nonnegative_residual",
        "fallback": (
            "non_activation_eligible_or_competition_loser_or_invalid_primitive_is_"
            "bitwise_o0"
        ),
        "query_order_equivariant": True,
        "strictly_monotone_within_query_reparameterization_invariant": True,
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(allocation_contract())


@dataclass(frozen=True)
class PercentileCompetitiveResidualResult:
    fused_logits: torch.Tensor
    allocated_residual_logits: torch.Tensor
    within_query_midranks: torch.Tensor
    activation_eligible_mask: torch.Tensor
    competition_mask: torch.Tensor
    allocation_mask: torch.Tensor


def _empirical_midranks(
    logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Return tie-aware [0,1] empirical midranks independently per query."""

    primitive_count, query_count = logits.shape
    ranks = torch.zeros_like(logits)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return ranks
    for query in range(query_count):
        values = logits[valid, query]
        unique, inverse, counts = torch.unique(
            values, sorted=True, return_inverse=True, return_counts=True
        )
        if int(unique.numel()) <= 0:
            raise RuntimeError("empirical midrank unique support is empty")
        less = torch.cumsum(counts, dim=0) - counts
        mid = less.double() + (counts.double() - 1.0) / 2.0
        denominator = max(valid_count - 1, 1)
        normalized = (mid / float(denominator)).float()
        if valid_count == 1:
            normalized.fill_(1.0)
        ranks[valid, query] = normalized[inverse]
    return ranks.contiguous()


def percentile_competitive_residual(
    *,
    o0_logits: torch.Tensor,
    proposed_residual_logits: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    selection_probability_threshold: float,
    maximum_residual_logit: float,
) -> PercentileCompetitiveResidualResult:
    """Allocate threshold-effective graph proposals without raw score mixing."""

    o0 = torch.as_tensor(o0_logits).detach()
    proposed = torch.as_tensor(proposed_residual_logits).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    threshold = float(selection_probability_threshold)
    maximum = float(maximum_residual_logit)
    if (
        o0.dtype != torch.float32
        or o0.device.type != "cpu"
        or o0.ndim != 2
        or min(o0.shape) <= 0
        or proposed.dtype != torch.float32
        or proposed.device.type != "cpu"
        or proposed.shape != o0.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (o0.shape[0],)
        or not bool(torch.isfinite(o0).all())
        or not bool(torch.isfinite(proposed).all())
        or bool((proposed < 0.0).any())
        or not math.isfinite(threshold)
        or not 0.0 < threshold < 1.0
        or not math.isfinite(maximum)
        or maximum < 0.0
        or float(proposed.max()) > maximum + 1e-7
    ):
        raise ValueError("percentile-competitive residual inputs differ")

    threshold_logit = math.log(threshold / (1.0 - threshold))
    activation_eligible = (
        valid[:, None]
        & (proposed > 0.0)
        & (o0 < threshold_logit)
        & (o0 + proposed >= threshold_logit)
    )
    midranks = _empirical_midranks(o0, valid)
    proposal_count = activation_eligible.sum(dim=1)
    competition = proposal_count > 1
    allocation = activation_eligible.clone()
    if bool(competition.any()):
        eligible_rank = midranks.masked_fill(~activation_eligible, -1.0)
        best_rank = eligible_rank.max(dim=1).values
        winners = activation_eligible & (midranks == best_rank[:, None])
        allocation[competition] = winners[competition]

    allocated = torch.where(allocation, proposed, torch.zeros_like(proposed))
    fused = o0.clone()
    fused[allocation] = o0[allocation] + allocated[allocation]
    unchanged = ~allocation
    if (
        bool((allocated < 0.0).any())
        or bool((allocated > proposed).any())
        or bool((allocation & ~activation_eligible).any())
        or not torch.equal(fused[unchanged], o0[unchanged])
        or not torch.equal(fused[~valid], o0[~valid])
        or not torch.equal(allocated[~valid], torch.zeros_like(allocated[~valid]))
    ):
        raise RuntimeError("percentile-competitive residual invariant failed")
    return PercentileCompetitiveResidualResult(
        fused_logits=fused.contiguous(),
        allocated_residual_logits=allocated.contiguous(),
        within_query_midranks=midranks.contiguous(),
        activation_eligible_mask=activation_eligible.contiguous(),
        competition_mask=competition.contiguous(),
        allocation_mask=allocation.contiguous(),
    )


__all__ = [
    "CONTRACT_SHA256",
    "PercentileCompetitiveResidualResult",
    "SCHEMA",
    "allocation_contract",
    "percentile_competitive_residual",
]
