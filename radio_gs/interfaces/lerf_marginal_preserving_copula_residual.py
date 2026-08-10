"""Bounded copula residual for calibrated LERF query scores.

The accepted score vector is the calibration authority.  A second score
vector may only change the ordering *inside fixed accepted-rank blocks*; the
accepted empirical marginal is then assigned back to the new ordering.  This
gives three useful, query-local guarantees without looking at a target mask:

* the multiset of valid output scores is bitwise identical to the accepted
  multiset, so min/max and the selected count at every threshold are exact;
* an item can move by at most ``block_size - 1`` accepted ranks;
* zero strength or a singleton block is fail-closed; each zero-reliability
  item is a fixed-rank barrier that neither moves nor can be crossed.

The operation is intended for the final primitive score layer, after the
frozen canonical-negative/KNN/min-max compiler.  Applying it to raw cosine
scores does not by itself preserve a later nonlinear compiler's marginal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


CONTRACT = "lerf-accepted-marginal-bounded-copula-residual-v1"


@dataclass(frozen=True)
class MarginalPreservingResidualResult:
    """Output and auditable invariants for one score tensor."""

    scores: torch.Tensor
    maximum_rank_displacement: int
    changed_valid_count: int
    valid_count: int
    block_size: int
    marginal_exact: bool


def _stable_ranks(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable ascending order and inverse integer ranks."""

    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device)
    return order, ranks


def _validate_inputs(
    accepted_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    valid: torch.Tensor,
    reliability: torch.Tensor | None,
    *,
    strength: float,
    maximum_rank_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    accepted = torch.as_tensor(accepted_scores)
    candidate = torch.as_tensor(candidate_scores)
    mask = torch.as_tensor(valid)
    if (
        accepted.ndim < 1
        or accepted.shape != candidate.shape
        or not accepted.is_floating_point()
        or not candidate.is_floating_point()
        or accepted.device != candidate.device
        or not bool(torch.isfinite(accepted).all())
        or not bool(torch.isfinite(candidate).all())
    ):
        raise ValueError("accepted/candidate score tensors differ")
    if mask.device != accepted.device or mask.dtype != torch.bool:
        raise ValueError("valid mask dtype/device differs")
    if mask.shape == accepted.shape:
        expanded_mask = mask
    elif mask.shape == accepted.shape[-1:]:
        expanded_mask = mask.expand_as(accepted)
    else:
        raise ValueError("valid mask must match scores or their final axis")
    if not bool(expanded_mask.any(dim=-1).all()):
        raise ValueError("every score row requires at least one valid item")
    if not math.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("residual strength must lie in [0,1]")
    if (
        not math.isfinite(float(maximum_rank_fraction))
        or not 0.0 <= float(maximum_rank_fraction) <= 1.0
    ):
        raise ValueError("maximum rank fraction must lie in [0,1]")
    if reliability is None:
        gate = torch.ones_like(accepted)
    else:
        raw_gate = torch.as_tensor(reliability)
        if raw_gate.device != accepted.device or not raw_gate.is_floating_point():
            raise ValueError("reliability dtype/device differs")
        if raw_gate.shape == accepted.shape:
            gate = raw_gate
        elif raw_gate.shape == accepted.shape[-1:]:
            gate = raw_gate.expand_as(accepted)
        else:
            raise ValueError("reliability must match scores or their final axis")
        if (
            not bool(torch.isfinite(gate).all())
            or bool((gate < 0).any())
            or bool((gate > 1).any())
        ):
            raise ValueError("reliability must be finite in [0,1]")
    return accepted, candidate, expanded_mask, gate


def marginal_preserving_copula_residual(
    accepted_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    strength: float,
    maximum_rank_fraction: float,
    reliability: torch.Tensor | None = None,
) -> MarginalPreservingResidualResult:
    """Inject bounded candidate ordering while preserving accepted marginals.

    Score rows are all dimensions except the final one.  The final dimension
    is the item/primitive axis.  Blocks are consecutive in accepted stable-rank
    order.  Within a block, items are ordered by a convex combination of their
    accepted and candidate global percentile ranks, with an optional
    reliability gate on the candidate displacement.
    """

    accepted, candidate, mask, gate = _validate_inputs(
        accepted_scores,
        candidate_scores,
        valid,
        reliability,
        strength=strength,
        maximum_rank_fraction=maximum_rank_fraction,
    )
    original_shape = accepted.shape
    item_count = original_shape[-1]
    accepted_rows = accepted.reshape(-1, item_count)
    candidate_rows = candidate.reshape(-1, item_count)
    mask_rows = mask.reshape(-1, item_count)
    gate_rows = gate.reshape(-1, item_count)
    output = accepted_rows.clone()
    maximum_displacement = 0
    changed = 0
    total_valid = 0
    largest_block = 1

    for row_index in range(accepted_rows.shape[0]):
        active = torch.where(mask_rows[row_index])[0]
        count = int(active.numel())
        total_valid += count
        if count <= 1 or strength == 0.0 or maximum_rank_fraction == 0.0:
            continue
        block_size = max(1, min(count, int(math.ceil(maximum_rank_fraction * count))))
        largest_block = max(largest_block, block_size)
        if block_size == 1:
            continue
        anchor = accepted_rows[row_index, active]
        proposal = candidate_rows[row_index, active]
        active_gate = gate_rows[row_index, active]
        anchor_order, anchor_rank = _stable_ranks(anchor)
        _, candidate_rank = _stable_ranks(proposal)
        denominator = float(max(count - 1, 1))
        anchor_percentile = anchor_rank.to(torch.float64) / denominator
        candidate_percentile = candidate_rank.to(torch.float64) / denominator
        fused_key = anchor_percentile + (
            float(strength)
            * active_gate.to(torch.float64)
            * (candidate_percentile - anchor_percentile)
        )

        assigned_rank = torch.empty_like(anchor_rank)
        sorted_anchor_values = anchor[anchor_order]
        # A deterministic half-block prefix avoids pinning common operating
        # points (for example a top decile) to a block boundary.  The prefix
        # remains an accepted-rank block and therefore keeps the same bound.
        prefix = block_size // 2
        boundaries = [0]
        if prefix > 0:
            boundaries.append(prefix)
        while boundaries[-1] < count:
            boundaries.append(min(count, boundaries[-1] + block_size))
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            block_items = anchor_order[start:stop]
            # A zero gate represents an authoritative observation.  Make it a
            # fixed-rank barrier and reorder only the open intervals between
            # barriers, so another item cannot indirectly displace it.
            barrier_positions = torch.where(active_gate[block_items] == 0)[0].tolist()
            segment_start = 0
            for barrier_position in barrier_positions + [stop - start]:
                if barrier_position > segment_start:
                    segment_items = block_items[segment_start:barrier_position]
                    local_order = torch.argsort(fused_key[segment_items], stable=True)
                    ordered_items = segment_items[local_order]
                    absolute_start = start + segment_start
                    absolute_stop = start + barrier_position
                    target_ranks = torch.arange(
                        absolute_start, absolute_stop, device=anchor.device
                    )
                    assigned_rank[ordered_items] = target_ranks
                    output[row_index, active[ordered_items]] = sorted_anchor_values[
                        absolute_start:absolute_stop
                    ]
                if barrier_position < stop - start:
                    barrier_item = block_items[barrier_position]
                    barrier_rank = start + barrier_position
                    assigned_rank[barrier_item] = barrier_rank
                    output[row_index, active[barrier_item]] = sorted_anchor_values[
                        barrier_rank
                    ]
                segment_start = barrier_position + 1

        displacement = (assigned_rank - anchor_rank).abs()
        maximum_displacement = max(
            maximum_displacement, int(displacement.max().item())
        )
        changed += int(
            (output[row_index, active] != accepted_rows[row_index, active]).sum().item()
        )

    scores = output.reshape(original_shape).contiguous()
    # The exact check is deliberately bitwise and remains on the score device.
    marginal_exact = True
    for row_index in range(accepted_rows.shape[0]):
        active = mask_rows[row_index]
        if not torch.equal(
            torch.sort(output[row_index, active], stable=True).values,
            torch.sort(accepted_rows[row_index, active], stable=True).values,
        ):
            marginal_exact = False
            break
    if not marginal_exact:
        raise AssertionError("accepted empirical marginal was not preserved exactly")
    if maximum_displacement > largest_block - 1:
        raise AssertionError("bounded copula residual exceeded its rank budget")
    if not torch.equal(scores[~mask], accepted[~mask]):
        raise AssertionError("bounded copula residual modified an invalid score")
    return MarginalPreservingResidualResult(
        scores=scores,
        maximum_rank_displacement=maximum_displacement,
        changed_valid_count=changed,
        valid_count=total_valid,
        block_size=largest_block,
        marginal_exact=True,
    )


def marginal_preserving_primitive_query_scores(
    accepted_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    strength: float,
    maximum_rank_fraction: float,
    reliability: torch.Tensor | None = None,
) -> MarginalPreservingResidualResult:
    """Formal LERF adapter for primitive-query score matrices ``[N,Q]``.

    Keeping this shape check at the public benchmark boundary prevents the
    generic row-wise copula kernel from silently treating queries as items.
    Reliability is query-free on the primitive axis and is expanded only
    after the explicit transpose.
    """

    accepted = torch.as_tensor(accepted_scores)
    candidate = torch.as_tensor(candidate_scores)
    mask = torch.as_tensor(valid)
    if (
        accepted.ndim != 2
        or candidate.shape != accepted.shape
        or mask.ndim != 1
        or mask.shape[0] != accepted.shape[0]
        or mask.dtype != torch.bool
        or mask.device != accepted.device
    ):
        raise ValueError("formal primitive-query scores must be [N,Q] with valid [N]")
    primitive_reliability: torch.Tensor | None
    if reliability is None:
        primitive_reliability = None
    else:
        raw = torch.as_tensor(reliability)
        if raw.ndim != 1 or raw.shape != mask.shape or raw.device != accepted.device:
            raise ValueError("formal primitive reliability must be [N]")
        primitive_reliability = raw
    internal = marginal_preserving_copula_residual(
        accepted.T.contiguous(),
        candidate.T.contiguous(),
        mask,
        strength=strength,
        maximum_rank_fraction=maximum_rank_fraction,
        reliability=primitive_reliability,
    )
    return MarginalPreservingResidualResult(
        scores=internal.scores.T.contiguous(),
        maximum_rank_displacement=internal.maximum_rank_displacement,
        changed_valid_count=internal.changed_valid_count,
        valid_count=internal.valid_count,
        block_size=internal.block_size,
        marginal_exact=internal.marginal_exact,
    )


__all__ = [
    "CONTRACT",
    "MarginalPreservingResidualResult",
    "marginal_preserving_copula_residual",
    "marginal_preserving_primitive_query_scores",
]
