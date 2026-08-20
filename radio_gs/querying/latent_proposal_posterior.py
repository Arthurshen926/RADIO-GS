"""Probability-correct latent proposal/null object posterior operators.

The operator in this module deliberately does not estimate proposal logits.
Those logits must come from one frozen, source-trained scorer.  This seam keeps
the probability model exact while preventing a benchmark-tuned threshold from
silently becoming proposal identity evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


UNKNOWN_RELATION = -1
DIFFERENT_RELATION = 0
SAME_RELATION = 1


@dataclass(frozen=True)
class LatentProposalPosterior:
    """One normalized null/proposal mixture and its primitive posterior."""

    probability: torch.Tensor
    null_probability: torch.Tensor
    proposal_probability: torch.Tensor


def ternary_comembership_authority(
    jointly_visible: torch.Tensor,
    same_instance_evidence: torch.Tensor,
    different_instance_evidence: torch.Tensor,
) -> torch.Tensor:
    """Return same/different/unknown labels without false occlusion negatives.

    A relation is trainable only when both endpoints are jointly visible.
    Missing co-membership, occlusion, and unsupported pairs remain ``-1`` and
    must be ignored by a downstream proper scoring loss.
    """

    visible = torch.as_tensor(jointly_visible).bool()
    same = torch.as_tensor(same_instance_evidence).bool()
    different = torch.as_tensor(different_instance_evidence).bool()
    if visible.shape != same.shape or visible.shape != different.shape:
        raise ValueError("ternary co-membership evidence axes differ")
    if bool((same & different).any()):
        raise ValueError("same and different evidence overlap")
    if bool(((same | different) & ~visible).any()):
        raise ValueError("relation evidence requires joint visibility")
    label = torch.full(
        visible.shape,
        UNKNOWN_RELATION,
        dtype=torch.int8,
        device=visible.device,
    )
    label[different] = DIFFERENT_RELATION
    label[same] = SAME_RELATION
    return label


def latent_proposal_null_posterior(
    primitive_probability: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_probability: torch.Tensor,
    proposal_logits: torch.Tensor,
    null_logits: torch.Tensor,
    *,
    proposal_valid: torch.Tensor | None = None,
) -> LatentProposalPosterior:
    """Marginalize one latent object proposal plus an explicit null branch.

    For query ``q`` and Gaussian ``i`` this computes

    ``P(y_i=1|q) = pi_0(q) p_primitive(i,q) + sum_k pi_k(q) m(i,k)``.

    ``proposal_logits`` and ``null_logits`` are normalized together.  Invalid
    proposals receive exactly zero mass.  A query with no valid proposals takes
    the primitive branch bit-for-bit instead of relying on floating point
    softmax fallback.  Sparse memberships must contain at most one value per
    Gaussian/proposal pair so the result remains a convex probability mixture.
    """

    primitive = torch.as_tensor(primitive_probability)
    rows = torch.as_tensor(row_indices, device=primitive.device).long()
    proposals = torch.as_tensor(proposal_indices, device=primitive.device).long()
    membership = torch.as_tensor(
        membership_probability, device=primitive.device, dtype=primitive.dtype
    )
    logits = torch.as_tensor(
        proposal_logits, device=primitive.device, dtype=primitive.dtype
    )
    null = torch.as_tensor(null_logits, device=primitive.device, dtype=primitive.dtype)
    if primitive.ndim != 2 or not primitive.is_floating_point():
        raise ValueError("primitive probability must be floating [N,Q]")
    num_rows, num_queries = map(int, primitive.shape)
    if logits.ndim != 2 or int(logits.shape[1]) != num_queries:
        raise ValueError("proposal logits must be [K,Q]")
    num_proposals = int(logits.shape[0])
    if null.shape != (num_queries,):
        raise ValueError("null logits must be [Q]")
    if not (rows.ndim == proposals.ndim == membership.ndim == 1) or not (
        rows.shape == proposals.shape == membership.shape
    ):
        raise ValueError("sparse proposal membership axes differ")
    if not bool(torch.isfinite(primitive).all()) or bool(
        ((primitive < 0) | (primitive > 1)).any()
    ):
        raise ValueError("primitive probability must be finite in [0,1]")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(null).all()):
        raise ValueError("latent proposal logits must be finite")
    if rows.numel() and (
        int(rows.min()) < 0
        or int(rows.max()) >= num_rows
        or int(proposals.min()) < 0
        or int(proposals.max()) >= num_proposals
    ):
        raise ValueError("sparse proposal membership index is out of range")
    if not bool(torch.isfinite(membership).all()) or bool(
        ((membership < 0) | (membership > 1)).any()
    ):
        raise ValueError("proposal membership must be finite in [0,1]")
    if rows.numel():
        pair_ids = rows * num_proposals + proposals
        if int(torch.unique(pair_ids).numel()) != int(pair_ids.numel()):
            raise ValueError("proposal membership repeats a Gaussian/proposal pair")

    valid = (
        torch.ones_like(logits, dtype=torch.bool)
        if proposal_valid is None
        else torch.as_tensor(proposal_valid, device=primitive.device).bool()
    )
    if valid.shape != logits.shape:
        raise ValueError("proposal-valid mask must be [K,Q]")
    any_valid = valid.any(dim=0)
    proposal_probability = torch.zeros_like(logits)
    null_probability = torch.ones_like(null)
    if bool(any_valid.any()):
        active_logits = logits[:, any_valid].masked_fill(~valid[:, any_valid], -torch.inf)
        joint_logits = torch.cat((null[any_valid][None], active_logits), dim=0)
        joint_probability = torch.softmax(joint_logits, dim=0)
        null_probability[any_valid] = joint_probability[0]
        proposal_probability[:, any_valid] = joint_probability[1:]

    region_probability = torch.zeros_like(primitive)
    if rows.numel():
        edge_probability = membership[:, None] * proposal_probability[proposals]
        region_probability.index_add_(0, rows, edge_probability)
    probability = null_probability[None] * primitive + region_probability
    # This is a useful contract assertion, not a numerical clamp: a convex
    # mixture of valid probabilities cannot leave [0,1].
    tolerance = 16 * torch.finfo(probability.dtype).eps
    if bool(((probability < -tolerance) | (probability > 1 + tolerance)).any()):
        raise RuntimeError("latent proposal posterior left probability bounds")
    probability = probability.clamp(0, 1)
    if bool((~any_valid).any()):
        probability[:, ~any_valid] = primitive[:, ~any_valid]
    return LatentProposalPosterior(
        probability=probability,
        null_probability=null_probability,
        proposal_probability=proposal_probability,
    )


__all__ = [
    "DIFFERENT_RELATION",
    "LatentProposalPosterior",
    "SAME_RELATION",
    "UNKNOWN_RELATION",
    "latent_proposal_null_posterior",
    "ternary_comembership_authority",
]
