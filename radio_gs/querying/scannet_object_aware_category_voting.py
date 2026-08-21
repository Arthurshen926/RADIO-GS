"""Bounded object-aware categorical voting for ScanNet OVS.

Official SAM memberships are category-free object observations.  Category
identity always comes from primitive RADIO logits; object observations may
only denoise an uncertain *thing* primitive through a bounded residual.
"""

from __future__ import annotations

from collections.abc import Collection

import torch


def object_aware_category_vote(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    *,
    num_proposals: int,
    class_ids: Collection[int],
    stuff_class_ids: Collection[int] = (1, 2, 22),
    strength: float = 0.0,
    residual_budget: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Fuse primitive logits with object-level votes without creating identity.

    The operator is proposal-permutation equivariant and class-permutation
    equivariant when ``class_ids`` are permuted with score columns.  Its only
    semantic prior is the standard thing/stuff role.  ``strength=0`` is a
    bitwise replay of ``scores``.

    The per-row mixing coefficient is analytic.  It is the product of:

    * a fixed safety budget;
    * proposal membership coverage weighted by proposal agreement and margin;
    * the resulting object-vote margin; and
    * primitive uncertainty.

    Consequently a proposal cannot supply a category that was absent from the
    primitive score bank, and confident or stuff predictions remain fixed.
    """

    values = torch.as_tensor(scores).detach().cpu().float()
    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1)
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1)
    weights = torch.as_tensor(membership_weights).detach().cpu().float().reshape(-1)
    ids = tuple(int(value) for value in class_ids)
    stuff = frozenset(int(value) for value in stuff_class_ids)
    proposal_count = int(num_proposals)
    gain = float(strength)
    budget = float(residual_budget)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must have shape [N,C], C>=2")
    if len(ids) != values.shape[1] or len(set(ids)) != len(ids):
        raise ValueError("class_ids must uniquely identify every score column")
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("sparse proposal membership axes differ")
    if proposal_count <= 0:
        raise ValueError("num_proposals must be positive")
    if not 0.0 <= gain <= 1.0 or not 0.0 <= budget <= 1.0:
        raise ValueError("strength and residual_budget must be in [0,1]")
    if gain == 0.0:
        return values.clone(), {
            "construction": "bounded_thing_object_vote_v1",
            "enabled": False,
            "strength": 0.0,
            "changed_rows": 0,
            "eligible_rows": 0,
        }

    count, classes = values.shape
    valid = (
        (rows >= 0)
        & (rows < count)
        & (proposals >= 0)
        & (proposals < proposal_count)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    rows, proposals, weights = rows[valid], proposals[valid], weights[valid]
    if rows.numel() == 0:
        return values.clone(), {
            "construction": "bounded_thing_object_vote_v1",
            "enabled": False,
            "strength": gain,
            "changed_rows": 0,
            "eligible_rows": 0,
        }

    # Row standardization removes a query-set-independent affine score gauge.
    centered = values - values.mean(dim=1, keepdim=True)
    standardized = centered / values.std(dim=1, keepdim=True).clamp_min(1.0e-6)
    primitive_probability = torch.softmax(standardized, dim=1)
    primitive_top2 = torch.topk(standardized, 2, dim=1)
    primitive_label = primitive_top2.indices[:, 0]
    primitive_margin = primitive_top2.values[:, 0] - primitive_top2.values[:, 1]

    proposal_sum = values.new_zeros((proposal_count, classes))
    proposal_mass = values.new_zeros((proposal_count,))
    proposal_sum.index_add_(0, proposals, standardized[rows] * weights[:, None])
    proposal_mass.index_add_(0, proposals, weights)
    proposal_probability = torch.softmax(
        proposal_sum / proposal_mass.clamp_min(1.0e-8)[:, None], dim=1
    )
    proposal_top2 = torch.topk(proposal_probability, 2, dim=1)
    proposal_label = proposal_top2.indices[:, 0]
    proposal_margin = (
        (proposal_top2.values[:, 0] - proposal_top2.values[:, 1])
        / proposal_top2.values[:, 0].clamp_min(1.0e-8)
    ).clamp(0.0, 1.0)
    agreement_mass = values.new_zeros((proposal_count,))
    agreement_mass.index_add_(
        0,
        proposals,
        weights * (primitive_label[rows] == proposal_label[proposals]).float(),
    )
    proposal_agreement = agreement_mass / proposal_mass.clamp_min(1.0e-8)
    proposal_reliability = proposal_agreement.clamp(0.0, 1.0) * proposal_margin

    edge_reliability = weights * proposal_reliability[proposals]
    row_vote = values.new_zeros((count, classes))
    reliable_mass = values.new_zeros((count,))
    observed_mass = values.new_zeros((count,))
    row_vote.index_add_(
        0, rows, proposal_probability[proposals] * edge_reliability[:, None]
    )
    reliable_mass.index_add_(0, rows, edge_reliability)
    observed_mass.index_add_(0, rows, weights)
    object_probability = row_vote / reliable_mass.clamp_min(1.0e-8)[:, None]
    object_top2 = torch.topk(object_probability, 2, dim=1)
    object_label = object_top2.indices[:, 0]
    object_margin = (
        (object_top2.values[:, 0] - object_top2.values[:, 1])
        / object_top2.values[:, 0].clamp_min(1.0e-8)
    ).clamp(0.0, 1.0)
    coverage = (reliable_mass / observed_mass.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    primitive_uncertainty = torch.exp(-primitive_margin.clamp_min(0.0))
    mixing = gain * budget * coverage * object_margin * primitive_uncertainty

    class_is_stuff = torch.tensor([value in stuff for value in ids], dtype=torch.bool)
    supported = reliable_mass > 0
    thing_only = (~class_is_stuff[primitive_label]) & (~class_is_stuff[object_label])
    eligible = supported & thing_only
    mixing = mixing * eligible.float()
    posterior = (
        (1.0 - mixing[:, None]) * primitive_probability
        + mixing[:, None] * object_probability
    )
    output = posterior.clamp_min(1.0e-12).log()
    changed = output.argmax(dim=1) != primitive_label
    return output, {
        "construction": "bounded_thing_object_vote_v1",
        "enabled": bool(eligible.any()),
        "strength": gain,
        "residual_budget": budget,
        "proposal_count": proposal_count,
        "membership_count": int(rows.numel()),
        "eligible_rows": int(eligible.sum()),
        "changed_rows": int(changed.sum()),
        "mean_mixing": float(mixing.mean()),
        "maximum_mixing": float(mixing.max()),
    }


__all__ = ["object_aware_category_vote"]
