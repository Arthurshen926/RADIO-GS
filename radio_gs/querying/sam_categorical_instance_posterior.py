"""Categorical identity propagation over exact-MPR lifted SAM instances."""

from __future__ import annotations

import torch


def propagate_categorical_identity_over_proposals(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    *,
    num_proposals: int,
    proposal_view_indices: torch.Tensor | None = None,
    seed_margin_threshold: float = 0.04,
    update_margin_threshold: float = 0.04,
    semantic_tolerance: float = 0.025,
    consensus_threshold: float = 0.70,
    minimum_supporting_proposals: int = 2,
    minimum_supporting_views: int = 1,
    iterations: int = 6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Extend confident class markers only through coherent SAM proposals.

    RADIO/SigLIP logits remain the identity authority.  Exact-MPR lifted SAM
    proposals provide only an instance-equivalence relation: they may assign a
    low-margin primitive when proposal markers agree and the proposed class is
    still plausible under that primitive's original logits.  High-margin
    primitives are immutable watershed markers, preventing a large mask from
    erasing an adjacent semantic identity.
    """

    values = torch.as_tensor(scores).detach().cpu().float()
    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1)
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1)
    weights = torch.as_tensor(membership_weights).detach().cpu().float().reshape(-1)
    count, classes = map(int, values.shape)
    proposal_count = int(num_proposals)
    if values.ndim != 2 or classes < 2:
        raise ValueError("categorical scores must have shape [N,C], C>=2")
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("proposal membership sparse vectors must align")
    if (
        proposal_count <= 0
        or int(iterations) <= 0
        or int(minimum_supporting_proposals) <= 0
        or int(minimum_supporting_views) <= 0
    ):
        raise ValueError("proposal count and iterations must be positive")
    if min(seed_margin_threshold, update_margin_threshold, semantic_tolerance) < 0:
        raise ValueError("margin and tolerance values must be non-negative")
    if not 0.0 <= float(consensus_threshold) <= 1.0:
        raise ValueError("consensus threshold must lie in [0,1]")
    proposal_views = None
    if proposal_view_indices is not None:
        proposal_views = (
            torch.as_tensor(proposal_view_indices).detach().cpu().long().reshape(-1)
        )
        if proposal_views.shape != (proposal_count,) or bool((proposal_views < 0).any()):
            raise ValueError("proposal view indices must provide one non-negative view per proposal")
    elif int(minimum_supporting_views) > 1:
        raise ValueError("multiple supporting views require proposal view indices")
    valid = (
        (rows >= 0)
        & (rows < count)
        & (proposals >= 0)
        & (proposals < proposal_count)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    rows, proposals, weights = rows[valid], proposals[valid], weights[valid]
    if not rows.numel():
        return values.clone(), {
            "construction": "exact_mpr_sam_proposal_marker_watershed",
            "seed_rows": 0,
            "eligible_rows": 0,
            "owned_rows": 0,
            "changed_rows": 0,
            "changed_per_iteration": [],
        }

    top2 = torch.topk(values, k=2, dim=-1)
    original_labels = top2.indices[:, 0]
    original_margin = top2.values[:, 0] - top2.values[:, 1]
    immutable = original_margin >= float(seed_margin_threshold)
    eligible = (~immutable) & (original_margin <= float(update_margin_threshold))
    owners = torch.full_like(original_labels, -1)
    owners[immutable] = original_labels[immutable]
    changed_per_iteration: list[int] = []

    for _round in range(int(iterations)):
        edge_owners = owners[rows]
        owned_edges = edge_owners >= 0
        proposal_votes = torch.zeros((proposal_count, classes), dtype=torch.float32)
        if bool(owned_edges.any()):
            proposal_votes.index_put_(
                (proposals[owned_edges], edge_owners[owned_edges]),
                weights[owned_edges],
                accumulate=True,
            )
        proposal_support, proposal_labels = proposal_votes.max(dim=1)
        proposal_mass = proposal_votes.sum(dim=1)
        proposal_consensus = proposal_support / proposal_mass.clamp_min(1e-8)
        coherent = (
            (proposal_mass > 0)
            & (proposal_consensus >= float(consensus_threshold))
        )

        coherent_edges = coherent[proposals]
        row_votes = torch.zeros((count, classes), dtype=torch.float32)
        row_support_counts = torch.zeros((count, classes), dtype=torch.int32)
        row_view_support_counts = torch.zeros((count, classes), dtype=torch.int32)
        if bool(coherent_edges.any()):
            edge_classes = proposal_labels[proposals[coherent_edges]]
            edge_rows = rows[coherent_edges]
            edge_proposals = proposals[coherent_edges]
            row_votes.index_put_(
                (edge_rows, edge_classes),
                weights[coherent_edges] * proposal_consensus[edge_proposals],
                accumulate=True,
            )
            row_support_counts.index_put_(
                (edge_rows, edge_classes),
                torch.ones(int(coherent_edges.sum()), dtype=torch.int32),
                accumulate=True,
            )
            if proposal_views is not None:
                edge_views = proposal_views[edge_proposals]
                num_views = int(proposal_views.max()) + 1
                row_class_view_keys = torch.unique(
                    ((edge_rows * classes + edge_classes) * num_views) + edge_views
                )
                row_class_keys = torch.div(
                    row_class_view_keys, num_views, rounding_mode="floor"
                )
                row_view_support_counts = torch.bincount(
                    row_class_keys, minlength=count * classes
                ).reshape(count, classes).to(torch.int32)
            else:
                row_view_support_counts.copy_(row_support_counts)
        support, proposed = row_votes.max(dim=1)
        mass = row_votes.sum(dim=1)
        row_consensus = support / mass.clamp_min(1e-8)
        proposed_score = values.gather(1, proposed[:, None])[:, 0]
        supporting_proposals = row_support_counts.gather(1, proposed[:, None])[:, 0]
        supporting_views = row_view_support_counts.gather(1, proposed[:, None])[:, 0]
        plausible = (values.max(dim=1).values - proposed_score) <= float(
            semantic_tolerance
        )
        accept = (
            eligible
            & (owners < 0)
            & (mass > 0)
            & (row_consensus >= float(consensus_threshold))
            & (supporting_proposals >= int(minimum_supporting_proposals))
            & (supporting_views >= int(minimum_supporting_views))
            & plausible
        )
        accepted = torch.where(accept)[0]
        owners[accepted] = proposed[accepted]
        changed_per_iteration.append(int(accepted.numel()))
        if not accepted.numel():
            break

    output = values.clone()
    reassigned = eligible & (owners >= 0) & (owners != original_labels)
    reassigned_rows = torch.where(reassigned)[0]
    if reassigned_rows.numel():
        winner = values[reassigned_rows].max(dim=1).values
        output[reassigned_rows, owners[reassigned_rows]] = winner + 1e-6
    return output, {
        "construction": "exact_mpr_sam_proposal_marker_watershed",
        "seed_margin_threshold": float(seed_margin_threshold),
        "update_margin_threshold": float(update_margin_threshold),
        "semantic_tolerance": float(semantic_tolerance),
        "consensus_threshold": float(consensus_threshold),
        "minimum_supporting_proposals": int(minimum_supporting_proposals),
        "minimum_supporting_views": int(minimum_supporting_views),
        "iterations": int(iterations),
        "proposal_count": proposal_count,
        "membership_count": int(rows.numel()),
        "seed_rows": int(immutable.sum()),
        "eligible_rows": int(eligible.sum()),
        "owned_rows": int((owners >= 0).sum()),
        "changed_rows": int(reassigned.sum()),
        "changed_per_iteration": changed_per_iteration,
    }
