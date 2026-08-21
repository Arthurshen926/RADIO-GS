"""Source-only learned compact object affinity for categorical readouts.

The learned state is one global linear projection from the frozen canonical
capability feature to a 16-D object code.  Exact-MPR SAM observations provide
ternary proposal relations: affirmative cross-view co-membership is ``same``;
simultaneously observed, comparable-scale, disjoint masks are ``different``;
all unsupported or granularity-conflicted pairs remain ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceProposalRelations:
    left: torch.Tensor
    right: torch.Tensor
    relation: torch.Tensor
    soft_cosine: torch.Tensor
    stats: dict[str, int | float | str]


class CompactObjectAffinity(nn.Module):
    """Globally shared 16-D projection with no category or query parameters."""

    def __init__(self, input_dim: int = 1536, object_dim: int = 16, seed: int = 20260821) -> None:
        super().__init__()
        if input_dim <= 0 or object_dim <= 0:
            raise ValueError("compact affinity dimensions must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        weight = torch.randn(object_dim, input_dim, generator=generator) / input_dim**0.5
        self.weight = nn.Parameter(weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.weight.shape[1]:
            raise ValueError("canonical capability feature axis differs")
        return F.normalize(F.linear(features.float(), self.weight), dim=-1)


def pool_proposal_features(
    features: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_proposals: int,
) -> torch.Tensor:
    """Exact-MPR weighted mean of frozen Gaussian capability features."""

    value = torch.as_tensor(features).detach().cpu().float()
    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1)
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1)
    membership = torch.as_tensor(weights).detach().cpu().float().reshape(-1)
    count = int(num_proposals)
    if value.ndim != 2 or count <= 0 or not (rows.shape == proposals.shape == membership.shape):
        raise ValueError("proposal pooling axes differ")
    valid = (
        (rows >= 0) & (rows < value.shape[0])
        & (proposals >= 0) & (proposals < count)
        & torch.isfinite(membership) & (membership > 0)
    )
    rows, proposals, membership = rows[valid], proposals[valid], membership[valid]
    index = torch.stack((proposals, rows))
    matrix = torch.sparse_coo_tensor(index, membership, (count, value.shape[0])).coalesce()
    mass = torch.sparse.sum(matrix, dim=1).to_dense().clamp_min(1.0e-8)
    return torch.sparse.mm(matrix, value) / mass[:, None]


def build_source_proposal_relations(
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_area_fraction: torch.Tensor,
    *,
    num_rows: int,
    num_proposals: int,
    same_soft_cosine: float = 0.5,
    different_soft_cosine: float = 0.05,
    minimum_area_ratio: float = 0.25,
) -> SourceProposalRelations:
    """Build fixed ternary relation authority without benchmark supervision.

    ``same`` requires a reciprocal-best cross-view majority overlap.
    ``different`` is restricted to same-view, comparable-scale masks whose
    lifted supports are essentially disjoint.  Same-view nesting and every
    unsupported cross-view pair stay unknown rather than becoming negatives.
    """

    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1).numpy()
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1).numpy()
    membership = torch.as_tensor(weights).detach().cpu().float().reshape(-1).numpy()
    views = torch.as_tensor(proposal_view_indices).detach().cpu().long().reshape(-1).numpy()
    areas = torch.as_tensor(proposal_area_fraction).detach().cpu().float().reshape(-1).numpy()
    count, row_count = int(num_proposals), int(num_rows)
    if not (rows.shape == proposals.shape == membership.shape) or views.shape != (count,) or areas.shape != (count,):
        raise ValueError("relation authority axes differ")
    if not (0 < same_soft_cosine <= 1 and 0 <= different_soft_cosine < same_soft_cosine):
        raise ValueError("same/different source boundaries differ")
    if not 0 < minimum_area_ratio <= 1:
        raise ValueError("minimum_area_ratio must be in (0,1]")
    valid = (
        (rows >= 0) & (rows < row_count) & (proposals >= 0) & (proposals < count)
        & np.isfinite(membership) & (membership > 0)
    )
    matrix = sparse.csr_matrix((membership[valid], (proposals[valid], rows[valid])), shape=(count, row_count))
    gram = (matrix @ matrix.T).toarray()
    norm = np.sqrt(np.maximum(np.diag(gram), 1.0e-12))
    cosine = gram / norm[:, None] / norm[None, :]

    relations: dict[tuple[int, int], tuple[int, float]] = {}
    unique_views = np.unique(views)
    for view in unique_views:
        local = np.where(views == view)[0]
        for position, left in enumerate(local):
            for right in local[position + 1 :]:
                ratio = min(float(areas[left]), float(areas[right])) / max(float(areas[left]), float(areas[right]), 1.0e-12)
                score = float(cosine[left, right])
                if ratio >= minimum_area_ratio and score <= different_soft_cosine:
                    relations[(int(left), int(right))] = (0, score)
    for left_position, left_view in enumerate(unique_views):
        left_group = np.where(views == left_view)[0]
        for right_view in unique_views[left_position + 1 :]:
            right_group = np.where(views == right_view)[0]
            pair = cosine[np.ix_(left_group, right_group)]
            if pair.size == 0:
                continue
            left_best = pair.argmax(axis=1)
            right_best = pair.argmax(axis=0)
            for left_local, right_local in enumerate(left_best):
                score = float(pair[left_local, right_local])
                if right_best[right_local] == left_local and score >= same_soft_cosine:
                    left, right = int(left_group[left_local]), int(right_group[right_local])
                    relations[(min(left, right), max(left, right))] = (1, score)
    ordered = sorted(relations.items())
    left = torch.tensor([key[0] for key, _ in ordered], dtype=torch.long)
    right = torch.tensor([key[1] for key, _ in ordered], dtype=torch.long)
    relation = torch.tensor([value[0] for _, value in ordered], dtype=torch.int8)
    overlap = torch.tensor([value[1] for _, value in ordered], dtype=torch.float32)
    return SourceProposalRelations(
        left, right, relation, overlap,
        {
            "construction": "source_exact_mpr_ternary_proposal_relation_v1",
            "same_soft_cosine": float(same_soft_cosine),
            "different_soft_cosine": float(different_soft_cosine),
            "minimum_area_ratio": float(minimum_area_ratio),
            "same_edges": int((relation == 1).sum()),
            "different_edges": int((relation == 0).sum()),
            "unknown_policy": "excluded_not_negative",
        },
    )


def balanced_relation_loss(logits: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """Class-balanced Bernoulli log score over known source relations."""

    score = torch.as_tensor(logits).reshape(-1)
    label = torch.as_tensor(relation, device=score.device).reshape(-1).to(torch.int8)
    same, different = label == 1, label == 0
    if not bool(same.any()) or not bool(different.any()):
        raise ValueError("proper relation loss requires same and different outcomes")
    return 0.5 * (
        F.binary_cross_entropy_with_logits(score[same], torch.ones_like(score[same]))
        + F.binary_cross_entropy_with_logits(score[different], torch.zeros_like(score[different]))
    )


def relation_proper_metrics(logits: torch.Tensor, relation: torch.Tensor) -> dict[str, float]:
    """Threshold-free source reliability metrics."""

    score = torch.as_tensor(logits).detach().cpu().float().reshape(-1)
    label = torch.as_tensor(relation).detach().cpu().to(torch.int8).reshape(-1)
    same, different = score[label == 1], score[label == 0]
    if not same.numel() or not different.numel():
        return {"balanced_log_score": float("nan"), "balanced_brier": float("nan"), "auc": float("nan")}
    probability = score.sigmoid()
    brier = 0.5 * (((probability[label == 1] - 1) ** 2).mean() + (probability[label == 0] ** 2).mean())
    log_score = balanced_relation_loss(score, label)
    wins = score.new_zeros(())
    for chunk in same.split(4096):
        delta = chunk[:, None] - different[None, :]
        wins += (delta > 0).sum() + 0.5 * (delta == 0).sum()
    return {
        "balanced_log_score": float(log_score),
        "balanced_brier": float(brier),
        "auc": float(wins / (same.numel() * different.numel())),
    }


__all__ = [
    "CompactObjectAffinity",
    "SourceProposalRelations",
    "balanced_relation_loss",
    "build_source_proposal_relations",
    "pool_proposal_features",
    "relation_proper_metrics",
]
