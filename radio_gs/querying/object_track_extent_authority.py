"""Compile source-only object-track support for extent completion.

Single target-view masks supervise only the visible object fragment.  This
module groups all confirmed same-object proposals and aggregates their sparse
exact-MPR memberships without turning missing evidence into background.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectTrackExtentAuthority:
    """Sparse, positive-unlabelled supervision for one physical object track."""

    object_id: int
    proposal_rows: torch.Tensor
    positive_rows: torch.Tensor
    positive_probability: torch.Tensor
    positive_view_count: torch.Tensor
    explicit_negative_rows: torch.Tensor


def compile_object_track_extent_authority(
    episode_object_ids: torch.Tensor,
    episode_query_proposals: torch.Tensor,
    episode_target_proposals: torch.Tensor,
    proposal_soft_rows: list[torch.Tensor],
    proposal_soft_values: list[torch.Tensor],
    episode_explicit_negative_rows: list[torch.Tensor],
) -> dict[int, ObjectTrackExtentAuthority]:
    """Aggregate view fragments into track support while preserving unknowns.

    A Gaussian is a positive when at least one confirmed view assigns it
    non-zero exact-MPR membership.  Repeated independent observations increase
    its noisy-OR probability and are recorded separately as an authority count.
    Only rows explicitly labelled as another instance become negatives; an
    absent sparse membership remains unknown.
    """

    objects = torch.as_tensor(episode_object_ids).detach().cpu().long().reshape(-1)
    queries = torch.as_tensor(episode_query_proposals).detach().cpu().long().reshape(-1)
    targets = torch.as_tensor(episode_target_proposals).detach().cpu().long().reshape(-1)
    if not (objects.shape == queries.shape == targets.shape):
        raise ValueError("object-track episode axes differ")
    if len(proposal_soft_rows) != len(proposal_soft_values):
        raise ValueError("object-track proposal membership axes differ")
    if len(episode_explicit_negative_rows) != objects.numel():
        raise ValueError("object-track negative episode axis differs")

    result: dict[int, ObjectTrackExtentAuthority] = {}
    for object_id in torch.unique(objects, sorted=True).tolist():
        episode_rows = torch.where(objects == int(object_id))[0]
        proposals = torch.unique(
            torch.cat((queries[episode_rows], targets[episode_rows])), sorted=True
        )
        track_rows: list[torch.Tensor] = []
        track_values: list[torch.Tensor] = []
        for proposal in proposals.tolist():
            rows = torch.as_tensor(proposal_soft_rows[proposal]).detach().cpu().long()
            values = torch.as_tensor(proposal_soft_values[proposal]).detach().cpu().float()
            if rows.shape != values.shape:
                raise ValueError("object-track sparse proposal membership differs")
            valid=values>0
            track_rows.append(rows[valid]);track_values.append(values[valid].clamp(0.0,1.0))
        concatenated_rows=torch.cat(track_rows) if track_rows else torch.empty(0,dtype=torch.long)
        concatenated_values=torch.cat(track_values) if track_values else torch.empty(0)
        positive_rows,inverse=torch.unique(concatenated_rows,sorted=True,return_inverse=True)
        log_survival=torch.zeros(positive_rows.numel(),dtype=torch.float32)
        log_survival.scatter_add_(0,inverse,torch.log1p(-concatenated_values.clamp_max(1.0-1e-7)))
        positive_probability=1.0-log_survival.exp()
        positive_view_count=torch.zeros(positive_rows.numel(),dtype=torch.long)
        positive_view_count.scatter_add_(0,inverse,torch.ones_like(inverse))
        negatives = [
            torch.as_tensor(episode_explicit_negative_rows[index]).detach().cpu().long()
            for index in episode_rows.tolist()
            if torch.as_tensor(episode_explicit_negative_rows[index]).numel()
        ]
        explicit_negative_rows = (
            torch.unique(torch.cat(negatives), sorted=True)
            if negatives else torch.empty(0, dtype=torch.long)
        )
        if positive_rows.numel() and explicit_negative_rows.numel():
            explicit_negative_rows = explicit_negative_rows[
                ~torch.isin(explicit_negative_rows, positive_rows)
            ]
        result[int(object_id)] = ObjectTrackExtentAuthority(
            object_id=int(object_id), proposal_rows=proposals,
            positive_rows=positive_rows,
            positive_probability=positive_probability,
            positive_view_count=positive_view_count,
            explicit_negative_rows=explicit_negative_rows,
        )
    return result


__all__ = ["ObjectTrackExtentAuthority", "compile_object_track_extent_authority"]
