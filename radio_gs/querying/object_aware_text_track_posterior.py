"""Identity-seeded, query-independent object-track posterior."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectTrackPosterior:
    probability: torch.Tensor
    selected_membership: torch.Tensor
    seed_proposal: torch.Tensor
    track_probability: torch.Tensor
    fallback: torch.Tensor


def object_aware_text_track_posterior(
    v1_probability: torch.Tensor,
    seed_probability: torch.Tensor,
    seed_valid: torch.Tensor,
    object_language_score: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_area_fraction: torch.Tensor | None,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_affinity: torch.Tensor,
    edge_relation: torch.Tensor | None,
    *,
    same_threshold: float,
    language_logit_scale: float = 8.0,
) -> ObjectTrackPosterior:
    """Select identity seeds, then expand solely through frozen affinity.

    Proposal membership after seed admission is independent of every query
    score.  Missing seeds, isolated seeds, and single-view components replay
    the input v1 posterior bit-for-bit.
    """

    base = torch.as_tensor(v1_probability).float().cpu()
    seed_p = torch.as_tensor(seed_probability).float().cpu()
    valid = torch.as_tensor(seed_valid).bool().cpu()
    language = torch.as_tensor(object_language_score).float().cpu()
    if base.ndim != 2 or seed_p.shape != valid.shape or seed_p.shape != language.shape:
        raise ValueError("object-track probability axes differ")
    num_rows, num_queries = base.shape
    num_proposals = int(seed_p.shape[0])
    rows = torch.as_tensor(row_indices).long().cpu()
    props = torch.as_tensor(proposal_indices).long().cpu()
    weights = torch.as_tensor(membership_weights).float().cpu()
    views = torch.as_tensor(proposal_view_indices).long().cpu()
    areas = (
        torch.as_tensor(proposal_area_fraction).float().cpu()
        if proposal_area_fraction is not None
        else None
    )
    left = torch.as_tensor(edge_left).long().cpu()
    right = torch.as_tensor(edge_right).long().cpu()
    affinity = torch.as_tensor(edge_affinity).float().cpu()
    relation = (
        torch.as_tensor(edge_relation).to(torch.int8).cpu()
        if edge_relation is not None
        else None
    )
    if rows.shape != props.shape or rows.shape != weights.shape:
        raise ValueError("sparse membership axes differ")
    if views.shape != (num_proposals,) or left.shape != right.shape or left.shape != affinity.shape:
        raise ValueError("object-track graph axes differ")
    if relation is not None and relation.shape != affinity.shape:
        raise ValueError("object-track relation axis differs")
    if areas is not None and areas.shape != (num_proposals,):
        raise ValueError("proposal area axis differs")
    if not float(language_logit_scale) > 0:
        raise ValueError("language logit scale must be positive")

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_proposals)]
    reliable = affinity >= float(same_threshold)
    if relation is not None:
        reliable &= relation != 0
    for a, b, value in zip(
        left[reliable].tolist(), right[reliable].tolist(), affinity[reliable].tolist()
    ):
        if a != b:
            adjacency[a].append((b, value))
            adjacency[b].append((a, value))

    maximum = torch.zeros(num_proposals)
    maximum.scatter_reduce_(0, props, weights, reduce="amax", include_self=True)
    conditional = (weights / maximum[props].clamp_min(1e-8)).clamp(0, 1)
    output = base.clone()
    selected = torch.zeros((num_proposals, num_queries), dtype=torch.bool)
    seeds = torch.full((num_queries,), -1, dtype=torch.long)
    track_probability = torch.zeros(num_queries)
    fallback = torch.ones(num_queries, dtype=torch.bool)
    for query in range(num_queries):
        candidates = torch.where(valid[:, query])[0]
        if candidates.numel() == 0:
            continue
        seed = int(candidates[torch.argmax(seed_p[candidates, query])])
        seeds[query] = seed
        if not adjacency[seed]:
            continue
        # One direct seed neighbour per source view prevents a weakly connected
        # multiscale graph from percolating into a scene-wide component.  There
        # is deliberately no neighbour-of-neighbour transitive closure.
        component: set[int] = {seed}
        best_by_view: dict[int, tuple[int, float]] = {}
        for neighbor, value in adjacency[seed]:
            view = int(views[neighbor])
            if view == int(views[seed]):
                continue
            if areas is not None:
                scale_ratio = float(
                    areas[neighbor].clamp_min(1e-8) / areas[seed].clamp_min(1e-8)
                )
                if scale_ratio < 0.5 or scale_ratio > 2.0:
                    continue
            previous = best_by_view.get(view)
            if previous is None or (value, -neighbor) > (previous[1], -previous[0]):
                best_by_view[view] = (neighbor, value)
        component.update(proposal for proposal, _ in best_by_view.values())
        component_rows = torch.tensor(sorted(component), dtype=torch.long)
        if component_rows.numel() < 2 or torch.unique(views[component_rows]).numel() < 2:
            continue
        selected[component_rows, query] = True
        track_language = language[component_rows, query].mean()
        probability = torch.sigmoid(
            float(language_logit_scale) * (track_language - 0.5)
        )
        pair_keep = selected[props, query]
        selected_rows = rows[pair_keep]
        selected_values = conditional[pair_keep]
        selected_views = views[props[pair_keep]]
        log_survival = torch.zeros(num_rows)
        log_survival.index_add_(
            0,
            selected_rows,
            torch.log1p(-selected_values.clamp_max(1.0 - 1e-6)),
        )
        extent = 1.0 - torch.exp(log_survival)
        # At most one selected proposal exists per view, so this sparse count
        # is an exact independent-view positive-support count.  One-view
        # evidence and absence are epistemic unknown, never negative.
        view_count = torch.zeros(num_rows, dtype=torch.int16)
        view_count.index_add_(
            0, selected_rows, torch.ones_like(selected_views, dtype=torch.int16)
        )
        extent[view_count < 2] = 0.0
        if not bool((extent > 0).any()):
            selected[:, query] = False
            continue
        affirmative = torch.maximum(base[:, query], extent)
        output[:, query] = (
            (1.0 - probability) * base[:, query] + probability * affirmative
        )
        track_probability[query] = probability
        fallback[query] = False
    return ObjectTrackPosterior(
        probability=output.to(torch.as_tensor(v1_probability).dtype),
        selected_membership=selected,
        seed_proposal=seeds,
        track_probability=track_probability,
        fallback=fallback,
    )


__all__ = ["ObjectTrackPosterior", "object_aware_text_track_posterior"]
