"""Soft cross-view object association with a per-Gaussian visibility denominator."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisibilityTrackPosterior:
    probability: torch.Tensor
    association_probability: torch.Tensor
    null_probability: torch.Tensor
    positive_evidence: torch.Tensor
    visibility_denominator: torch.Tensor
    seed_proposal: torch.Tensor
    fallback: torch.Tensor


def object_aware_visibility_track_posterior(
    v1_probability: torch.Tensor,
    seed_probability: torch.Tensor,
    seed_valid: torch.Tensor,
    object_language_score: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_area_fraction: torch.Tensor,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_affinity: torch.Tensor,
    edge_relation: torch.Tensor,
    view_visibility_mass: torch.Tensor,
    view_observed: torch.Tensor,
    *,
    relation_logit_scale: float = 8.0,
    language_logit_scale: float = 8.0,
) -> VisibilityTrackPosterior:
    """Marginalize source-view proposal association and visible membership.

    Every view has an explicit null state.  Proposal multiplicity is corrected
    by an exchangeable ``-log(K)`` prior.  Missing membership is negative only
    when the Gaussian is observed in that view; otherwise it is unknown and
    contributes neither numerator nor denominator.
    """

    original = torch.as_tensor(v1_probability)
    base = original.float().cpu()
    seed_p = torch.as_tensor(seed_probability).float().cpu()
    valid = torch.as_tensor(seed_valid).bool().cpu()
    language = torch.as_tensor(object_language_score).float().cpu()
    if base.ndim != 2 or seed_p.shape != valid.shape or seed_p.shape != language.shape:
        raise ValueError("visibility-track probability axes differ")
    num_rows, num_queries = base.shape
    num_proposals = int(seed_p.shape[0])
    rows = torch.as_tensor(row_indices).long().cpu()
    props = torch.as_tensor(proposal_indices).long().cpu()
    weights = torch.as_tensor(membership_weights).float().cpu()
    views = torch.as_tensor(proposal_view_indices).long().cpu()
    areas = torch.as_tensor(proposal_area_fraction).float().cpu()
    left = torch.as_tensor(edge_left).long().cpu()
    right = torch.as_tensor(edge_right).long().cpu()
    affinity = torch.as_tensor(edge_affinity).float().cpu()
    relation = torch.as_tensor(edge_relation).to(torch.int8).cpu()
    visibility_mass = torch.as_tensor(view_visibility_mass).float().cpu()
    observed = torch.as_tensor(view_observed).bool().cpu()
    if rows.shape != props.shape or rows.shape != weights.shape:
        raise ValueError("visibility-track sparse axes differ")
    if views.shape != areas.shape or views.shape != (num_proposals,):
        raise ValueError("visibility-track proposal axes differ")
    if not (left.shape == right.shape == affinity.shape == relation.shape):
        raise ValueError("visibility-track edge axes differ")
    num_views = int(observed.shape[0])
    if visibility_mass.shape != observed.shape or observed.shape != (num_views, num_rows) or bool((views < 0).any()) or bool((views >= num_views).any()):
        raise ValueError("visibility authority axes differ")
    if bool((~torch.isfinite(visibility_mass) | (visibility_mass < 0)).any()):
        raise ValueError("visibility mass must be finite and non-negative")
    if bool((visibility_mass[~observed] != 0).any()):
        raise ValueError("unobserved rows carry visibility mass")

    if bool((~torch.isfinite(weights) | (weights < 0) | (weights > 1)).any()):
        raise ValueError("exact membership probabilities must lie in [0,1]")
    # Membership is already exact numerator/view_denominator.  Proposal-max
    # normalization would destroy its Bernoulli probability meaning.
    conditional = weights
    edge_logits = torch.full((num_proposals, num_proposals), float("nan"))
    model_logits = float(relation_logit_scale) * affinity
    # Explicit known outcomes own the sign; unknown uses the proper learned logit.
    logits = torch.where(
        relation == 1,
        torch.full_like(model_logits, float(relation_logit_scale)),
        torch.where(
            relation == 0,
            torch.full_like(model_logits, -float(relation_logit_scale)),
            model_logits,
        ),
    )
    edge_logits[left, right] = logits
    edge_logits[right, left] = logits

    output = base.clone()
    association = torch.zeros((num_proposals, num_queries))
    null = torch.ones((num_views, num_queries))
    numerator_all = torch.zeros_like(base)
    denominator_all = torch.zeros_like(base)
    seeds = torch.full((num_queries,), -1, dtype=torch.long)
    fallback = torch.ones(num_queries, dtype=torch.bool)
    proposals_by_view = [torch.where(views == view)[0] for view in range(num_views)]
    for query in range(num_queries):
        candidates = torch.where(valid[:, query])[0]
        if candidates.numel() == 0:
            continue
        seed = int(candidates[torch.argmax(seed_p[candidates, query])])
        seeds[query] = seed
        association[seed, query] = 1.0
        null[int(views[seed]), query] = 0.0
        seed_area = areas[seed].clamp_min(1e-8)
        for view, view_proposals in enumerate(proposals_by_view):
            if view == int(views[seed]) or view_proposals.numel() == 0:
                continue
            candidate_logits = edge_logits[seed, view_proposals]
            known = torch.isfinite(candidate_logits)
            if not bool(known.any()):
                continue
            view_proposals = view_proposals[known]
            candidate_logits = candidate_logits[known]
            # Continuous scale evidence, not an admission threshold.
            scale_log_prior = -torch.abs(
                torch.log2(areas[view_proposals].clamp_min(1e-8) / seed_area)
            )
            candidate_logits = (
                candidate_logits + scale_log_prior - torch.log(torch.tensor(float(view_proposals.numel())))
            )
            normalizer = torch.logsumexp(
                torch.cat((candidate_logits, candidate_logits.new_zeros(1))), dim=0
            )
            probabilities = torch.exp(candidate_logits - normalizer)
            association[view_proposals, query] = probabilities
            null[view, query] = torch.exp(-normalizer)

        nonnull_by_view = torch.zeros(num_views)
        nonnull_by_view.index_add_(0, views, association[:, query])
        denominator = torch.matmul(nonnull_by_view, visibility_mass)
        numerator = torch.zeros(num_rows)
        numerator.index_add_(
            0,
            rows,
            conditional
            * association[props, query]
            * visibility_mass[views[props], rows],
        )
        positive = (numerator / denominator.clamp_min(1e-8)).clamp(0, 1)
        supported = denominator > 0
        if not bool(supported.any()):
            continue
        track_mass = association[:, query].sum().clamp_min(1e-8)
        track_language = (
            association[:, query] * language[:, query]
        ).sum() / track_mass
        language_probability = torch.sigmoid(
            float(language_logit_scale) * (track_language - 0.5)
        )
        confidence = 1.0 - torch.exp(-denominator)
        affirmative = (positive - base[:, query]).clamp_min(0)
        output[:, query] = base[:, query] + language_probability * confidence * affirmative
        numerator_all[:, query] = numerator
        denominator_all[:, query] = denominator
        fallback[query] = False
    return VisibilityTrackPosterior(
        probability=output.to(original.dtype),
        association_probability=association,
        null_probability=null,
        positive_evidence=numerator_all,
        visibility_denominator=denominator_all,
        seed_proposal=seeds,
        fallback=fallback,
    )


__all__ = ["VisibilityTrackPosterior", "object_aware_visibility_track_posterior"]
