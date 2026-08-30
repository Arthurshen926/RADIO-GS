"""Null-capable row-wise soft mask-to-token matching without one-to-one constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .observed_evidence import ObservedObjectEvidence


@dataclass(frozen=True)
class PartialSoftMatch:
    token_probability: torch.Tensor
    null_probability: torch.Tensor
    visible_overlap: torch.Tensor
    mask_containment: torch.Tensor
    visible_token_recall: torch.Tensor
    conflict: torch.Tensor
    granularity: torch.Tensor


def partial_soft_match(
    evidence: ObservedObjectEvidence,
    token_membership: torch.Tensor,
    *,
    element_visibility: torch.Tensor,
    mask_centres: torch.Tensor,
    token_centres: torch.Tensor,
    token_scales: torch.Tensor,
    appearance_score: torch.Tensor | None = None,
    cross_view_score: torch.Tensor | None = None,
    overlap_weight: float = 1.0,
    appearance_weight: float = 0.0,
    geometry_weight: float = 0.25,
    cross_view_weight: float = 0.0,
    conflict_weight: float = 1.0,
    null_logit: float = 0.5,
    temperature: float = 0.1,
    whole_recall: float = 0.5,
    part_containment: float = 0.7,
) -> PartialSoftMatch:
    membership = torch.as_tensor(token_membership, dtype=torch.float32)
    visibility = torch.as_tensor(element_visibility, dtype=torch.float32)
    if membership.ndim != 2 or membership.shape[0] != evidence.positive.shape[1]:
        raise ValueError("token membership must have shape [E, K]")
    if visibility.shape != evidence.positive.shape:
        raise ValueError("element visibility must have shape [M, E]")
    if temperature <= 0:
        raise ValueError("matching temperature must be positive")
    masks, tokens = evidence.positive.shape[0], membership.shape[1]
    mask_centres = torch.as_tensor(mask_centres, dtype=torch.float32)
    token_centres = torch.as_tensor(token_centres, dtype=torch.float32)
    token_scales = torch.as_tensor(token_scales, dtype=torch.float32)
    if mask_centres.shape != (masks, 3) or token_centres.shape != (tokens, 3) or token_scales.shape != (tokens, 3):
        raise ValueError("mask/token geometry has invalid shape")

    positive = evidence.positive
    intersection = positive @ membership
    mask_mass = positive.sum(-1, keepdim=True).clamp_min(1e-12)
    visible_token_mass = visibility @ membership
    union = mask_mass + visible_token_mass - intersection
    iou = intersection / union.clamp_min(1e-12)
    containment = intersection / mask_mass
    recall = intersection / visible_token_mass.clamp_min(1e-12)
    visible_token = visible_token_mass > 1e-8
    conflict = (evidence.negative @ membership) / visible_token_mass.clamp_min(1e-12)

    scale = token_scales.norm(dim=-1).clamp_min(1e-4)
    distance = (mask_centres[:, None] - token_centres[None]).norm(dim=-1) / scale[None]
    geometry = torch.exp(-0.5 * distance.square())
    overlap = torch.maximum(iou, containment)
    logits = overlap_weight * overlap + geometry_weight * geometry - conflict_weight * conflict
    if appearance_score is not None:
        appearance = torch.as_tensor(appearance_score, dtype=torch.float32)
        if appearance.shape != (masks, tokens):
            raise ValueError("appearance score must have shape [M, K]")
        logits = logits + appearance_weight * appearance
    if cross_view_score is not None:
        cross_view = torch.as_tensor(cross_view_score, dtype=torch.float32)
        if cross_view.shape != (masks, tokens):
            raise ValueError("cross-view score must have shape [M, K]")
        logits = logits + cross_view_weight * cross_view
    logits = logits.masked_fill(~visible_token, -torch.inf)
    joined = torch.cat(
        [logits, torch.full((masks, 1), float(null_logit), device=logits.device)], dim=-1
    )
    probability = torch.softmax(joined / temperature, dim=-1)

    best_token = probability[:, :-1].argmax(-1)
    rows = torch.arange(masks, device=probability.device)
    best_recall = recall[rows, best_token]
    best_containment = containment[rows, best_token]
    # 0 whole-object, 1 part/auxiliary, 2 ambiguous/null.
    granularity = torch.full((masks,), 2, dtype=torch.long, device=probability.device)
    matched = probability[:, :-1].max(-1).values > probability[:, -1]
    granularity[matched & (best_recall >= whole_recall)] = 0
    granularity[matched & (best_recall < whole_recall) & (best_containment >= part_containment)] = 1
    return PartialSoftMatch(
        token_probability=probability[:, :-1],
        null_probability=probability[:, -1],
        visible_overlap=iou,
        mask_containment=containment,
        visible_token_recall=recall,
        conflict=conflict,
        granularity=granularity,
    )
