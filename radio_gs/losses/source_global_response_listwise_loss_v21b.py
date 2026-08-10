"""V2.1B hard-negative denominator correction over the frozen V2.1 loss.

The query-bank, absolute-relevance, response-profile, pairwise, and typed
relation implementations remain the frozen V2.1 implementation.  V2.1B
changes one gradient-denominator bug: pairwise terms retain a pair when either
endpoint is trainable, while triplets retain it only when the anchor endpoint
is trainable.  This wrapper recomputes the triplet term and analytically
replaces the frozen term in the total and auxiliary loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from radio_gs.losses import source_global_response_listwise_loss as v2
from radio_gs.losses import source_global_response_listwise_loss_v21 as v21
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
)


def hard_negative_denominator_masks(
    trainable_region_mask: torch.Tensor,
    anchor_region_indices: torch.Tensor,
    negative_region_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pairwise_any_endpoint, triplet_anchor_only)`` masks."""

    trainable = torch.as_tensor(trainable_region_mask)
    anchors = torch.as_tensor(anchor_region_indices, device=trainable.device)
    negatives = torch.as_tensor(negative_region_indices, device=trainable.device)
    if (
        trainable.dtype != torch.bool
        or trainable.ndim != 1
        or anchors.dtype != torch.int64
        or negatives.dtype != torch.int64
        or anchors.ndim != 1
        or negatives.shape != anchors.shape
        or bool((anchors < 0).any())
        or bool((negatives < 0).any())
        or bool((anchors >= trainable.numel()).any())
        or bool((negatives >= trainable.numel()).any())
    ):
        raise ValueError("V2.1B hard-negative denominator inputs differ")
    return trainable[anchors] | trainable[negatives], trainable[anchors]


def _triplet_loss(
    student_descriptors: torch.Tensor,
    teacher_pair_descriptors: torch.Tensor,
    teacher_pair_region_indices: torch.Tensor,
    anchors: torch.Tensor,
    negatives: torch.Tensor,
    teacher_cosines: torch.Tensor,
    retained: torch.Tensor,
    *,
    config: v21.SourceGlobalResponseLossV21Config,
) -> torch.Tensor:
    if not bool(retained.any()):
        raise ValueError("V2.1B triplet denominator has no trainable anchor")
    student = F.normalize(student_descriptors.float(), dim=-1)
    _, _, teacher_consensus = v2._teacher_views(
        teacher_pair_descriptors,
        teacher_pair_region_indices,
        region_count=int(student.shape[0]),
    )
    selected_anchors = anchors[retained]
    selected_negatives = negatives[retained]
    positive = (
        student[selected_anchors] * teacher_consensus[selected_anchors]
    ).sum(dim=-1)
    negative = (
        student[selected_anchors] * teacher_consensus[selected_negatives]
    ).sum(dim=-1)
    margin = (
        float(config.triplet_margin_scale)
        * (1.0 - teacher_cosines[retained]).clamp_min(0.0)
    ).clamp_max(float(config.triplet_margin_ceiling))
    return F.relu(margin - positive + negative).mean()


def source_global_response_listwise_loss_v21b(
    base_loss: torch.Tensor,
    student_descriptors: torch.Tensor,
    teacher_pair_descriptors: torch.Tensor,
    teacher_pair_region_indices: torch.Tensor,
    fit_text_embeddings: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    authority: v2.FrozenSourceResponseAuthority,
    canonical_negative_bank: v21.FrozenCanonicalNegativeBank,
    *,
    accepted_v2_file_sha256: str,
    teacher_file_sha256: str,
    teacher_pair_descriptors_sha256: str,
    fit_text_bank_file_sha256: str,
    compositional_banks: Sequence[v21.FrozenCompositionalGenericBank] = (),
    relation_authority: FrozenTypedTextRelationAuthority | None = None,
    trainable_region_mask: torch.Tensor,
    config: v21.SourceGlobalResponseLossV21Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Apply frozen V2.1 with separate pairwise and triplet denominators."""

    chosen = v21.recommended_v21_config() if config is None else config
    if not isinstance(chosen, v21.SourceGlobalResponseLossV21Config):
        raise TypeError("config must be SourceGlobalResponseLossV21Config")
    if chosen.auxiliary_weight <= 0:
        raise ValueError("V2.1B requires the fixed positive auxiliary weight")
    trainable = torch.as_tensor(trainable_region_mask).detach()
    if (
        trainable.dtype != torch.bool
        or trainable.ndim != 1
        or trainable.numel() != student_descriptors.shape[0]
    ):
        raise ValueError("trainable_region_mask must be bool [canonical rows]")
    device = student_descriptors.device
    channels = authority.payload["channels"]
    anchors = channels["anchor_region_indices"].to(device)
    negatives = channels["negative_region_indices"].to(device)
    teacher_cosines = channels["teacher_cosines"].to(
        device=device,
        dtype=torch.float32,
    )
    pairwise_mask, triplet_mask = hard_negative_denominator_masks(
        trainable.to(device),
        anchors,
        negatives,
    )
    pairwise_count = int(pairwise_mask.sum())
    triplet_count = int(triplet_mask.sum())
    if pairwise_count <= 0:
        raise ValueError("V2.1B pairwise denominator has no trainable endpoint")
    if triplet_count <= 0:
        raise ValueError("V2.1B triplet denominator has no trainable anchor")

    frozen_total, frozen_metrics = v21.source_global_response_listwise_loss_v21(
        base_loss,
        student_descriptors,
        teacher_pair_descriptors,
        teacher_pair_region_indices,
        fit_text_embeddings,
        canonical_region_indices,
        authority,
        canonical_negative_bank,
        accepted_v2_file_sha256=accepted_v2_file_sha256,
        teacher_file_sha256=teacher_file_sha256,
        teacher_pair_descriptors_sha256=teacher_pair_descriptors_sha256,
        fit_text_bank_file_sha256=fit_text_bank_file_sha256,
        compositional_banks=compositional_banks,
        relation_authority=relation_authority,
        trainable_region_mask=trainable,
        exclude_both_immutable_pairs=True,
        config=chosen,
    )
    if int(frozen_metrics["objective_hard_negative_pairs"]) != pairwise_count:
        raise RuntimeError("frozen V2.1 pairwise denominator replay differs")
    frozen_triplet = frozen_metrics["hard_negative_triplet_loss"]
    corrected_triplet = _triplet_loss(
        student_descriptors,
        teacher_pair_descriptors,
        teacher_pair_region_indices,
        anchors,
        negatives,
        teacher_cosines,
        triplet_mask,
        config=chosen,
    )
    triplet_delta = (corrected_triplet - frozen_triplet) / 4.0
    corrected_auxiliary = frozen_metrics["auxiliary_loss"] + triplet_delta
    corrected_total = frozen_total + float(chosen.auxiliary_weight) * triplet_delta
    if not bool(torch.isfinite(corrected_total.detach())):
        raise RuntimeError("source-global V2.1B response objective is nonfinite")
    authority_count = int(anchors.numel())
    metrics = {
        **frozen_metrics,
        "hard_negative_triplet_loss": corrected_triplet,
        "auxiliary_loss": corrected_auxiliary,
        "objective_hard_negative_pairs": pairwise_count,
        "pairwise_objective_hard_negative_pairs": pairwise_count,
        "triplet_objective_hard_negative_pairs": triplet_count,
        "triplet_anchor_trainable_coverage": student_descriptors.new_tensor(
            triplet_count / authority_count
        ),
        "both_immutable_pairs_excluded": authority_count - pairwise_count,
    }
    return corrected_total, metrics


__all__ = [
    "hard_negative_denominator_masks",
    "source_global_response_listwise_loss_v21b",
]
