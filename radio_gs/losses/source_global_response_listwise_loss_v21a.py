"""Source-only V2.1A rescue loss with an anchor-trainable triplet denominator.

The frozen V2.1 implementation remains the authority for every response,
absolute-relevance, continuous-pairwise, and typed-relation term.  This
module changes only the training triplet denominator: continuous pairwise
terms retain any-trainable-endpoint pairs, while triplets retain only pairs
whose anchor descriptor can receive a gradient.  Validation retains every
authority pair for both terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from radio_gs.losses import source_global_response_listwise_loss as v2
from radio_gs.losses import source_global_response_listwise_loss_v21 as v21
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    FrozenCanonicalNegativeBank,
    FrozenCompositionalGenericBank,
    SourceGlobalResponseLossV21Config,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
)


def source_global_response_listwise_loss_v21a(
    base_loss: torch.Tensor,
    student_descriptors: torch.Tensor,
    teacher_pair_descriptors: torch.Tensor,
    teacher_pair_region_indices: torch.Tensor,
    fit_text_embeddings: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    authority: v2.FrozenSourceResponseAuthority,
    canonical_negative_bank: FrozenCanonicalNegativeBank,
    *,
    accepted_v2_file_sha256: str,
    teacher_file_sha256: str,
    teacher_pair_descriptors_sha256: str,
    fit_text_bank_file_sha256: str,
    compositional_banks: Sequence[FrozenCompositionalGenericBank] = (),
    relation_authority: FrozenTypedTextRelationAuthority | None = None,
    trainable_region_mask: torch.Tensor | None = None,
    training: bool = False,
    config: SourceGlobalResponseLossV21Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Apply V2.1 while decoupling pairwise and triplet train denominators."""

    chosen = v21.recommended_v21_config() if config is None else config
    total, raw_metrics = v21.source_global_response_listwise_loss_v21(
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
        trainable_region_mask=trainable_region_mask,
        # The frozen V2.1 path continues to define the pairwise denominator.
        exclude_both_immutable_pairs=training,
        config=chosen,
    )
    metrics = dict(raw_metrics)
    channels = authority.payload["channels"]
    anchors = channels["anchor_region_indices"].to(student_descriptors.device)
    negatives = channels["negative_region_indices"].to(student_descriptors.device)
    authority_pairs = int(anchors.numel())
    if training:
        if trainable_region_mask is None:
            raise ValueError("V2.1A training requires a trainable region mask")
        trainable = torch.as_tensor(trainable_region_mask).detach()
        if (
            trainable.dtype != torch.bool
            or trainable.ndim != 1
            or trainable.numel() != student_descriptors.shape[0]
        ):
            raise ValueError("trainable_region_mask must be bool [canonical rows]")
        trainable = trainable.to(student_descriptors.device)
        pairwise_mask = trainable[anchors] | trainable[negatives]
        triplet_mask = trainable[anchors]
    else:
        pairwise_mask = torch.ones_like(anchors, dtype=torch.bool)
        triplet_mask = torch.ones_like(anchors, dtype=torch.bool)
    pairwise_pairs = int(pairwise_mask.sum())
    triplet_pairs = int(triplet_mask.sum())
    if pairwise_pairs <= 0 or triplet_pairs <= 0:
        raise ValueError("V2.1A requires nonempty pairwise and triplet denominators")
    if int(metrics["objective_hard_negative_pairs"]) != pairwise_pairs:
        raise RuntimeError("frozen V2.1 pairwise denominator changed")

    if float(chosen.auxiliary_weight) > 0.0:
        student = F.normalize(student_descriptors.float(), dim=-1)
        teacher_raw = v2._finite_float_matrix(
            teacher_pair_descriptors,
            label="teacher_pair_descriptors",
            device=student.device,
        )
        _views, _mask, teacher_consensus = v2._teacher_views(
            teacher_raw,
            teacher_pair_region_indices,
            region_count=int(student.shape[0]),
        )
        selected_anchors = anchors[triplet_mask]
        selected_negatives = negatives[triplet_mask]
        declared_cosine = channels["teacher_cosines"].to(
            device=student.device, dtype=torch.float32
        )[triplet_mask]
        positive = (
            student[selected_anchors] * teacher_consensus[selected_anchors]
        ).sum(dim=-1)
        negative = (
            student[selected_anchors] * teacher_consensus[selected_negatives]
        ).sum(dim=-1)
        margin = (
            float(chosen.triplet_margin_scale)
            * (1.0 - declared_cosine).clamp_min(0.0)
        ).clamp_max(float(chosen.triplet_margin_ceiling))
        anchor_triplet = F.relu(margin - positive + negative).mean()
        frozen_triplet = torch.as_tensor(
            metrics["hard_negative_triplet_loss"], device=total.device
        )
        total = total + float(chosen.auxiliary_weight) * (
            anchor_triplet - frozen_triplet
        ) / 4.0
        metrics["hard_negative_triplet_loss"] = anchor_triplet
    metrics.update(
        {
            "pairwise_objective_hard_negative_pairs": pairwise_pairs,
            "triplet_objective_hard_negative_pairs": triplet_pairs,
            "triplet_nonanchor_only_pairs_excluded": pairwise_pairs - triplet_pairs,
            "triplet_anchor_trainable_coverage": student_descriptors.new_tensor(
                triplet_pairs / authority_pairs
            ),
        }
    )
    if not bool(torch.isfinite(total.detach())):
        raise RuntimeError("source-global V2.1A response objective is nonfinite")
    return total, metrics


__all__ = ["source_global_response_listwise_loss_v21a"]
