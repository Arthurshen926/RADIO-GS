"""Source-global precision-constrained boundary and ranking loss for DBA-v2."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from radio_gs.losses import factorized_native_source_boundary_alignment as dba_v1


MINIMUM_PRECISION = 0.25
HARD_NEGATIVES_PER_POSITIVE = int((1.0 - MINIMUM_PRECISION) / MINIMUM_PRECISION)
SOFT_FIDELITY_WEIGHT = 0.25
BOUNDARY_RANK_WEIGHT = 0.25
GLOBAL_ORDER_WEIGHT = 0.25
SMOOTH_L1_BETA = 0.05
GLOBAL_ORDER_PAIR_CAP = 4096
MINIMUM_TEACHER_ORDER_GAP = 0.05


@dataclass(frozen=True)
class PrecisionRankingOutput:
    loss: torch.Tensor
    hard_boundary_loss: torch.Tensor
    soft_fidelity_loss: torch.Tensor
    boundary_pairwise_rank_loss: torch.Tensor
    global_order_loss: torch.Tensor
    teacher_positive_pairs: int
    teacher_negative_pairs: int
    selected_hard_negative_pairs: int
    global_order_pairs: int
    student_positive_rate: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


def _balanced_selected_mean(
    positive_value: torch.Tensor, hard_negative_value: torch.Tensor
) -> torch.Tensor:
    if positive_value.numel() <= 0 or hard_negative_value.numel() <= 0:
        raise ValueError("DBA-v2 requires positive and selected-negative units")
    return 0.5 * positive_value.mean() + 0.5 * hard_negative_value.mean()


def _boundary_pairwise_rank(
    positive_margin: torch.Tensor, hard_negative_margin: torch.Tensor
) -> torch.Tensor:
    positive = torch.sort(positive_margin.flatten()).values
    negative = torch.sort(hard_negative_margin.flatten(), descending=True).values
    count = int(negative.numel())
    positive_axis = torch.div(
        torch.arange(count, device=positive.device) * positive.numel(),
        count,
        rounding_mode="floor",
    )
    return F.softplus(
        dba_v1.INFERENCE_LOGIT_SCALE
        * (negative - positive[positive_axis])
    ).mean()


def _global_order_rank(
    student_margin: torch.Tensor, teacher_probability: torch.Tensor
) -> tuple[torch.Tensor, int]:
    student = student_margin.flatten()
    teacher = teacher_probability.flatten()
    count = min(GLOBAL_ORDER_PAIR_CAP, int(student.numel()) // 2)
    if count <= 0:
        raise ValueError("DBA-v2 global rank axis is empty")
    order = torch.argsort(teacher, stable=True)
    half = int(order.numel()) // 2
    low_position = torch.div(
        torch.arange(count, device=student.device) * half,
        count,
        rounding_mode="floor",
    )
    high_position = order.numel() - 1 - low_position
    low = order[low_position]
    high = order[high_position]
    keep = teacher[high] - teacher[low] >= MINIMUM_TEACHER_ORDER_GAP
    if not bool(keep.any()):
        raise ValueError("DBA-v2 source batch lacks teacher order support")
    difference = student[high[keep]] - student[low[keep]]
    return (
        F.softplus(-dba_v1.INFERENCE_LOGIT_SCALE * difference).mean(),
        int(keep.sum()),
    )


def source_precision_constrained_ranking_loss(
    student_descriptor: torch.Tensor,
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    *,
    logit_scale: float = dba_v1.INFERENCE_LOGIT_SCALE,
    soft_fidelity_weight: float = SOFT_FIDELITY_WEIGHT,
    boundary_rank_weight: float = BOUNDARY_RANK_WEIGHT,
    global_order_weight: float = GLOBAL_ORDER_WEIGHT,
    smooth_l1_beta: float = SMOOTH_L1_BETA,
    query_chunk_rows: int = 128,
) -> PrecisionRankingOutput:
    """Align the zero boundary while concentrating gradient on false positives."""

    scalars = (
        logit_scale,
        soft_fidelity_weight,
        boundary_rank_weight,
        global_order_weight,
        smooth_l1_beta,
    )
    if (
        any(not math.isfinite(float(value)) for value in scalars)
        or float(logit_scale) <= 0
        or min(
            float(soft_fidelity_weight),
            float(boundary_rank_weight),
            float(global_order_weight),
        )
        < 0
        or float(smooth_l1_beta) <= 0
    ):
        raise ValueError("DBA-v2 scalar configuration differs")
    margin = dba_v1.exact_student_margin(
        student_descriptor, positive_text, canonical_negative_text
    )
    teacher_probability = dba_v1.exact_multiview_teacher_probability(
        teacher_views,
        teacher_mask,
        positive_text,
        canonical_negative_text,
        logit_scale=float(logit_scale),
        query_chunk_rows=query_chunk_rows,
    ).to(margin.device)
    if margin.shape != teacher_probability.shape:
        raise ValueError("DBA-v2 student and teacher axes differ")
    teacher_positive = teacher_probability >= 0.5
    positive_count = int(teacher_positive.sum())
    negative_count = int((~teacher_positive).sum())
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("DBA-v2 batch requires both teacher boundary classes")
    positive_margin = margin[teacher_positive]
    negative_margin = margin[~teacher_positive]
    hard_count = min(
        negative_count, HARD_NEGATIVES_PER_POSITIVE * positive_count
    )
    hard_negative_margin = torch.topk(
        negative_margin, hard_count, largest=True, sorted=True
    ).values

    positive_hard = F.softplus(-float(logit_scale) * positive_margin)
    negative_hard = F.softplus(float(logit_scale) * hard_negative_margin)
    hard_boundary = _balanced_selected_mean(positive_hard, negative_hard)

    student_probability = torch.sigmoid(float(logit_scale) * margin)
    soft_units = F.smooth_l1_loss(
        student_probability,
        teacher_probability,
        beta=float(smooth_l1_beta),
        reduction="none",
    )
    positive_soft = soft_units[teacher_positive]
    # The top-k indices are recomputed on the negative margin axis so the soft
    # fidelity term supervises precisely the same false-positive tail.
    hard_axis = torch.topk(
        negative_margin, hard_count, largest=True, sorted=True
    ).indices
    negative_soft = soft_units[~teacher_positive][hard_axis]
    soft_fidelity = _balanced_selected_mean(positive_soft, negative_soft)

    boundary_rank = _boundary_pairwise_rank(
        positive_margin, hard_negative_margin
    )
    global_order, global_pairs = _global_order_rank(
        margin, teacher_probability
    )
    loss = (
        hard_boundary
        + float(soft_fidelity_weight) * soft_fidelity
        + float(boundary_rank_weight) * boundary_rank
        + float(global_order_weight) * global_order
    )

    with torch.no_grad():
        student_positive = margin >= 0
        true_positive = (student_positive & teacher_positive).sum().float()
        predicted = student_positive.sum().float()
        teacher_count = teacher_positive.sum().float()
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / teacher_count.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("DBA-v2 precision-ranking loss is nonfinite")
    return PrecisionRankingOutput(
        loss=loss,
        hard_boundary_loss=hard_boundary,
        soft_fidelity_loss=soft_fidelity,
        boundary_pairwise_rank_loss=boundary_rank,
        global_order_loss=global_order,
        teacher_positive_pairs=positive_count,
        teacher_negative_pairs=negative_count,
        selected_hard_negative_pairs=hard_count,
        global_order_pairs=global_pairs,
        student_positive_rate=student_positive.float().mean(),
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = [
    "BOUNDARY_RANK_WEIGHT",
    "GLOBAL_ORDER_PAIR_CAP",
    "GLOBAL_ORDER_WEIGHT",
    "HARD_NEGATIVES_PER_POSITIVE",
    "MINIMUM_PRECISION",
    "MINIMUM_TEACHER_ORDER_GAP",
    "PrecisionRankingOutput",
    "SMOOTH_L1_BETA",
    "SOFT_FIDELITY_WEIGHT",
    "source_precision_constrained_ranking_loss",
]
