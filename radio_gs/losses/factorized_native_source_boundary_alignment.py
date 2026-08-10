"""Source-only text-boundary alignment for factorized-native descriptors.

This module deliberately supervises the decision quantity used by the frozen
LERF query interface rather than another visual-to-visual proxy.  The positive
axis is a target-blind generic text bank and the negative axis is the same
four-row canonical bank used at inference.  No scene or query-specific
parameter is introduced.

The teacher target is intentionally *not* the older response-loss surrogate
``sigmoid(logmeanexp(pos) - max(logmeanexp(neg)))``.  It is the arithmetic
mean, over valid official teacher views, of the exact per-view probability
``sigmoid(scale * (pos - max(neg)))``.  Keeping ``max`` and ``sigmoid`` inside
the view reduction matches the frozen absolute-relevance decision semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


INFERENCE_LOGIT_SCALE = 10.0
CANONICAL_NEGATIVE_ROWS = 4
SOFT_FIDELITY_WEIGHT = 0.25
SMOOTH_L1_BETA = 0.05


@dataclass(frozen=True)
class BoundaryAlignmentOutput:
    """Loss components and detached source-only boundary diagnostics."""

    loss: torch.Tensor
    balanced_hard_boundary_loss: torch.Tensor
    balanced_soft_fidelity_loss: torch.Tensor
    teacher_positive_pairs: int
    teacher_negative_pairs: int
    student_positive_rate: torch.Tensor
    teacher_positive_rate: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


def _unit_matrix(value: torch.Tensor, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if (
        tensor.ndim != 2
        or not tensor.is_floating_point()
        or min(tensor.shape) <= 0
        or not bool(torch.isfinite(tensor).all())
        or bool((torch.linalg.vector_norm(tensor.float(), dim=-1) <= 1e-12).any())
    ):
        raise ValueError(f"{label} must be finite nonzero floating [N,D]")
    return F.normalize(tensor.float(), dim=-1)


def _validated_banks(
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    *,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = _unit_matrix(positive_text, label="positive text").to(device)
    negative = _unit_matrix(
        canonical_negative_text, label="canonical negative text"
    ).to(device)
    if positive.shape[1] != width or negative.shape != (
        CANONICAL_NEGATIVE_ROWS,
        width,
    ):
        raise ValueError("text banks and descriptor dimensions differ")
    return positive, negative


def exact_student_margin(
    student_descriptor: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
) -> torch.Tensor:
    """Return exact inference margins ``[region, generic query]``."""

    student = _unit_matrix(student_descriptor, label="student descriptor")
    positive, negative = _validated_banks(
        positive_text,
        canonical_negative_text,
        width=int(student.shape[1]),
        device=student.device,
    )
    positive_score = student @ positive.T
    negative_score = (student @ negative.T).amax(dim=-1, keepdim=True)
    return positive_score - negative_score


def exact_multiview_teacher_probability(
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    *,
    logit_scale: float = INFERENCE_LOGIT_SCALE,
    query_chunk_rows: int = 128,
) -> torch.Tensor:
    """Average exact per-view relevance probabilities over valid views.

    The returned tensor is detached because official teacher observations and
    both text banks are immutable supervision, never optimization variables.
    """

    views_raw = torch.as_tensor(teacher_views)
    mask = torch.as_tensor(teacher_mask, device=views_raw.device)
    if (
        views_raw.ndim != 3
        or not views_raw.is_floating_point()
        or mask.dtype != torch.bool
        or mask.shape != views_raw.shape[:2]
        or not bool(mask.any(dim=1).all())
        or not bool(torch.isfinite(views_raw[mask]).all())
        or type(query_chunk_rows) is not int
        or query_chunk_rows <= 0
        or not math.isfinite(float(logit_scale))
        or float(logit_scale) <= 0
    ):
        raise ValueError("official multiview teacher layout differs")
    views = F.normalize(views_raw.detach().float(), dim=-1)
    positive, negative = _validated_banks(
        positive_text,
        canonical_negative_text,
        width=int(views.shape[-1]),
        device=views.device,
    )
    negative_score = torch.einsum("bvd,kd->bvk", views, negative).amax(dim=-1)
    denominator = mask.sum(dim=1, keepdim=True)
    chunks: list[torch.Tensor] = []
    for start in range(0, positive.shape[0], query_chunk_rows):
        query = positive[start : start + query_chunk_rows]
        positive_score = torch.einsum("bvd,qd->bvq", views, query)
        probability = torch.sigmoid(
            float(logit_scale) * (positive_score - negative_score[..., None])
        )
        chunks.append(
            ((probability * mask[..., None]).sum(dim=1) / denominator).detach()
        )
    result = torch.cat(chunks, dim=1)
    if (
        result.shape != (views.shape[0], positive.shape[0])
        or not bool(torch.isfinite(result).all())
        or bool((result < 0).any())
        or bool((result > 1).any())
    ):
        raise RuntimeError("exact multiview teacher probability is invalid")
    return result


def _class_balanced_mean(
    value: torch.Tensor, positive: torch.Tensor
) -> torch.Tensor:
    if value.shape != positive.shape or positive.dtype != torch.bool:
        raise ValueError("class-balanced values and labels differ")
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("boundary alignment batch requires both teacher classes")
    return 0.5 * value[positive].mean() + 0.5 * value[~positive].mean()


def source_balanced_boundary_alignment_loss(
    student_descriptor: torch.Tensor,
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    positive_text: torch.Tensor,
    canonical_negative_text: torch.Tensor,
    *,
    logit_scale: float = INFERENCE_LOGIT_SCALE,
    soft_fidelity_weight: float = SOFT_FIDELITY_WEIGHT,
    smooth_l1_beta: float = SMOOTH_L1_BETA,
    query_chunk_rows: int = 128,
) -> BoundaryAlignmentOutput:
    """Fit the exact 0.5 boundary with equal positive/negative class mass.

    Hard boundary BCE is the primary term, so unlike positive-slope
    temperature scaling it can rotate the descriptor and change which
    region/query pairs lie above 0.5.  A smaller balanced soft-fidelity term
    prevents the hard classification target from discarding teacher
    confidence.  Class balancing is global over the supplied source batch;
    it creates neither scene-specific nor query-specific parameters.
    """

    if (
        not math.isfinite(float(logit_scale))
        or float(logit_scale) <= 0
        or not math.isfinite(float(soft_fidelity_weight))
        or float(soft_fidelity_weight) < 0
        or not math.isfinite(float(smooth_l1_beta))
        or float(smooth_l1_beta) <= 0
    ):
        raise ValueError("boundary alignment scalar configuration differs")
    margin = exact_student_margin(
        student_descriptor, positive_text, canonical_negative_text
    )
    teacher_probability = exact_multiview_teacher_probability(
        teacher_views,
        teacher_mask,
        positive_text,
        canonical_negative_text,
        logit_scale=float(logit_scale),
        query_chunk_rows=query_chunk_rows,
    ).to(margin.device)
    if teacher_probability.shape != margin.shape:
        raise ValueError("student and teacher boundary axes differ")

    teacher_positive = teacher_probability >= 0.5
    logits = float(logit_scale) * margin
    hard_units = torch.where(
        teacher_positive, F.softplus(-logits), F.softplus(logits)
    )
    hard_loss = _class_balanced_mean(hard_units, teacher_positive)
    student_probability = torch.sigmoid(logits)
    soft_units = F.smooth_l1_loss(
        student_probability,
        teacher_probability,
        beta=float(smooth_l1_beta),
        reduction="none",
    )
    soft_loss = _class_balanced_mean(soft_units, teacher_positive)
    loss = hard_loss + float(soft_fidelity_weight) * soft_loss

    with torch.no_grad():
        student_positive = logits >= 0
        true_positive = (student_positive & teacher_positive).sum().float()
        predicted_count = student_positive.sum().float()
        teacher_count = teacher_positive.sum().float()
        precision = true_positive / predicted_count.clamp_min(1.0)
        recall = true_positive / teacher_count.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("source boundary alignment loss is nonfinite")
    return BoundaryAlignmentOutput(
        loss=loss,
        balanced_hard_boundary_loss=hard_loss,
        balanced_soft_fidelity_loss=soft_loss,
        teacher_positive_pairs=int(teacher_positive.sum()),
        teacher_negative_pairs=int((~teacher_positive).sum()),
        student_positive_rate=student_positive.float().mean(),
        teacher_positive_rate=teacher_positive.float().mean(),
        precision=precision,
        recall=recall,
        f1=f1,
    )


__all__ = [
    "BoundaryAlignmentOutput",
    "CANONICAL_NEGATIVE_ROWS",
    "INFERENCE_LOGIT_SCALE",
    "SMOOTH_L1_BETA",
    "SOFT_FIDELITY_WEIGHT",
    "exact_multiview_teacher_probability",
    "exact_student_margin",
    "source_balanced_boundary_alignment_loss",
]
