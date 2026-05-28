"""Text-query logit distillation for direct primitive summary heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is None:
        return values.mean()
    active_weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if active_weights.shape[0] != values.shape[0]:
        raise ValueError(f"Expected {values.shape[0]} weights, got {active_weights.shape[0]}")
    return (values * active_weights).sum() / active_weights.sum().clamp_min(1e-6)


def _normalise_support_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode or "none").lower()
    if mode == "none":
        return logits
    centered = logits - logits.mean(dim=-1, keepdim=True)
    if mode == "center":
        return centered
    if mode == "zscore":
        scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / scale
    raise ValueError("support_logit_norm must be one of: none, center, zscore")


def compute_direct_point_query_logit_distill_loss(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match teacher query distributions for arbitrary text banks.

    This complements feature cosine distillation: small vector errors can still
    flip open-vocabulary query rankings, so the primitive head also preserves
    the VPR teacher's logits over the evaluation query bank.
    """
    if student_summary.shape != teacher_summary.shape:
        raise ValueError(
            "student_summary and teacher_summary must have the same shape, got "
            f"{tuple(student_summary.shape)} vs {tuple(teacher_summary.shape)}"
        )
    if student_summary.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Expected [N,C] summaries and [Q,C] text embeddings")
    if student_summary.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "summary/text dimension mismatch: "
            f"{student_summary.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    text = F.normalize(text_embeddings.float().to(student_summary.device), dim=-1)
    student = F.normalize(student_summary.float(), dim=-1)
    teacher = F.normalize(teacher_summary.float(), dim=-1)
    student_logits = student @ text.T
    with torch.no_grad():
        teacher_logits = teacher @ text.T
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        teacher_conf, teacher_label = teacher_prob.max(dim=-1)
        valid = teacher_conf >= float(confidence_threshold)
    zero = student_summary.sum() * 0.0
    valid_ratio = valid.float().mean()
    if not valid.any():
        return zero, {
            "valid_ratio": valid_ratio.detach(),
            "teacher_conf": zero.detach(),
            "agreement": zero.detach(),
        }

    log_prob = F.log_softmax(student_logits[valid] / temperature, dim=-1)
    per_point = F.kl_div(log_prob, teacher_prob[valid], reduction="none").sum(dim=-1)
    loss = _weighted_mean(
        per_point,
        sample_weights[valid] if sample_weights is not None else None,
    )
    if temperature != 1.0:
        loss = loss * (temperature ** 2)
    with torch.no_grad():
        agreement = (student_logits[valid].argmax(dim=-1) == teacher_label[valid]).float().mean()
        mean_conf = teacher_conf[valid].mean()
    return loss, {
        "valid_ratio": valid_ratio.detach(),
        "teacher_conf": mean_conf.detach(),
        "agreement": agreement.detach(),
    }


def compute_direct_point_query_support_distill_loss(
    student_summary: torch.Tensor,
    teacher_summary: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 1.0,
    confidence_threshold: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    support_logit_norm: str = "none",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match teacher primitive-support distributions for each text query.

    ``compute_direct_point_query_logit_distill_loss`` preserves the text ranking
    inside each primitive. Direct 3D object selection, however, thresholds or
    ranks primitives for one query at a time. This loss distills that evaluation
    axis directly by matching the teacher distribution over sampled primitives
    for every text query.
    """
    if student_summary.shape != teacher_summary.shape:
        raise ValueError(
            "student_summary and teacher_summary must have the same shape, got "
            f"{tuple(student_summary.shape)} vs {tuple(teacher_summary.shape)}"
        )
    if student_summary.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Expected [N,C] summaries and [Q,C] text embeddings")
    if student_summary.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "summary/text dimension mismatch: "
            f"{student_summary.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    text = F.normalize(text_embeddings.float().to(student_summary.device), dim=-1)
    student = F.normalize(student_summary.float(), dim=-1)
    teacher = F.normalize(teacher_summary.float(), dim=-1)
    student_logits = student @ text.T
    with torch.no_grad():
        teacher_logits = teacher @ text.T
        teacher_conf, teacher_argmax = teacher_logits.max(dim=0)
        valid_queries = teacher_conf >= float(confidence_threshold)

    zero = student_summary.sum() * 0.0
    valid_ratio = valid_queries.float().mean()
    if not valid_queries.any():
        return zero, {
            "valid_ratio": valid_ratio.detach(),
            "teacher_conf": zero.detach(),
            "top1_agreement": zero.detach(),
        }

    # Transpose to [Q, N], so each row is one query's primitive support.
    teacher_support_logits = _normalise_support_logits(
        teacher_logits.T[valid_queries],
        support_logit_norm,
    ) / float(temperature)
    student_support_logits = _normalise_support_logits(
        student_logits.T[valid_queries],
        support_logit_norm,
    ) / float(temperature)
    if sample_weights is not None:
        weights = sample_weights.to(device=student_summary.device, dtype=student_support_logits.dtype)
        if weights.ndim != 1 or weights.shape[0] != student_summary.shape[0]:
            raise ValueError(
                f"Expected sample_weights [{student_summary.shape[0]}], got {tuple(weights.shape)}"
            )
        teacher_support_logits = teacher_support_logits + weights.clamp_min(1e-6).log().unsqueeze(0)

    with torch.no_grad():
        teacher_prob = F.softmax(teacher_support_logits, dim=-1)
    log_prob = F.log_softmax(student_support_logits, dim=-1)
    loss = F.kl_div(log_prob, teacher_prob, reduction="batchmean")
    with torch.no_grad():
        student_argmax = student_logits[:, valid_queries].argmax(dim=0)
        top1_agreement = (student_argmax == teacher_argmax[valid_queries]).float().mean()
        mean_conf = teacher_conf[valid_queries].mean()
    return loss, {
        "valid_ratio": valid_ratio.detach(),
        "teacher_conf": mean_conf.detach(),
        "top1_agreement": top1_agreement.detach(),
    }
