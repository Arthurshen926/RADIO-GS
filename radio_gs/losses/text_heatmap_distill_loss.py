"""Text-query heatmap distillation losses."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _resize_for_loss(features: Tensor, downsample: int) -> Tensor:
    if downsample <= 1:
        return features
    height, width = features.shape[-2:]
    target_size = (max(1, height // downsample), max(1, width // downsample))
    return F.interpolate(
        features.float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def compute_text_heatmap_distill_loss(
    rendered_siglip: Tensor,
    teacher_siglip: Tensor,
    text_embeddings: Tensor,
    *,
    downsample: int = 1,
    temperature: float = 20.0,
    mode: str = "query",
) -> Tuple[Tensor, Dict[str, Any]]:
    """Match rendered and teacher text-query response distributions.

    Args:
        rendered_siglip: Rendered feature map already projected to SigLIP space,
            shaped ``[B, C, H, W]``.
        teacher_siglip: Teacher feature map projected to the same SigLIP space.
        text_embeddings: L2-normalized query embeddings, shaped ``[Q, C]``.
        downsample: Optional spatial downsample factor for cheaper training.
        temperature: Logit scale applied before the query softmax.
        mode: ``"query"`` matches per-pixel query distributions, ``"spatial"``
            matches per-query spatial response distributions, and
            ``"query_spatial"`` averages both terms.
    """
    normalized_mode = mode.lower().replace("-", "_")
    if normalized_mode == "both":
        normalized_mode = "query_spatial"
    if normalized_mode not in {"query", "spatial", "query_spatial"}:
        raise ValueError(
            "mode must be one of 'query', 'spatial', 'query_spatial', or 'both'"
        )
    if rendered_siglip.ndim != 4 or teacher_siglip.ndim != 4:
        raise ValueError("rendered_siglip and teacher_siglip must be [B, C, H, W]")
    if rendered_siglip.shape != teacher_siglip.shape:
        raise ValueError(
            "rendered_siglip and teacher_siglip shape mismatch: "
            f"{tuple(rendered_siglip.shape)} vs {tuple(teacher_siglip.shape)}"
        )
    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must be [Q, C]")
    if rendered_siglip.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "Feature/text dim mismatch: "
            f"{rendered_siglip.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if text_embeddings.shape[0] == 0:
        zero = rendered_siglip.sum() * 0.0
        return zero, {
            "num_queries": 0,
            "height": 0,
            "width": 0,
            "mode": normalized_mode,
        }
    if torch.equal(rendered_siglip, teacher_siglip):
        zero = rendered_siglip.sum() * 0.0
        return zero, {
            "num_queries": int(text_embeddings.shape[0]),
            "height": int(rendered_siglip.shape[-2]),
            "width": int(rendered_siglip.shape[-1]),
            "temperature": float(temperature),
            "mode": normalized_mode,
        }

    rendered = _resize_for_loss(rendered_siglip, int(downsample))
    teacher = _resize_for_loss(teacher_siglip, int(downsample))
    rendered = F.normalize(rendered.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)
    text = F.normalize(text_embeddings.float(), dim=1)

    logit_scale = float(temperature)
    rendered_logits = torch.einsum("bchw,qc->bqhw", rendered, text) * logit_scale
    with torch.no_grad():
        teacher_logits = torch.einsum("bchw,qc->bqhw", teacher, text) * logit_scale

    loss_terms = []
    if normalized_mode in {"query", "query_spatial"}:
        with torch.no_grad():
            teacher_query_probs = F.softmax(teacher_logits, dim=1)
        query_loss = F.kl_div(
            F.log_softmax(rendered_logits, dim=1),
            teacher_query_probs,
            reduction="batchmean",
        )
        loss_terms.append(query_loss)
    if normalized_mode in {"spatial", "query_spatial"}:
        batch, queries, height, width = rendered_logits.shape
        rendered_spatial = rendered_logits.flatten(2).reshape(
            batch * queries, height * width
        )
        with torch.no_grad():
            teacher_spatial = teacher_logits.flatten(2).reshape(
                batch * queries, height * width
            )
            teacher_spatial_probs = F.softmax(teacher_spatial, dim=-1)
        spatial_loss = F.kl_div(
            F.log_softmax(rendered_spatial, dim=-1),
            teacher_spatial_probs,
            reduction="batchmean",
        )
        loss_terms.append(spatial_loss)

    loss = torch.stack(loss_terms).mean().clamp_min(0.0)
    stats: Dict[str, Any] = {
        "num_queries": int(text.shape[0]),
        "height": int(rendered.shape[-2]),
        "width": int(rendered.shape[-1]),
        "temperature": float(temperature),
        "mode": normalized_mode,
    }
    return loss, stats
