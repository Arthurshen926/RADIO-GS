"""Feature-quality targets for the unified multi-head field.

The helpers here deliberately stay small and tensor-only so both training
code and offline evaluators can share the same quality/visibility semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(
    target: torch.Tensor,
    reference: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if target.shape[-2:] == reference.shape[-2:]:
        return target
    if mode == "nearest":
        return F.interpolate(target.float(), size=reference.shape[-2:], mode=mode)
    return F.interpolate(
        target.float(),
        size=reference.shape[-2:],
        mode=mode,
        align_corners=False,
    )


def cosine_feature_quality_target(
    predicted_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a detached [0, 1] target from student-teacher cosine agreement."""
    if predicted_features.shape != teacher_features.shape:
        raise ValueError(
            "predicted_features and teacher_features must have the same shape, "
            f"got {tuple(predicted_features.shape)} and {tuple(teacher_features.shape)}"
        )
    pred = predicted_features.float()
    target = teacher_features.float()
    cosine = F.cosine_similarity(pred, target, dim=1, eps=eps).unsqueeze(1)
    quality = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = valid_mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = _resize_like(mask, quality, mode="nearest").clamp(0.0, 1.0)
        quality = quality * mask
    return quality.detach()


def visibility_target_from_alpha(
    alpha_map: torch.Tensor,
    reference: torch.Tensor,
    *,
    threshold: float = 0.02,
    binary: bool = False,
) -> torch.Tensor:
    """Resize alpha visibility to a logit-compatible target map."""
    alpha = alpha_map.float()
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.dim() != 4 or alpha.shape[1] != 1:
        raise ValueError(f"Expected alpha_map shaped [B,1,H,W] or [B,H,W], got {tuple(alpha_map.shape)}")
    target = _resize_like(alpha.clamp(0.0, 1.0), reference, mode="bilinear")
    if binary:
        target = (target >= float(threshold)).float()
    return target.detach()
