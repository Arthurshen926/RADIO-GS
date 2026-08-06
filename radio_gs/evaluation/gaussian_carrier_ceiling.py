"""Pure helpers for diagnostic Gaussian semantic-carrier ceiling audits.

The audit is deliberately task contaminated: benchmark masks are used only to
measure whether one scalar membership per frozen RGB Gaussian can reproduce
the masks through the exact fixed visibility operator.  Nothing in this file
is a deployable query interface or a valid benchmark method.
"""

from __future__ import annotations

from typing import Iterable

import torch


def binary_membership_entropy(
    foreground_mass: torch.Tensor,
    total_mass: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return foreground membership and base-2 entropy for primitive rows."""

    foreground = torch.as_tensor(foreground_mass).float()
    total = torch.as_tensor(total_mass).float()
    if foreground.shape != total.shape:
        raise ValueError("foreground_mass and total_mass must have matching shapes")
    numerical_tolerance = max(
        float(eps), 32.0 * float(torch.finfo(foreground.dtype).eps)
    ) * total.clamp_min(1.0)
    if (
        not bool(torch.isfinite(foreground).all())
        or not bool(torch.isfinite(total).all())
        or bool((foreground < 0).any())
        or bool((total < 0).any())
        or bool((foreground > total + numerical_tolerance).any())
    ):
        raise ValueError("membership masses must be finite and satisfy 0 <= fg <= total")
    # Sparse reductions of the same non-negative weights can differ by a few
    # FP32 ulps between ``W.T @ mask`` and ``W.T @ 1``.  Clamp only after the
    # explicit relative-tolerance check above.
    foreground = torch.minimum(foreground, total)
    observed = total > float(eps)
    probability = torch.where(
        observed,
        foreground / total.clamp_min(float(eps)),
        torch.zeros_like(total),
    ).clamp(0.0, 1.0)
    numerical_eps = max(float(eps), float(torch.finfo(probability.dtype).eps))
    clipped = probability.clamp(numerical_eps, 1.0 - numerical_eps)
    entropy = -(clipped * torch.log2(clipped) + (1.0 - clipped) * torch.log2(1.0 - clipped))
    entropy = torch.where(observed, entropy, torch.zeros_like(entropy))
    return probability, entropy


def weighted_carrier_mixing_summary(
    foreground_mass: torch.Tensor,
    total_mass: torch.Tensor,
    *,
    ambiguity_low: float = 0.1,
    ambiguity_high: float = 0.9,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Summarize how much exact contribution is carried by mixed rows."""

    low = float(ambiguity_low)
    high = float(ambiguity_high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("ambiguity bounds must satisfy 0 <= low < high <= 1")
    foreground = torch.as_tensor(foreground_mass).float()
    total = torch.as_tensor(total_mass).float()
    membership, entropy = binary_membership_entropy(foreground, total, eps=eps)
    observed = total > float(eps)
    ambiguous = observed & (membership > low) & (membership < high)
    observed_mass = total[observed].sum()
    foreground_total = foreground.sum()
    return {
        "observed_rows": int(observed.sum()),
        "ambiguous_rows": int(ambiguous.sum()),
        "ambiguous_row_fraction": float(
            ambiguous.float().sum() / observed.float().sum().clamp_min(1.0)
        ),
        "contribution_weighted_entropy": float(
            (entropy * total).sum() / observed_mass.clamp_min(float(eps))
        ),
        "ambiguous_total_mass_fraction": float(
            total[ambiguous].sum() / observed_mass.clamp_min(float(eps))
        ),
        "ambiguous_foreground_mass_fraction": float(
            foreground[ambiguous].sum() / foreground_total.clamp_min(float(eps))
        ),
    }


def binary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute one IoU per final dimension for matching boolean tensors."""

    pred = torch.as_tensor(prediction).bool()
    truth = torch.as_tensor(target).bool()
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have matching shapes")
    if pred.ndim == 1:
        pred = pred[:, None]
        truth = truth[:, None]
    reduce_dims = tuple(range(pred.ndim - 1))
    intersection = (pred & truth).sum(dim=reduce_dims).float()
    union = (pred | truth).sum(dim=reduce_dims).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def threshold_iou_curve(
    scores: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor | Iterable[float],
) -> torch.Tensor:
    """Return ``[T,Q]`` IoUs without coupling thresholds to score extrema."""

    score = torch.as_tensor(scores).float()
    target = torch.as_tensor(targets).bool()
    if score.shape != target.shape or score.ndim not in (1, 2):
        raise ValueError("scores and targets must be matching [P] or [P,Q]")
    if score.ndim == 1:
        score = score[:, None]
        target = target[:, None]
    values = torch.as_tensor(tuple(thresholds), device=score.device).float().reshape(-1)
    if values.numel() == 0 or bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("thresholds must be a non-empty subset of [0,1]")
    prediction = score[None] >= values[:, None, None]
    truth = target[None].expand_as(prediction)
    intersection = (prediction & truth).sum(dim=1).float()
    union = (prediction | truth).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def soft_iou(scores: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return a differentiable soft IoU per final dimension."""

    score = torch.as_tensor(scores).float()
    target = torch.as_tensor(targets).float()
    if score.shape != target.shape:
        raise ValueError("scores and targets must have matching shapes")
    if score.ndim == 1:
        score = score[:, None]
        target = target[:, None]
    intersection = (score * target).sum(dim=0)
    union = score.sum(dim=0) + target.sum(dim=0) - intersection
    return torch.where(union > 0, intersection / union.clamp_min(float(eps)), torch.ones_like(union))
