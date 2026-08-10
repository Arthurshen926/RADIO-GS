"""Query-free geometry preservation for common-component-dominated descriptors.

The functions in this module operate only on unit visual descriptors.  They
remove one source-train-only teacher centroid before comparing directions, so
that a scene-independent SigLIP component cannot make a collapsed predictor
look accurate.  No text, query, target, or scene identifier is consumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F


RAW_COSINE_WEIGHT = 0.25
CENTERED_RESIDUAL_WEIGHT = 1.0
CENTERED_GRAM_WEIGHT = 0.50
ABSOLUTE_VISUAL_PROBE_WEIGHT = 0.25
SPREAD_FLOOR_WEIGHT = 0.25
MINIMUM_SPREAD_RATIO = 0.75
EPSILON = 1e-6


def _unit_rows(value: torch.Tensor, *, label: str) -> torch.Tensor:
    rows = torch.as_tensor(value)
    if rows.ndim != 2 or not rows.is_floating_point() or rows.shape[0] < 1:
        raise ValueError(f"{label} must be a nonempty floating-point matrix")
    rows = rows.float()
    if not bool(torch.isfinite(rows).all()):
        raise ValueError(f"{label} must be finite")
    return F.normalize(rows, dim=-1)


def teacher_prototype(
    teacher_views: torch.Tensor, teacher_mask: torch.Tensor
) -> torch.Tensor:
    """Return one unit teacher prototype per row from a padded view tensor."""

    views = torch.as_tensor(teacher_views)
    mask = torch.as_tensor(teacher_mask, device=views.device)
    if (
        views.ndim != 3
        or not views.is_floating_point()
        or mask.dtype != torch.bool
        or mask.shape != views.shape[:2]
        or not bool(mask.any(dim=1).all())
        or not bool(torch.isfinite(views[mask]).all())
    ):
        raise ValueError("teacher view layout differs")
    unit = F.normalize(views.float(), dim=-1)
    count = mask.sum(dim=1, keepdim=True)
    mean = (unit * mask[..., None]).sum(dim=1) / count
    return F.normalize(mean, dim=-1)


def fit_equal_scene_teacher_center(
    scene_teacher_prototypes: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Fit the common teacher component with equal source-scene weighting."""

    if not scene_teacher_prototypes:
        raise ValueError("at least one source-train scene is required")
    means: list[torch.Tensor] = []
    width: int | None = None
    for value in scene_teacher_prototypes:
        rows = _unit_rows(value, label="scene teacher prototypes")
        if width is None:
            width = int(rows.shape[1])
        elif rows.shape[1] != width:
            raise ValueError("scene teacher descriptor dimensions differ")
        means.append(rows.mean(dim=0))
    center = torch.stack(means).mean(dim=0)
    if not bool(torch.isfinite(center).all()) or float(center.norm()) >= 1.0 + 1e-5:
        raise ValueError("source-train teacher center differs")
    return center.detach().float().contiguous()


def centered_direction(value: torch.Tensor, teacher_center: torch.Tensor) -> torch.Tensor:
    """Normalize the direction remaining after subtracting the fixed center."""

    rows = _unit_rows(value, label="descriptor")
    center = torch.as_tensor(
        teacher_center, device=rows.device, dtype=rows.dtype
    ).reshape(-1)
    if (
        center.shape != (rows.shape[1],)
        or not bool(torch.isfinite(center).all())
        or float(center.norm()) >= 1.0 + 1e-5
    ):
        raise ValueError("teacher center differs")
    return F.normalize(rows - center[None], dim=-1, eps=EPSILON)


def _off_diagonal(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


def contrast_preserving_objective(
    student: torch.Tensor,
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    teacher_center: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Raw fidelity plus centered direction, Gram, and non-collapse losses."""

    predicted = _unit_rows(student, label="student descriptor")
    views = torch.as_tensor(
        teacher_views, device=predicted.device, dtype=predicted.dtype
    )
    mask = torch.as_tensor(teacher_mask, device=predicted.device)
    if views.shape[0] != predicted.shape[0] or views.shape[2] != predicted.shape[1]:
        raise ValueError("student and teacher descriptor layouts differ")
    prototype = teacher_prototype(views, mask)

    view_cosine = torch.einsum("bd,bvd->bv", predicted, F.normalize(views, dim=-1))
    raw_loss = (
        ((1.0 - view_cosine) * mask).sum(dim=1) / mask.sum(dim=1)
    ).mean()

    student_residual = centered_direction(predicted, teacher_center)
    teacher_residual = centered_direction(prototype, teacher_center)
    residual_cosine = (student_residual * teacher_residual).sum(dim=-1)
    residual_loss = 1.0 - residual_cosine.mean()

    if predicted.shape[0] >= 2:
        pairs = _off_diagonal(predicted.shape[0], predicted.device)
        student_gram = student_residual @ student_residual.T
        teacher_gram = teacher_residual @ teacher_residual.T
        gram_loss = F.smooth_l1_loss(student_gram[pairs], teacher_gram[pairs])
        # Teacher prototypes are query-free visual probes.  Unlike the
        # centered self-Gram term above, this keeps the absolute raw response
        # scale calibrated: student_i·teacher_j must match teacher_i·teacher_j.
        student_probe_response = predicted @ prototype.T
        teacher_probe_response = prototype @ prototype.T
        absolute_probe_loss = F.smooth_l1_loss(
            student_probe_response[pairs], teacher_probe_response[pairs]
        )
    else:
        gram_loss = predicted.sum() * 0.0
        absolute_probe_loss = predicted.sum() * 0.0

    student_spread = ((predicted - predicted.mean(dim=0)) ** 2).sum(dim=-1).mean()
    teacher_spread = ((prototype - prototype.mean(dim=0)) ** 2).sum(dim=-1).mean()
    spread_loss = F.relu(
        MINIMUM_SPREAD_RATIO * teacher_spread.detach() - student_spread
    ) / teacher_spread.detach().clamp_min(EPSILON)

    total = (
        RAW_COSINE_WEIGHT * raw_loss
        + CENTERED_RESIDUAL_WEIGHT * residual_loss
        + CENTERED_GRAM_WEIGHT * gram_loss
        + ABSOLUTE_VISUAL_PROBE_WEIGHT * absolute_probe_loss
        + SPREAD_FLOOR_WEIGHT * spread_loss
    )
    return total, {
        "raw_all_view_cosine_loss": raw_loss,
        "teacher_centered_residual_cosine_loss": residual_loss,
        "teacher_centered_gram_loss": gram_loss,
        "absolute_visual_probe_calibration_loss": absolute_probe_loss,
        "variance_noncollapse_loss": spread_loss,
        "student_spread": student_spread,
        "teacher_spread": teacher_spread,
        "student_to_teacher_spread_ratio": student_spread
        / teacher_spread.clamp_min(EPSILON),
    }


def deterministic_pairs(size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed O(N) ring pairs spanning local and long offsets."""

    if type(size) is not int or size < 2:
        raise ValueError("pair geometry requires at least two rows")
    source = torch.arange(size, device=device)
    offsets = sorted({1, max(1, size // 3), max(1, size // 2)})
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    for offset in offsets:
        candidate = (source + offset) % size
        valid = candidate != source
        left.append(source[valid])
        right.append(candidate[valid])
    return torch.cat(left), torch.cat(right)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator) <= EPSILON:
        return left.new_zeros(())
    return (left_centered * right_centered).sum() / denominator


@torch.no_grad()
def contrast_metrics_from_prototypes(
    student: torch.Tensor,
    teacher_prototypes: torch.Tensor,
    teacher_center: torch.Tensor,
    *,
    row_all_view_cosine: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute metrics from prototypes without retaining padded teacher views."""

    predicted = _unit_rows(student, label="student descriptor")
    prototype = _unit_rows(
        torch.as_tensor(teacher_prototypes, device=predicted.device),
        label="teacher prototypes",
    )
    if predicted.shape != prototype.shape or predicted.shape[0] < 2:
        raise ValueError("contrast metrics require at least two rows")
    if row_all_view_cosine is None:
        row_cosine = (predicted * prototype).sum(dim=-1)
    else:
        row_cosine = torch.as_tensor(
            row_all_view_cosine, device=predicted.device, dtype=predicted.dtype
        ).reshape(-1)
        if row_cosine.shape != (predicted.shape[0],) or not bool(
            torch.isfinite(row_cosine).all()
        ):
            raise ValueError("row all-view cosine layout differs")
    student_residual = centered_direction(predicted, teacher_center)
    teacher_residual = centered_direction(prototype, teacher_center)
    residual_cosine = (student_residual * teacher_residual).sum(dim=-1)
    left, right = deterministic_pairs(predicted.shape[0], device=predicted.device)
    student_pairs = (student_residual[left] * student_residual[right]).sum(dim=-1)
    teacher_pairs = (teacher_residual[left] * teacher_residual[right]).sum(dim=-1)
    student_probe_response = (predicted[left] * prototype[right]).sum(dim=-1)
    teacher_probe_response = (prototype[left] * prototype[right]).sum(dim=-1)
    teacher_probe_std = teacher_probe_response.std(unbiased=False)
    student_spread = ((predicted - predicted.mean(dim=0)) ** 2).sum(dim=-1).mean()
    teacher_spread = ((prototype - prototype.mean(dim=0)) ** 2).sum(dim=-1).mean()
    return {
        "eligible_rows": int(predicted.shape[0]),
        "mean_all_view_cosine": float(row_cosine.mean()),
        "p05_row_mean_all_view_cosine": float(torch.quantile(row_cosine, 0.05)),
        "mean_teacher_centered_residual_cosine": float(residual_cosine.mean()),
        "p05_teacher_centered_residual_cosine": float(
            torch.quantile(residual_cosine, 0.05)
        ),
        "student_centroid_norm": float(predicted.mean(dim=0).norm()),
        "teacher_centroid_norm": float(prototype.mean(dim=0).norm()),
        "student_spread": float(student_spread),
        "teacher_spread": float(teacher_spread),
        "student_to_teacher_spread_ratio": float(
            student_spread / teacher_spread.clamp_min(EPSILON)
        ),
        "teacher_centered_pair_gram_mae": float(
            (student_pairs - teacher_pairs).abs().mean()
        ),
        "teacher_centered_pair_gram_correlation": float(
            _correlation(student_pairs, teacher_pairs)
        ),
        "teacher_centered_pair_count": int(left.numel()),
        "absolute_visual_probe_response_mae": float(
            (student_probe_response - teacher_probe_response).abs().mean()
        ),
        "absolute_visual_probe_response_correlation": float(
            _correlation(student_probe_response, teacher_probe_response)
        ),
        "absolute_visual_probe_response_std_ratio": float(
            student_probe_response.std(unbiased=False)
            / teacher_probe_std.clamp_min(EPSILON)
        ),
    }


@torch.no_grad()
def contrast_metrics(
    student: torch.Tensor,
    teacher_views: torch.Tensor,
    teacher_mask: torch.Tensor,
    teacher_center: torch.Tensor,
) -> dict[str, Any]:
    """Compute source-only raw fidelity and centered non-collapse diagnostics."""

    predicted = _unit_rows(student, label="student descriptor")
    views = torch.as_tensor(
        teacher_views, device=predicted.device, dtype=predicted.dtype
    )
    mask = torch.as_tensor(teacher_mask, device=predicted.device)
    prototype = teacher_prototype(views, mask)
    view_cosine = torch.einsum("bd,bvd->bv", predicted, F.normalize(views, dim=-1))
    row_cosine = (view_cosine * mask).sum(dim=1) / mask.sum(dim=1)
    return contrast_metrics_from_prototypes(
        predicted,
        prototype,
        teacher_center,
        row_all_view_cosine=row_cosine,
    )


__all__ = [
    "ABSOLUTE_VISUAL_PROBE_WEIGHT",
    "CENTERED_GRAM_WEIGHT",
    "CENTERED_RESIDUAL_WEIGHT",
    "EPSILON",
    "MINIMUM_SPREAD_RATIO",
    "RAW_COSINE_WEIGHT",
    "SPREAD_FLOOR_WEIGHT",
    "centered_direction",
    "contrast_metrics",
    "contrast_metrics_from_prototypes",
    "contrast_preserving_objective",
    "deterministic_pairs",
    "fit_equal_scene_teacher_center",
    "teacher_prototype",
]
