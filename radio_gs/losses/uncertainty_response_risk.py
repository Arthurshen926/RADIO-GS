"""Differentiable scene-query risk for multiview text-response distillation."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.losses.direct_point_query_logit_distill_loss import (
    _scene_group_indices,
    _validate_cosine_response_inputs,
    compute_multiview_teacher_response_uncertainty,
    fractional_upper_cvar,
)


def compute_uncertainty_weighted_scene_query_pairwise_gap_units(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    standard_error_multiplier: float = 2.0,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return differentiable weighted pairwise loss for each scene/query.

    The returned unit tensor has shape ``[S,Q]`` and remains connected only to
    ``student_descriptors``.  Teacher descriptors, per-view response variance,
    confidence weights, validity, and the frozen text bank are all detached.
    Invalid scene/query units are represented by zero and identified by the
    separate boolean validity tensor.
    """

    _validate_cosine_response_inputs(
        student_descriptors,
        teacher_descriptors,
        frozen_text_bank,
    )
    if not isinstance(teacher_view_descriptors, torch.Tensor) or not isinstance(
        teacher_mask, torch.Tensor
    ):
        raise TypeError("teacher view descriptors and mask must be tensors")
    if (
        teacher_view_descriptors.ndim != 3
        or teacher_view_descriptors.shape[0] != student_descriptors.shape[0]
        or teacher_view_descriptors.shape[2] != student_descriptors.shape[1]
        or teacher_mask.shape != teacher_view_descriptors.shape[:2]
    ):
        raise ValueError("teacher view descriptors/mask must align with [B,V,D]")
    for label, value in (
        ("standard_error_multiplier", standard_error_multiplier),
        ("tie_tolerance", tie_tolerance),
        ("eps", eps),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    if standard_error_multiplier < 0.0 or tie_tolerance < 0.0 or eps <= 0.0:
        raise ValueError("risk scalar parameters are outside their valid ranges")

    student = F.normalize(student_descriptors.float(), dim=-1, eps=float(eps))
    uncertainty = compute_multiview_teacher_response_uncertainty(
        teacher_view_descriptors.detach().to(device=student.device),
        teacher_mask.detach().to(device=student.device),
        frozen_text_bank.detach().to(device=student.device),
        eps=float(eps),
    )
    with torch.no_grad():
        teacher = F.normalize(
            teacher_descriptors.detach().to(
                device=student.device, dtype=torch.float32
            ),
            dim=-1,
            eps=float(eps),
        )
        text = F.normalize(
            frozen_text_bank.detach().to(
                device=student.device, dtype=torch.float32
            ),
            dim=-1,
            eps=float(eps),
        )
        teacher_responses = teacher @ text.T
        response_standard_error = uncertainty["response_standard_error"]
    student_responses = student @ text.T
    groups = _scene_group_indices(
        scene_ids,
        batch_size=student_descriptors.shape[0],
        device=student.device,
    )

    unit_losses: list[torch.Tensor] = []
    unit_validity: list[torch.Tensor] = []
    unit_weight_sums: list[torch.Tensor] = []
    valid_weights: list[torch.Tensor] = []
    valid_pair_query_count = 0
    for indices in groups:
        region_count = int(indices.numel())
        if region_count < 2:
            raise ValueError(
                "differentiable risk requires complete scenes with at least "
                "two regions"
            )
        student_scene = student_responses.index_select(0, indices)
        teacher_scene = teacher_responses.index_select(0, indices)
        standard_error_scene = response_standard_error.index_select(0, indices)
        pairs = torch.triu_indices(
            region_count,
            region_count,
            offset=1,
            device=student.device,
        )
        student_gaps = student_scene[pairs[0]] - student_scene[pairs[1]]
        with torch.no_grad():
            teacher_gaps = teacher_scene[pairs[0]] - teacher_scene[pairs[1]]
            teacher_span = teacher_scene.amax(dim=0) - teacher_scene.amin(dim=0)
            pair_standard_error = torch.sqrt(
                standard_error_scene[pairs[0]].square()
                + standard_error_scene[pairs[1]].square()
            )
            confidence = teacher_gaps.abs() / (
                teacher_gaps.abs()
                + float(standard_error_multiplier) * pair_standard_error
                + float(eps)
            )
            valid = (
                teacher_gaps.abs() > float(tie_tolerance)
            ) & (teacher_span > float(tie_tolerance)).unsqueeze(0)
            weights = confidence * valid.to(dtype=confidence.dtype)
            weight_sum = weights.sum(dim=0)
            query_valid = weight_sum > float(eps)
            scale = teacher_span.clamp_min(float(eps)).unsqueeze(0)
        pair_loss = F.smooth_l1_loss(
            student_gaps / scale,
            teacher_gaps / scale,
            reduction="none",
        )
        query_loss = (pair_loss * weights).sum(dim=0) / weight_sum.clamp_min(
            float(eps)
        )
        # Multiplication instead of masked_fill keeps a simple differentiable
        # zero for invalid units without connecting validity to autograd.
        query_loss = query_loss * query_valid.to(dtype=query_loss.dtype)
        unit_losses.append(query_loss)
        unit_validity.append(query_valid)
        unit_weight_sums.append(weight_sum)
        if bool(valid.any()):
            valid_weights.append(confidence[valid])
            valid_pair_query_count += int(valid.sum())

    units = torch.stack(unit_losses)
    validity = torch.stack(unit_validity)
    weight_sums = torch.stack(unit_weight_sums)
    if not bool(validity.any()):
        raise ValueError("differentiable risk has no valid scene-query unit")
    weights_flat = torch.cat(valid_weights)
    return units, validity.detach(), {
        "scene_count": units.new_tensor(len(groups)).detach(),
        "valid_scene_query_count": validity.sum().detach(),
        "valid_pair_query_count": units.new_tensor(
            valid_pair_query_count
        ).detach(),
        "uncertainty_weight_mean": weights_flat.mean().detach(),
        "scene_query_weight_sum": weight_sums.detach(),
        "teacher_response_variance_mean": uncertainty[
            "response_variance"
        ].mean().detach(),
    }


def compute_equal_scene_mean_fractional_cvar_risk(
    scene_query_unit_loss: torch.Tensor,
    scene_query_valid: torch.Tensor,
    *,
    mean_weight: float = 0.5,
    cvar_weight: float = 0.5,
    cvar_tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average ``mean_weight*mean + cvar_weight*upper-CVaR`` over scenes.

    Each scene contributes exactly one scalar regardless of its number of
    valid queries.  Fractional CVaR is differentiable with respect to the
    selected worst units and preserves the requested tail mass without a
    rounded top-k.
    """

    units = scene_query_unit_loss
    validity = scene_query_valid
    if not isinstance(units, torch.Tensor) or not isinstance(validity, torch.Tensor):
        raise TypeError("scene-query units and validity must be tensors")
    if (
        units.ndim != 2
        or units.numel() == 0
        or not units.is_floating_point()
        or validity.dtype != torch.bool
        or validity.shape != units.shape
        or not bool(torch.isfinite(units).all())
        or not bool(validity.any(dim=1).all())
    ):
        raise ValueError("scene-query risk inputs are invalid")
    for label, value in (
        ("mean_weight", mean_weight),
        ("cvar_weight", cvar_weight),
        ("cvar_tail_fraction", cvar_tail_fraction),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    if (
        mean_weight < 0.0
        or cvar_weight < 0.0
        or not math.isclose(mean_weight + cvar_weight, 1.0, abs_tol=1e-12)
        or not 0.0 < cvar_tail_fraction <= 1.0
    ):
        raise ValueError("mean/CVaR risk weights or tail fraction are invalid")

    scene_means: list[torch.Tensor] = []
    scene_cvars: list[torch.Tensor] = []
    scene_risks: list[torch.Tensor] = []
    for scene_index in range(units.shape[0]):
        active = units[scene_index][validity[scene_index]]
        mean = active.mean()
        cvar = fractional_upper_cvar(active, float(cvar_tail_fraction))
        risk = float(mean_weight) * mean + float(cvar_weight) * cvar
        scene_means.append(mean)
        scene_cvars.append(cvar)
        scene_risks.append(risk)
    means = torch.stack(scene_means)
    cvars = torch.stack(scene_cvars)
    risks = torch.stack(scene_risks)
    return risks.mean(), {
        "scene_mean": means.detach(),
        "scene_upper_fractional_cvar": cvars.detach(),
        "scene_risk": risks.detach(),
        "equal_scene_count": risks.new_tensor(len(scene_risks)).detach(),
    }


def compute_uncertainty_weighted_pairwise_mean_cvar_risk(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    teacher_view_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    frozen_text_bank: torch.Tensor,
    scene_ids: Sequence[Hashable] | torch.Tensor,
    *,
    standard_error_multiplier: float = 2.0,
    tie_tolerance: float = 1e-6,
    eps: float = 1e-6,
    mean_weight: float = 0.5,
    cvar_weight: float = 0.5,
    cvar_tail_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convenience composition of differentiable units and equal-scene risk."""

    units, validity, unit_stats = (
        compute_uncertainty_weighted_scene_query_pairwise_gap_units(
            student_descriptors,
            teacher_descriptors,
            teacher_view_descriptors,
            teacher_mask,
            frozen_text_bank,
            scene_ids,
            standard_error_multiplier=standard_error_multiplier,
            tie_tolerance=tie_tolerance,
            eps=eps,
        )
    )
    risk, risk_stats = compute_equal_scene_mean_fractional_cvar_risk(
        units,
        validity,
        mean_weight=mean_weight,
        cvar_weight=cvar_weight,
        cvar_tail_fraction=cvar_tail_fraction,
    )
    return risk, {
        **unit_stats,
        **risk_stats,
        "scene_query_unit_loss": units.detach(),
        "scene_query_valid": validity.detach(),
    }

