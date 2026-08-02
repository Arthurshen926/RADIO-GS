"""Fit-only loss and seed-0 gate for the dual-descriptor response branch.

The optimization loss has the predeclared form

``L_struct + lambda_unary * L_unary + lambda_risk * L_risk``, where
``L_struct = L_all_view + 0.1 * L_relation``.  ``L_risk`` is a thin call to
the existing control-referenced scene/query mean-CVaR L1-hinge objective.
The fixed hinge multiplier is empirical: this module makes no mathematical
exact-penalty claim.

Only fit/dev aggregate values enter the pure functions below.  Benchmark
targets and benchmark metrics are deliberately absent from every interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.losses.control_referenced_uncertainty_response_risk import (
    compute_control_referenced_exact_hinge_risk,
)
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
)


RELATION_GRAM_WEIGHT = 0.1
GRADIENT_CALIBRATION_RATIO = 0.25

DEV_MEAN_DELTA_MAX = -0.0025
DEV_GLOBAL_CVAR_DELTA_MAX = 0.005
DEV_WORST_SCENE_MEAN_DELTA_MAX = 0.010
DEV_WORST_SCENE_CVAR_DELTA_MAX = 0.010
UNARY_RELATIVE_DELTA_MAX = 0.0
DESCRIPTOR_DELTA_MIN = -0.002
POINT_RENDER_MAX_ABS_ERROR = 1e-6

UNARY_METRIC_NAMES = (
    "text_response_smooth_l1",
    "text_response_mae",
)
DESCRIPTOR_METRIC_NAMES = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
FIT_CONSTRAINT_NAMES = (
    "global_cvar10_delta",
    "worst_scene_mean_delta",
    "worst_scene_cvar10_delta",
    "independent_unary_delta",
)


def _validate_descriptor_pair(
    semantic_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
) -> None:
    for name, value in (
        ("semantic_descriptors", semantic_descriptors),
        ("teacher_descriptors", teacher_descriptors),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 2 or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating [B,D] tensor")
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if semantic_descriptors.shape != teacher_descriptors.shape:
        raise ValueError("semantic and teacher descriptor shapes differ")


def compute_all_view_cosine_loss(
    semantic_descriptors: torch.Tensor,
    all_view_descriptors: torch.Tensor,
    all_view_valid: torch.Tensor,
) -> torch.Tensor:
    """Return masked mean ``1-cosine`` against detached view descriptors."""

    if not isinstance(semantic_descriptors, torch.Tensor):
        raise TypeError("semantic_descriptors must be a tensor")
    if not isinstance(all_view_descriptors, torch.Tensor):
        raise TypeError("all_view_descriptors must be a tensor")
    if not isinstance(all_view_valid, torch.Tensor):
        raise TypeError("all_view_valid must be a tensor")
    if (
        semantic_descriptors.ndim != 2
        or all_view_descriptors.ndim != 3
        or semantic_descriptors.shape[0] != all_view_descriptors.shape[0]
        or semantic_descriptors.shape[1] != all_view_descriptors.shape[2]
        or all_view_valid.shape != all_view_descriptors.shape[:2]
        or all_view_valid.dtype != torch.bool
        or not semantic_descriptors.is_floating_point()
        or not all_view_descriptors.is_floating_point()
        or semantic_descriptors.numel() == 0
        or not bool(torch.isfinite(semantic_descriptors).all())
        or not bool(torch.isfinite(all_view_descriptors).all())
    ):
        raise ValueError("all-view cosine inputs are invalid")
    valid = all_view_valid.detach().to(device=semantic_descriptors.device)
    if not bool(valid.any()) or not bool(valid.any(dim=1).all()):
        raise ValueError("every descriptor row must have a valid view")
    student = F.normalize(semantic_descriptors.float(), dim=-1)
    teacher = F.normalize(
        all_view_descriptors.detach().to(
            device=student.device, dtype=torch.float32
        ),
        dim=-1,
    )
    cosine = torch.einsum("bd,bvd->bv", student, teacher)
    return (1.0 - cosine)[valid].mean()


def compute_relation_gram_smooth_l1_loss(
    semantic_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
) -> torch.Tensor:
    """Preserve the detached teacher's full normalized relation Gram matrix."""

    _validate_descriptor_pair(semantic_descriptors, teacher_descriptors)
    student = F.normalize(semantic_descriptors.float(), dim=-1)
    teacher = F.normalize(
        teacher_descriptors.detach().to(device=student.device, dtype=torch.float32),
        dim=-1,
    )
    student_gram = student @ student.T
    with torch.no_grad():
        teacher_gram = teacher @ teacher.T
    return F.smooth_l1_loss(student_gram, teacher_gram, reduction="mean")


def compute_dual_descriptor_loss_components(
    semantic_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    all_view_descriptors: torch.Tensor,
    all_view_valid: torch.Tensor,
    frozen_fit_text_bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(structural, all_view, relation, independent_unary)`` losses."""

    _validate_descriptor_pair(semantic_descriptors, teacher_descriptors)
    all_view = compute_all_view_cosine_loss(
        semantic_descriptors, all_view_descriptors, all_view_valid
    )
    relation = compute_relation_gram_smooth_l1_loss(
        semantic_descriptors, teacher_descriptors
    )
    independent_unary = (
        compute_independent_normalized_cosine_response_smooth_l1_loss(
            semantic_descriptors, teacher_descriptors, frozen_fit_text_bank
        )
    )
    structural = all_view + RELATION_GRAM_WEIGHT * relation
    return structural, all_view, relation, independent_unary


def _gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return squared.sqrt()


def calibrate_epoch0_gradient_weights(
    structural_loss: torch.Tensor,
    independent_unary_loss: torch.Tensor,
    control_referenced_risk: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    auxiliary_gradient_ratio: float = GRADIENT_CALIBRATION_RATIO,
    eps: float = 1e-12,
) -> tuple[float, float, dict[str, float]]:
    """Fix lambda weights from epoch-0 gradients; this is not a search step.

    Each weighted auxiliary gradient is calibrated to
    ``auxiliary_gradient_ratio * ||grad(L_struct)||``.  A zero or non-finite
    gradient fails closed instead of silently manufacturing a weight.
    """

    losses = (structural_loss, independent_unary_loss, control_referenced_risk)
    if any(
        not isinstance(loss, torch.Tensor)
        or loss.numel() != 1
        or not loss.is_floating_point()
        or not bool(torch.isfinite(loss).all())
        for loss in losses
    ):
        raise ValueError("gradient calibration losses must be finite scalars")
    parameter_tuple = tuple(parameters)
    if not parameter_tuple or any(
        not isinstance(parameter, torch.Tensor) or not parameter.requires_grad
        for parameter in parameter_tuple
    ):
        raise ValueError("gradient calibration requires trainable tensors")
    if (
        not math.isfinite(float(auxiliary_gradient_ratio))
        or float(auxiliary_gradient_ratio) <= 0.0
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        raise ValueError("gradient calibration scalars are invalid")

    structural_norm = _gradient_l2_norm(structural_loss, parameter_tuple)
    unary_norm = _gradient_l2_norm(independent_unary_loss, parameter_tuple)
    risk_norm = _gradient_l2_norm(control_referenced_risk, parameter_tuple)
    norms = (structural_norm, unary_norm, risk_norm)
    if any(
        not bool(torch.isfinite(value)) or float(value) <= float(eps)
        for value in norms
    ):
        raise ValueError("epoch-0 gradient calibration encountered a degenerate norm")
    target_norm = float(auxiliary_gradient_ratio) * structural_norm
    lambda_unary = float((target_norm / unary_norm).cpu())
    lambda_risk = float((target_norm / risk_norm).cpu())
    report = {
        "structural_gradient_l2": float(structural_norm.cpu()),
        "independent_unary_gradient_l2": float(unary_norm.cpu()),
        "control_referenced_risk_gradient_l2": float(risk_norm.cpu()),
        "auxiliary_gradient_ratio": float(auxiliary_gradient_ratio),
        "target_weighted_auxiliary_gradient_l2": float(target_norm.cpu()),
        "lambda_unary": lambda_unary,
        "lambda_risk": lambda_risk,
        "epoch": 0,
        "selection_kind": "deterministic_gradient_norm_calibration_not_search",
    }
    return lambda_unary, lambda_risk, report


def compute_dual_descriptor_response_risk(
    candidate_unit_loss: torch.Tensor,
    candidate_valid: torch.Tensor,
    control_unit_loss: torch.Tensor,
    control_valid: torch.Tensor,
    semantic_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    all_view_descriptors: torch.Tensor,
    all_view_valid: torch.Tensor,
    frozen_fit_text_bank: torch.Tensor,
    control_independent_unary_loss: torch.Tensor | float,
    *,
    lambda_unary: float,
    lambda_risk: float,
    cvar_tail_fraction: float = 0.10,
    global_cvar_tolerance: float = 0.005,
    worst_scene_mean_tolerance: float = 0.010,
    worst_scene_cvar_tolerance: float = 0.010,
    unary_delta_tolerance: float = 0.0,
    l1_hinge_weight: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose the four declared losses without reimplementing fit risk."""

    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (lambda_unary, lambda_risk)
    ):
        raise ValueError("lambda_unary and lambda_risk must be positive and finite")
    structural, all_view, relation, independent_unary = (
        compute_dual_descriptor_loss_components(
            semantic_descriptors,
            teacher_descriptors,
            all_view_descriptors,
            all_view_valid,
            frozen_fit_text_bank,
        )
    )
    risk, risk_statistics = compute_control_referenced_exact_hinge_risk(
        candidate_unit_loss,
        candidate_valid,
        control_unit_loss,
        control_valid,
        independent_unary,
        control_independent_unary_loss,
        cvar_tail_fraction=cvar_tail_fraction,
        global_cvar_tolerance=global_cvar_tolerance,
        worst_scene_mean_tolerance=worst_scene_mean_tolerance,
        worst_scene_cvar_tolerance=worst_scene_cvar_tolerance,
        unary_delta_tolerance=unary_delta_tolerance,
        exact_penalty_weight=l1_hinge_weight,
        eps=eps,
    )
    total = (
        structural
        + float(lambda_unary) * independent_unary
        + float(lambda_risk) * risk
    )
    return total, {
        "structural_loss": structural.detach(),
        "all_view_cosine_loss": all_view.detach(),
        "relation_gram_smooth_l1_loss": relation.detach(),
        "relation_gram_weight": RELATION_GRAM_WEIGHT,
        "independent_normalized_cosine_response_smooth_l1_loss": (
            independent_unary.detach()
        ),
        "control_referenced_risk": risk.detach(),
        "lambda_unary": float(lambda_unary),
        "lambda_risk": float(lambda_risk),
        "objective": total.detach(),
        "control_referenced_risk_statistics": risk_statistics,
        "constraint_penalty_contract": {
            "kind": "unsmoothed_l1_hinge",
            "fixed_empirical_weight": float(l1_hinge_weight),
            "mathematical_exact_penalty_guarantee": False,
            "reason": (
                "no bound on the unknown optimal Lagrange multipliers is claimed"
            ),
        },
    }


def _finite_metric_mapping(
    values: Mapping[str, float],
    expected_names: Sequence[str],
    label: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(expected_names):
        raise ValueError(f"{label} must contain exactly {tuple(expected_names)}")
    result = {name: float(values[name]) for name in expected_names}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} must be finite")
    return result


def build_seed0_single_conjunction_gate(
    *,
    selected_epoch: int,
    dev_normalized_mean_delta: float,
    dev_global_cvar10_delta: float,
    dev_worst_scene_mean_delta: float,
    dev_worst_scene_cvar10_delta: float,
    validation_unary_relative_deltas: Mapping[str, float],
    validation_descriptor_deltas: Mapping[str, float],
    official_token_bitwise_equal: bool,
    official_descriptor_bitwise_equal: bool,
    fit_constraint_checks: Mapping[str, bool],
    point_render_max_abs_error: float,
) -> dict[str, Any]:
    """Return the sole seed-0 advance gate as one auditable conjunction."""

    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int):
        raise TypeError("selected_epoch must be an integer")
    scalar_values = {
        "dev_normalized_mean_delta": float(dev_normalized_mean_delta),
        "dev_global_cvar10_delta": float(dev_global_cvar10_delta),
        "dev_worst_scene_mean_delta": float(dev_worst_scene_mean_delta),
        "dev_worst_scene_cvar10_delta": float(dev_worst_scene_cvar10_delta),
        "point_render_max_abs_error": float(point_render_max_abs_error),
    }
    if any(not math.isfinite(value) for value in scalar_values.values()):
        raise ValueError("gate scalar metrics must be finite")
    if scalar_values["point_render_max_abs_error"] < 0.0:
        raise ValueError("point/render error must be non-negative")
    unary = _finite_metric_mapping(
        validation_unary_relative_deltas,
        UNARY_METRIC_NAMES,
        "validation unary deltas",
    )
    descriptor = _finite_metric_mapping(
        validation_descriptor_deltas,
        DESCRIPTOR_METRIC_NAMES,
        "validation descriptor deltas",
    )
    if not isinstance(fit_constraint_checks, Mapping) or set(
        fit_constraint_checks
    ) != set(FIT_CONSTRAINT_NAMES):
        raise ValueError(
            f"fit_constraint_checks must contain exactly {FIT_CONSTRAINT_NAMES}"
        )
    if any(
        type(fit_constraint_checks[name]) is not bool
        for name in FIT_CONSTRAINT_NAMES
    ):
        raise TypeError("fit constraint checks must be booleans")
    if type(official_token_bitwise_equal) is not bool or type(
        official_descriptor_bitwise_equal
    ) is not bool:
        raise TypeError("official bitwise checks must be booleans")

    checks = {
        "selected_epoch_gt_zero": selected_epoch > 0,
        "dev_normalized_mean_delta_le_negative_0p0025": (
            scalar_values["dev_normalized_mean_delta"]
            <= DEV_MEAN_DELTA_MAX + 1e-12
        ),
        "dev_global_cvar10_delta_le_0p005": (
            scalar_values["dev_global_cvar10_delta"]
            <= DEV_GLOBAL_CVAR_DELTA_MAX + 1e-12
        ),
        "dev_worst_scene_mean_delta_le_0p010": (
            scalar_values["dev_worst_scene_mean_delta"]
            <= DEV_WORST_SCENE_MEAN_DELTA_MAX + 1e-12
        ),
        "dev_worst_scene_cvar10_delta_le_0p010": (
            scalar_values["dev_worst_scene_cvar10_delta"]
            <= DEV_WORST_SCENE_CVAR_DELTA_MAX + 1e-12
        ),
        "validation_unary_relative_deltas_le_zero": all(
            value <= UNARY_RELATIVE_DELTA_MAX + 1e-12 for value in unary.values()
        ),
        "validation_descriptor_deltas_ge_negative_0p002": all(
            value >= DESCRIPTOR_DELTA_MIN - 1e-12 for value in descriptor.values()
        ),
        "official_outputs_bitwise_equal": (
            official_token_bitwise_equal and official_descriptor_bitwise_equal
        ),
        "all_fit_constraints_feasible": all(
            fit_constraint_checks[name] for name in FIT_CONSTRAINT_NAMES
        ),
        "point_render_max_abs_error_le_1e_minus_6": (
            scalar_values["point_render_max_abs_error"]
            <= POINT_RENDER_MAX_ABS_ERROR + 1e-15
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "dual_descriptor_seed0_single_conjunction_v1",
        "seed": 0,
        "checks": checks,
        "passed": passed,
        "conjunction": "all(checks.values())",
        "measurements": {
            **scalar_values,
            "validation_unary_relative_deltas": unary,
            "validation_descriptor_deltas": descriptor,
            "official_token_bitwise_equal": official_token_bitwise_equal,
            "official_descriptor_bitwise_equal": official_descriptor_bitwise_equal,
            "fit_constraint_checks": {
                name: fit_constraint_checks[name] for name in FIT_CONSTRAINT_NAMES
            },
        },
        "thresholds": {
            "dev_normalized_mean_delta_max": DEV_MEAN_DELTA_MAX,
            "dev_global_cvar10_delta_max": DEV_GLOBAL_CVAR_DELTA_MAX,
            "dev_worst_scene_mean_delta_max": DEV_WORST_SCENE_MEAN_DELTA_MAX,
            "dev_worst_scene_cvar10_delta_max": DEV_WORST_SCENE_CVAR_DELTA_MAX,
            "validation_unary_relative_delta_max": UNARY_RELATIVE_DELTA_MAX,
            "validation_descriptor_delta_min": DESCRIPTOR_DELTA_MIN,
            "point_render_max_abs_error_max": POINT_RENDER_MAX_ABS_ERROR,
        },
        "data_boundary": {
            "fit_and_frozen_dev_aggregates_only": True,
            "benchmark_targets_or_metrics_used": False,
        },
    }
