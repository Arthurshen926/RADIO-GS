"""Strongly regularized per-view affine depth calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AffineDepthCalibration:
    scale: float
    offset: float
    sample_count: int
    median_absolute_residual: float
    accepted: bool
    rejection_reason: str | None

    def apply(self, depth: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(depth) * self.scale + self.offset


def fit_constrained_affine_depth(
    predicted_depth: torch.Tensor,
    reference_depth: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    scale_prior_weight: float = 100.0,
    offset_prior_weight: float = 100.0,
    maximum_scale_deviation: float = 0.2,
    maximum_offset_fraction: float = 0.1,
    iterations: int = 8,
) -> AffineDepthCalibration:
    """Fit ``reference = scale * predicted + offset`` with Huber IRLS.

    Bounds and priors are part of the method contract: calibration cannot make
    an independently estimated view fit the scene through an unconstrained
    affine transform.
    """

    predicted = torch.as_tensor(predicted_depth, dtype=torch.float64).reshape(-1)
    reference = torch.as_tensor(reference_depth, dtype=torch.float64).reshape(-1)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference depth must have equal shape")
    weights = (
        torch.ones_like(predicted)
        if confidence is None
        else torch.as_tensor(confidence, dtype=torch.float64).reshape(-1)
    )
    if weights.shape != predicted.shape:
        raise ValueError("confidence must match depth samples")
    valid = (
        torch.isfinite(predicted)
        & torch.isfinite(reference)
        & torch.isfinite(weights)
        & (predicted > 0)
        & (reference > 0)
        & (weights > 0)
    )
    predicted, reference, weights = predicted[valid], reference[valid], weights[valid]
    if predicted.numel() < 3:
        return AffineDepthCalibration(1.0, 0.0, int(predicted.numel()), float("nan"), False, "too_few_correspondences")
    design = torch.stack([predicted, torch.ones_like(predicted)], dim=-1)
    solution = torch.tensor([1.0, 0.0], dtype=torch.float64)
    robust = torch.ones_like(weights)
    prior_design = torch.eye(2, dtype=torch.float64)
    prior_target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    prior_weights = torch.tensor([scale_prior_weight, offset_prior_weight], dtype=torch.float64)
    for _ in range(iterations):
        combined_weights = weights * robust
        normal = design.T @ (combined_weights[:, None] * design)
        normal += prior_design.T @ (prior_weights[:, None] * prior_design)
        target = design.T @ (combined_weights * reference)
        target += prior_design.T @ (prior_weights * prior_target)
        solution = torch.linalg.solve(normal, target)
        residual = reference - design @ solution
        scale = 1.4826 * residual.abs().median().clamp_min(1e-8)
        normalized = residual.abs() / (1.345 * scale)
        robust = torch.where(normalized <= 1, torch.ones_like(normalized), normalized.reciprocal())
    residual = reference - design @ solution
    scale_value, offset_value = map(float, solution)
    reference_scale = float(reference.median())
    reason = None
    if abs(scale_value - 1.0) > maximum_scale_deviation:
        reason = "scale_outside_preregistered_bound"
    elif abs(offset_value) > maximum_offset_fraction * reference_scale:
        reason = "offset_outside_preregistered_bound"
    elif scale_value <= 0:
        reason = "non_positive_scale"
    return AffineDepthCalibration(
        scale=scale_value,
        offset=offset_value,
        sample_count=int(predicted.numel()),
        median_absolute_residual=float(residual.abs().median()),
        accepted=reason is None,
        rejection_reason=reason,
    )
