"""Source-only authorization for anchor-preserving completion residuals.

Completion proposals are interventions: depending on the query and scene they
may either expand or contract an analytic unary, and those two directions need
not have the same causal value.  This module estimates one coefficient per
direction from complete source-footprint holdouts only.  It minimizes weighted
soft-label log loss and takes the minimum leave-one-fold coefficient, so a
direction unsupported by any source counterfactual is rejected exactly.

The fitted coefficients are passed to the shared anchor-preserving transport
as completion confidence.  Fully observed and inactive rows therefore remain
bitwise identical to the analytic anchor regardless of the fitted values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .anchor_preserving_transport import (
    AnchorPreservingTransportOutput,
    apply_anchor_preserving_probability_proposal,
)


@dataclass(frozen=True)
class DirectionalAdmissionCalibration:
    """Conservative source-OOF coefficients for proposal expansion/contraction."""

    expansion: float
    contraction: float
    leave_one_fold_expansion: tuple[float, ...]
    leave_one_fold_contraction: tuple[float, ...]
    folds: tuple[int, ...]
    eligible_rows: int


def method_contract(*, max_abs_logit_residual: float = 4.0) -> dict[str, object]:
    """Return the benchmark-independent scientific contract."""

    maximum = float(max_abs_logit_residual)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_abs_logit_residual must be finite and positive")
    return {
        "schema_version": 1,
        "method": "source_oof_directional_transport_admission_v1",
        "counterfactual_unit": "complete_source_observation_footprint",
        "objective": "responsibility_weighted_soft_label_log_loss",
        "directions": ["proposal_expansion", "proposal_contraction"],
        "direction_definition": "sign_of_bounded_proposal_minus_anchor_logit",
        "coefficient_domain": [0.0, 1.0],
        "coefficient_solver": "exact_endpoint_test_then_monotone_bisection",
        "fold_aggregation": "minimum_leave_one_fold_coefficient",
        "unsupported_direction_policy": "exact_anchor_identity",
        "transport": "anchor_preserving_confidence_gated_logit_residual_v1",
        "max_abs_logit_residual": maximum,
        "uses_target_rgb_mask_or_metric": False,
        "scene_specific_hyperparameter": False,
        "connected_selection": False,
    }


def _probability(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().double().cpu().reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a nonempty finite vector")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError(f"{name} must lie in [0,1]")
    return tensor


def _bounded_logit_delta(
    anchor: torch.Tensor,
    proposal: torch.Tensor,
    *,
    max_abs_logit_residual: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = float(max_abs_logit_residual)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_abs_logit_residual must be finite and positive")
    if not math.isfinite(float(eps)) or not 0 < float(eps) < 0.5:
        raise ValueError("eps must be finite in (0,0.5)")
    safe_anchor = anchor.clamp(float(eps), 1.0 - float(eps))
    safe_proposal = proposal.clamp(float(eps), 1.0 - float(eps))
    anchor_logit = torch.logit(safe_anchor)
    delta = (torch.logit(safe_proposal) - anchor_logit).clamp(-maximum, maximum)
    return anchor_logit, delta


def _fit_direction_coefficient(
    anchor_logit: torch.Tensor,
    delta: torch.Tensor,
    soft_target: torch.Tensor,
    weight: torch.Tensor,
    population: torch.Tensor,
    *,
    expansion: bool,
) -> float:
    direction = delta > 0 if expansion else delta < 0
    selected = population & direction & (weight > 0)
    if not bool(selected.any()):
        return 0.0
    z0 = anchor_logit[selected]
    step = delta[selected]
    target = soft_target[selected]
    mass = weight[selected]

    def derivative(coefficient: float) -> float:
        probability = torch.sigmoid(z0 + float(coefficient) * step)
        return float((mass * step * (probability - target)).sum())

    # Weighted Bernoulli log loss is convex along this one-dimensional logit
    # path.  Endpoint derivatives therefore identify clipped optima exactly.
    if derivative(0.0) >= 0.0:
        return 0.0
    if derivative(1.0) <= 0.0:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        if derivative(middle) <= 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def fit_conservative_directional_admission(
    anchor_probability: torch.Tensor,
    proposal_probability: torch.Tensor,
    positive_weight: torch.Tensor,
    negative_weight: torch.Tensor,
    eligible: torch.Tensor,
    fold_ids: torch.Tensor,
    *,
    max_abs_logit_residual: float = 4.0,
    eps: float = 1e-6,
) -> DirectionalAdmissionCalibration:
    """Fit a worst-fold coefficient for each intervention direction.

    ``positive_weight`` and ``negative_weight`` are source-raster
    responsibilities, so their normalized ratio is a soft foreground target
    and their sum is the exact supervision mass.  Each coefficient is fitted
    while withholding one complete footprint fold; deployment uses the lower
    envelope across those fits rather than a mean or a tunable percentile.
    """

    anchor = _probability(anchor_probability, name="anchor_probability")
    proposal = _probability(proposal_probability, name="proposal_probability")
    positive = torch.as_tensor(positive_weight).detach().double().cpu().reshape(-1)
    negative = torch.as_tensor(negative_weight).detach().double().cpu().reshape(-1)
    use = torch.as_tensor(eligible).detach().bool().cpu().reshape(-1)
    folds_raw = torch.as_tensor(fold_ids).detach().long().cpu().reshape(-1)
    if any(
        value.shape != anchor.shape
        for value in (proposal, positive, negative, use, folds_raw)
    ):
        raise ValueError("source-OOF admission inputs must align")
    if not bool(torch.isfinite(positive).all()) or not bool(
        torch.isfinite(negative).all()
    ) or bool((positive < 0).any()) or bool((negative < 0).any()):
        raise ValueError("source responsibility weights must be finite and nonnegative")
    weight = positive + negative
    use = use & (weight > 0)
    if not bool(use.any()):
        raise ValueError("source-OOF admission has no eligible responsibility mass")
    fold_values = tuple(int(value) for value in torch.unique(folds_raw[use]).tolist())
    if len(fold_values) < 3:
        raise ValueError("source-OOF admission requires at least three folds")
    if any(value < 0 for value in fold_values):
        raise ValueError("source-OOF fold ids must be nonnegative")
    target = torch.where(weight > 0, positive / weight.clamp_min(1e-15), 0.0)
    anchor_logit, delta = _bounded_logit_delta(
        anchor,
        proposal,
        max_abs_logit_residual=max_abs_logit_residual,
        eps=eps,
    )
    expansion: list[float] = []
    contraction: list[float] = []
    for heldout_fold in fold_values:
        training = use & (folds_raw != heldout_fold)
        if not bool(training.any()):
            raise ValueError("leave-one-fold admission training population is empty")
        expansion.append(
            _fit_direction_coefficient(
                anchor_logit,
                delta,
                target,
                weight,
                training,
                expansion=True,
            )
        )
        contraction.append(
            _fit_direction_coefficient(
                anchor_logit,
                delta,
                target,
                weight,
                training,
                expansion=False,
            )
        )
    return DirectionalAdmissionCalibration(
        expansion=float(min(expansion)),
        contraction=float(min(contraction)),
        leave_one_fold_expansion=tuple(expansion),
        leave_one_fold_contraction=tuple(contraction),
        folds=fold_values,
        eligible_rows=int(use.sum()),
    )


def directional_completion_confidence(
    anchor_probability: torch.Tensor,
    proposal_probability: torch.Tensor,
    calibration: DirectionalAdmissionCalibration,
    *,
    max_abs_logit_residual: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map proposal direction to its conservative source-OOF authorization."""

    anchor = _probability(anchor_probability, name="anchor_probability")
    proposal = _probability(proposal_probability, name="proposal_probability")
    if proposal.shape != anchor.shape:
        raise ValueError("anchor and proposal probability must align")
    _, delta = _bounded_logit_delta(
        anchor,
        proposal,
        max_abs_logit_residual=max_abs_logit_residual,
        eps=eps,
    )
    confidence = torch.where(
        delta > 0,
        torch.full_like(delta, float(calibration.expansion)),
        torch.where(
            delta < 0,
            torch.full_like(delta, float(calibration.contraction)),
            torch.zeros_like(delta),
        ),
    )
    return confidence.float().contiguous()


def apply_source_oof_directional_admission(
    anchor_probability: torch.Tensor,
    proposal_probability: torch.Tensor,
    observation_confidence: torch.Tensor,
    calibration: DirectionalAdmissionCalibration,
    *,
    active_domain: torch.Tensor | None = None,
    max_abs_logit_residual: float = 4.0,
    fully_observed_tolerance: float = 1e-5,
    eps: float = 1e-6,
) -> AnchorPreservingTransportOutput:
    """Apply the calibrated directional budget through the shared transport."""

    completion = directional_completion_confidence(
        anchor_probability,
        proposal_probability,
        calibration,
        max_abs_logit_residual=max_abs_logit_residual,
        eps=eps,
    )
    return apply_anchor_preserving_probability_proposal(
        anchor_probability,
        proposal_probability,
        observation_confidence,
        completion_confidence=completion,
        active_domain=active_domain,
        max_abs_logit_residual=max_abs_logit_residual,
        fully_observed_tolerance=fully_observed_tolerance,
        eps=eps,
    )


__all__ = [
    "DirectionalAdmissionCalibration",
    "apply_source_oof_directional_admission",
    "directional_completion_confidence",
    "fit_conservative_directional_admission",
    "method_contract",
]
