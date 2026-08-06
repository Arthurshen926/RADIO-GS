"""Reconstruction loss for the versioned factorized RADIO representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from radio_gs.field.factorized_radio_contract import (
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    reliability_scalar_names_sha256,
)


FACTORIZED_RADIO_RECONSTRUCTION_LOSS_CONTRACT = "canonical-factorized-radio-loss-v1"
FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY = "legacy_evidence_resultant"
FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE = (
    "matched_exact_marginal_visibility_safe_v1"
)
FACTORIZED_RADIO_EXACT_MARGINAL_VISIBILITY_SAFE_LOSS_CONTRACT = (
    "canonical-factorized-radio-loss-matched-exact-marginal-visibility-safe-v1"
)


@dataclass(frozen=True)
class FactorizedRadioLoss:
    total: torch.Tensor
    direction: torch.Tensor
    log_amplitude: torch.Tensor
    valid_rows: int
    mean_direction_weight: float
    mean_amplitude_weight: float


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("factorized loss values and weights must be aligned vectors")
    denominator = weights.sum()
    if not bool(denominator.detach() > 0):
        raise ValueError("factorized loss has no positive row weight")
    return (values * weights).sum() / denominator


def uniform_half_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Keep ambiguous source rows trainable while retaining confidence ordering."""

    values = torch.as_tensor(confidence)
    if not values.dtype.is_floating_point or not bool(torch.isfinite(values).all()):
        raise ValueError("uniform-half confidence must be finite floating point")
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("uniform-half confidence must lie in [0,1]")
    return 0.5 * (torch.ones_like(values) + values)


def factorized_radio_reconstruction_loss(
    predicted: torch.Tensor,
    target_canonical: torch.Tensor,
    target_log_amplitude: torch.Tensor,
    valid: torch.Tensor,
    reliability: torch.Tensor,
    *,
    reliability_scalar_names: Sequence[str],
    reliability_scalar_names_digest: str,
    reliability_policy: str = FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
    amplitude_weight: float = 0.25,
    norm_epsilon: float = 1e-8,
) -> FactorizedRadioLoss:
    """Measure semantic direction and raw RADIO gauge without conflating them."""

    names = tuple(str(name) for name in reliability_scalar_names)
    if names != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES or (
        str(reliability_scalar_names_digest)
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        or reliability_scalar_names_sha256(names)
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    ):
        raise ValueError("factorized RADIO loss reliability scalar schema differs")
    if predicted.ndim != 2 or min(predicted.shape) <= 0:
        raise ValueError("predicted factorized RADIO must be non-empty [B,D]")
    if target_canonical.shape != predicted.shape:
        raise ValueError("factorized RADIO prediction and target shapes differ")
    rows = predicted.shape[0]
    if target_log_amplitude.shape != (rows,) or valid.shape != (rows,):
        raise ValueError("factorized RADIO amplitude/valid shapes differ")
    if reliability.shape != (
        rows,
        len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
    ):
        raise ValueError("factorized RADIO reliability shape differs")
    if valid.dtype != torch.bool:
        raise TypeError("factorized RADIO valid mask must be boolean")
    if not all(
        value.dtype.is_floating_point
        for value in (predicted, target_canonical, target_log_amplitude, reliability)
    ):
        raise TypeError("factorized RADIO loss tensors must be floating point")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (predicted, target_canonical, target_log_amplitude, reliability)
    ):
        raise ValueError("factorized RADIO loss tensors must be finite")
    if not bool(valid.any()):
        raise ValueError("factorized RADIO loss requires at least one valid row")
    if not float(amplitude_weight) >= 0.0 or not float(norm_epsilon) > 0.0:
        raise ValueError("factorized RADIO loss weights/epsilon are invalid")

    active = valid
    target_active = target_canonical[active]
    target_norm = torch.linalg.vector_norm(target_active, dim=-1)
    if bool((target_norm <= 0).any()):
        raise ValueError("valid factorized RADIO targets must be nonzero")
    predicted_active = predicted[active]
    target_direction = target_active / target_norm[:, None]
    predicted_direction = F.normalize(predicted_active, dim=-1, eps=float(norm_epsilon))
    direction_per_row = 1.0 - (predicted_direction * target_direction).sum(
        dim=-1
    ).clamp(-1.0, 1.0)

    active_reliability = reliability[active].float()
    resultant = active_reliability[:, 0]
    log_amplitude_std = active_reliability[:, 2]
    observation_evidence = active_reliability[:, 3]
    visibility_purity = active_reliability[:, 4]
    if (
        bool((resultant <= 0).any())
        or bool((resultant > 1).any())
        or bool((log_amplitude_std < 0).any())
        or bool((observation_evidence <= 0).any())
        or bool((observation_evidence >= 1).any())
    ):
        raise ValueError("factorized RADIO loss reliability values differ")
    direction_weight = observation_evidence * resultant
    amplitude_row_weight = observation_evidence / (1.0 + log_amplitude_std)
    policy = str(reliability_policy)
    if policy == FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY:
        # Preserve canonical-factorized-radio-loss-v1 exactly.  In particular,
        # the top-1 cache's zero purity sentinel remains unknown, not evidence.
        pass
    elif (
        policy
        == FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
    ):
        # Exact marginal purity is independent compositor precision evidence.
        # Its fixed uniform-half mixture preserves at least half of each row's
        # historical weight, so real boundaries/occlusions remain trainable.
        visibility_factor = uniform_half_confidence(visibility_purity)
        direction_weight = direction_weight * visibility_factor
        amplitude_row_weight = amplitude_row_weight * visibility_factor
    else:
        raise ValueError("unsupported factorized RADIO reliability policy")

    predicted_log_amplitude = torch.log(
        torch.sqrt(
            predicted_active.float().square().sum(dim=-1) + float(norm_epsilon) ** 2
        )
    )
    amplitude_per_row = F.smooth_l1_loss(
        predicted_log_amplitude,
        target_log_amplitude[active].float(),
        reduction="none",
    )
    direction_loss = _weighted_mean(direction_per_row.float(), direction_weight)
    amplitude_loss = _weighted_mean(amplitude_per_row, amplitude_row_weight)
    total = direction_loss + float(amplitude_weight) * amplitude_loss
    return FactorizedRadioLoss(
        total=total,
        direction=direction_loss,
        log_amplitude=amplitude_loss,
        valid_rows=int(active.sum().item()),
        mean_direction_weight=float(direction_weight.detach().mean()),
        mean_amplitude_weight=float(amplitude_row_weight.detach().mean()),
    )
