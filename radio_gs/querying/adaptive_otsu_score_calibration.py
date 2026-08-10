"""Target-blind calibration of a primitive score field to a frozen threshold.

The frozen LERF protocol thresholds external primitive scores at ``0.6``.
This module estimates a query-wise foreground split from the score field alone
and applies a monotone piecewise-linear map that sends that split to ``0.6``.
Consequently, calibration changes only the operating point; it preserves every
within-query ordering and never reads images, labels, masks, or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from radio_gs.querying.adaptive_support import recursive_upper_otsu_threshold


CONTRACT = "radio_gs.lerf_adaptive_otsu_score_calibration.v1"
FROZEN_THRESHOLD = 0.6


@dataclass(frozen=True)
class AdaptiveOtsuCalibration:
    scores: torch.Tensor
    source_thresholds: torch.Tensor
    selected_counts: torch.Tensor
    stages: int


def _validate(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(scores).detach().float().cpu().contiguous()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"scores must be non-empty [N,Q], got {tuple(values.shape)}")
    if valid.shape != (values.shape[0],) or not bool(valid.any()):
        raise ValueError("valid_mask must be row-aligned and non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores must be finite")
    if bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise ValueError("scores must remain in [0,1]")
    return values, valid


def calibrate_to_frozen_threshold(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    stages: int,
    frozen_threshold: float = FROZEN_THRESHOLD,
) -> AdaptiveOtsuCalibration:
    """Map recursive-upper-Otsu thresholds to one frozen score threshold.

    For each query, ``[0, t, 1]`` maps to ``[0, tau, 1]``, where ``t`` is the
    target-blind Otsu split and ``tau`` is the frozen evaluator threshold.  The
    transform is strictly monotone on either side of ``t`` and guarantees
    ``calibrated > tau`` iff ``original > t`` (up to exact threshold ties).
    """

    values, valid = _validate(scores, valid_mask)
    if int(stages) <= 0:
        raise ValueError("stages must be positive")
    tau = float(frozen_threshold)
    if not 0.0 < tau < 1.0:
        raise ValueError("frozen_threshold must lie strictly inside (0,1)")

    thresholds = torch.tensor(
        [
            recursive_upper_otsu_threshold(values[valid, query], stages=int(stages))
            for query in range(values.shape[1])
        ],
        dtype=torch.float32,
    )
    if bool((thresholds <= 0.0).any()) or bool((thresholds >= 1.0).any()):
        raise ValueError("adaptive thresholds must lie strictly inside (0,1)")

    threshold_rows = thresholds.unsqueeze(0)
    lower = tau * values / threshold_rows
    upper = tau + (1.0 - tau) * (values - threshold_rows) / (1.0 - threshold_rows)
    calibrated = torch.where(values <= threshold_rows, lower, upper).clamp(0.0, 1.0)
    calibrated[~valid] = 0.0
    selected = (calibrated > tau) & valid[:, None]
    expected = (values > threshold_rows) & valid[:, None]
    if not torch.equal(selected, expected):
        raise RuntimeError("calibration changed the intended adaptive membership")
    return AdaptiveOtsuCalibration(
        scores=calibrated.contiguous(),
        source_thresholds=thresholds.contiguous(),
        selected_counts=selected.sum(dim=0).long().contiguous(),
        stages=int(stages),
    )


__all__ = [
    "AdaptiveOtsuCalibration",
    "CONTRACT",
    "FROZEN_THRESHOLD",
    "calibrate_to_frozen_threshold",
]
