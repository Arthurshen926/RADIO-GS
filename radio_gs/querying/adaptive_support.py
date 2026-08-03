"""Target-blind support calibration for primitive query score fields.

The functions in this module consume only a row-aligned primitive score field
and its authority-bound valid mask. They deliberately have no access to RGB,
benchmark annotations, cameras, or rendered target masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class AdaptiveSupportCalibrationError(ValueError):
    """Raised when a score distribution cannot define a safe split."""


@dataclass(frozen=True)
class AdaptiveSupportSelection:
    selected: torch.Tensor
    thresholds: torch.Tensor
    selected_counts: torch.Tensor
    valid_count: int
    otsu_stages: int


def exact_otsu_threshold(values: torch.Tensor) -> float:
    """Return the exact two-class Otsu split without histogram bin tuning.

    Candidate splits are the midpoints between adjacent distinct sorted
    values. Ties in between-class variance resolve to the lowest threshold,
    matching ``argmax`` deterministically.
    """

    values_cpu = torch.as_tensor(values).detach().float().cpu().reshape(-1)
    if values_cpu.numel() < 2:
        raise AdaptiveSupportCalibrationError(
            "Otsu calibration requires at least two score observations"
        )
    if not bool(torch.isfinite(values_cpu).all()):
        raise AdaptiveSupportCalibrationError("Otsu scores must be finite")
    ordered = torch.sort(values_cpu).values.double()
    distinct_split = ordered[:-1] < ordered[1:]
    if not bool(distinct_split.any()):
        raise AdaptiveSupportCalibrationError(
            "Otsu calibration requires at least two distinct score values"
        )

    prefix = ordered.cumsum(dim=0)
    count = int(ordered.numel())
    left_count = torch.arange(1, count, dtype=torch.float64)
    right_count = float(count) - left_count
    left_mean = prefix[:-1] / left_count
    right_mean = (prefix[-1] - prefix[:-1]) / right_count
    objective = left_count * right_count * (left_mean - right_mean).square()
    objective[~distinct_split] = -1.0
    split = int(objective.argmax().item())
    return float(((ordered[split] + ordered[split + 1]) * 0.5).item())


def recursive_upper_otsu_threshold(
    values: torch.Tensor,
    *,
    stages: int,
) -> float:
    """Repeatedly separate the upper response class with exact two-class Otsu."""

    if int(stages) < 1:
        raise AdaptiveSupportCalibrationError("Otsu stages must be positive")
    active = torch.as_tensor(values).detach().float().cpu().reshape(-1)
    threshold = float("nan")
    for stage in range(int(stages)):
        try:
            threshold = exact_otsu_threshold(active)
        except AdaptiveSupportCalibrationError as exc:
            raise AdaptiveSupportCalibrationError(
                f"Otsu stage {stage + 1}/{int(stages)} is degenerate: {exc}"
            ) from exc
        active = active[active > threshold]
    return threshold


def select_adaptive_otsu_support(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    otsu_stages: int = 1,
) -> AdaptiveSupportSelection:
    """Select each query's valid primitive support using target-blind Otsu."""

    values = torch.as_tensor(scores).detach().float().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise AdaptiveSupportCalibrationError(
            f"scores must be a non-empty [N,Q] matrix, got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all()):
        raise AdaptiveSupportCalibrationError("scores must be finite")
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if valid.shape != (int(values.shape[0]),) or not bool(valid.any()):
        raise AdaptiveSupportCalibrationError(
            "valid_mask must be row-aligned and keep at least one primitive"
        )

    thresholds = []
    for query_index in range(int(values.shape[1])):
        thresholds.append(
            recursive_upper_otsu_threshold(
                values[valid, query_index],
                stages=int(otsu_stages),
            )
        )
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)
    selected_bool = (values > threshold_tensor.unsqueeze(0)) & valid.unsqueeze(1)
    selected_counts = selected_bool.sum(dim=0)
    if not bool((selected_counts > 0).all()):
        raise AdaptiveSupportCalibrationError(
            "adaptive support must keep at least one valid primitive per query"
        )
    return AdaptiveSupportSelection(
        selected=selected_bool.float(),
        thresholds=threshold_tensor,
        selected_counts=selected_counts,
        valid_count=int(valid.sum().item()),
        otsu_stages=int(otsu_stages),
    )
