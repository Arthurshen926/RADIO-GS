"""Pure metrics for the canonical-field render-ceiling audit.

The key distinction is between total rendered alpha and the part of that
alpha whose primitive row has a valid multiview observation.  Rendering the
binary row-valid flag as a scalar colour while retaining every Gaussian gives
the latter under exactly the same transmittance and occlusion ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F


def contribution_coverage(
    valid_contribution_mass: torch.Tensor,
    total_alpha: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the exact valid-row contribution fraction for each pixel."""

    if valid_contribution_mass.shape != total_alpha.shape:
        raise ValueError("valid contribution mass and total alpha must have the same shape")
    coverage = valid_contribution_mass / total_alpha.clamp_min(float(eps))
    return torch.where(total_alpha > float(eps), coverage.clamp(0.0, 1.0), 0.0)


def normalize_premultiplied(
    feature_map: torch.Tensor,
    mass: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize a premultiplied ``[C,H,W]`` map by a scalar mass map.

    Unsupported pixels are set to zero.  Callers must still exclude those
    pixels from metrics because a conditional descriptor is undefined there.
    """

    if feature_map.ndim != 3 or mass.shape != feature_map.shape[1:]:
        raise ValueError("feature_map must be [C,H,W] and mass must be [H,W]")
    supported = mass > float(eps)
    normalized = feature_map / mass.clamp_min(float(eps)).unsqueeze(0)
    return torch.where(supported.unsqueeze(0), normalized, 0.0)


def parse_coverage_edges(raw: str | Iterable[float]) -> tuple[float, ...]:
    """Parse and validate monotonically increasing coverage-bin edges."""

    if isinstance(raw, str):
        edges = tuple(float(value) for value in raw.split(",") if value.strip())
    else:
        edges = tuple(float(value) for value in raw)
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] < 1.0:
        raise ValueError("coverage edges must start at 0 and end at or above 1")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("coverage edges must be strictly increasing")
    return edges


def coverage_bin_masks(
    coverage: torch.Tensor,
    base_mask: torch.Tensor,
    edges: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Build disjoint masks whose union is ``base_mask`` for coverage in [0,1]."""

    if coverage.shape != base_mask.shape:
        raise ValueError("coverage and base_mask must have the same shape")
    result: dict[str, torch.Tensor] = {}
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_last = index == len(edges) - 2
        inside = (coverage >= left) & (
            (coverage <= right) if is_last else (coverage < right)
        )
        result[f"[{left:.2f},{right:.2f}{']' if is_last else ')'}"] = base_mask & inside
    return result


def pixel_reconstruction_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel cosine and channel RMSE for ``[C,H,W]`` maps."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching [C,H,W] maps")
    if mask.shape != prediction.shape[1:]:
        raise ValueError("mask shape must match the spatial feature-map shape")
    predicted_pixels = prediction.permute(1, 2, 0)[mask].float()
    target_pixels = target.permute(1, 2, 0)[mask].float()
    if predicted_pixels.numel() == 0:
        empty = torch.empty(0, device=prediction.device, dtype=torch.float32)
        return empty, empty
    cosine = F.cosine_similarity(predicted_pixels, target_pixels, dim=-1, eps=1e-8)
    rmse = (predicted_pixels - target_pixels).square().mean(dim=-1).sqrt()
    return cosine, rmse


@dataclass
class PixelMetricAccumulator:
    """Small CPU accumulator retaining values for quantiles and correlations."""

    cosine: list[torch.Tensor] = field(default_factory=list)
    rmse: list[torch.Tensor] = field(default_factory=list)
    coverage: list[torch.Tensor] = field(default_factory=list)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        coverage: torch.Tensor | None = None,
    ) -> None:
        cosine, rmse = pixel_reconstruction_values(prediction, target, mask)
        if cosine.numel() == 0:
            return
        self.cosine.append(cosine.detach().cpu())
        self.rmse.append(rmse.detach().cpu())
        if coverage is not None:
            if coverage.shape != mask.shape:
                raise ValueError("coverage shape must match mask")
            self.coverage.append(coverage[mask].detach().float().cpu())

    def summary(self) -> dict[str, float | int | None]:
        if not self.cosine:
            return {
                "pixels": 0,
                "mean_cosine": None,
                "p05_cosine": None,
                "median_cosine": None,
                "mean_rmse": None,
                "coverage_error_pearson": None,
                "coverage_error_spearman": None,
            }
        cosine = torch.cat(self.cosine).float()
        rmse = torch.cat(self.rmse).float()
        result: dict[str, float | int | None] = {
            "pixels": int(cosine.numel()),
            "mean_cosine": float(cosine.mean()),
            "p05_cosine": float(torch.quantile(cosine, 0.05)),
            "median_cosine": float(cosine.median()),
            "mean_rmse": float(rmse.mean()),
            "coverage_error_pearson": None,
            "coverage_error_spearman": None,
        }
        if self.coverage:
            coverage = torch.cat(self.coverage).float()
            error = 1.0 - cosine
            if coverage.numel() == error.numel() and coverage.numel() > 1:
                if float(coverage.std(unbiased=False)) > 0 and float(error.std(unbiased=False)) > 0:
                    result["coverage_error_pearson"] = float(
                        torch.corrcoef(torch.stack([coverage, error]))[0, 1]
                    )
                    try:
                        from scipy.stats import spearmanr

                        result["coverage_error_spearman"] = float(
                            spearmanr(coverage.numpy(), error.numpy()).statistic
                        )
                    except (ImportError, AttributeError, ValueError):
                        result["coverage_error_spearman"] = None
        return result

