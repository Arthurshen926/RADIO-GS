"""Query-peak-anchored conservative extent posterior for raster supports.

The operator is deliberately downstream of a frozen semantic score posterior:
it may remove disconnected support, but it cannot create support, move the
semantic peak, update a scene field, or inspect RGB/ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


ExtentDomain = Literal["dense_raster", "projected_primitive_alpha"]


@dataclass(frozen=True)
class PeakAnchoredExtentPolicy:
    domain: ExtentDomain
    minimum_retained_fraction: float
    connectivity: int = 8
    peak_outside_support: str = "nearest_foreground"

    def __post_init__(self) -> None:
        if self.connectivity != 8:
            raise ValueError("peak-anchored extent currently freezes 8-connectivity")
        if not 0.0 <= float(self.minimum_retained_fraction) <= 1.0:
            raise ValueError("minimum_retained_fraction must be in [0,1]")
        if self.peak_outside_support != "nearest_foreground":
            raise ValueError("unsupported peak-outside-support policy")


@dataclass(frozen=True)
class ExtentPosterior:
    mask: np.ndarray
    peak_yx: tuple[int, int]
    original_pixels: int
    candidate_pixels: int
    retained_fraction: float
    candidate_accepted: bool
    policy: PeakAnchoredExtentPolicy

    def receipt(self) -> dict[str, object]:
        return {
            "operator": "peak_anchored_conservative_extent_v1",
            "domain": self.policy.domain,
            "connectivity": self.policy.connectivity,
            "peak_outside_support": self.policy.peak_outside_support,
            "minimum_retained_fraction": float(
                self.policy.minimum_retained_fraction
            ),
            "peak_yx": list(self.peak_yx),
            "original_pixels": self.original_pixels,
            "candidate_pixels": self.candidate_pixels,
            "retained_fraction": self.retained_fraction,
            "candidate_accepted": self.candidate_accepted,
            "rgb_opened": False,
            "ground_truth_opened": False,
            "persistent_state_updated": False,
        }


def apply_peak_anchored_extent(
    support: np.ndarray,
    peak_yx: tuple[int, int],
    *,
    policy: PeakAnchoredExtentPolicy,
) -> ExtentPosterior:
    """Return a subset-only extent with a conservative identity fallback."""

    original = np.asarray(support).astype(bool)
    if original.ndim != 2:
        raise ValueError(f"Expected 2D support, got {original.shape}")
    original_pixels = int(original.sum())
    if original_pixels == 0:
        peak = (0, 0)
        candidate = original.copy()
    else:
        peak = (
            min(max(int(peak_yx[0]), 0), original.shape[0] - 1),
            min(max(int(peak_yx[1]), 0), original.shape[1] - 1),
        )
        number, labels = cv2.connectedComponents(
            original.astype(np.uint8), connectivity=policy.connectivity
        )
        if number <= 1:
            candidate = original.copy()
        else:
            label = int(labels[peak])
            if label <= 0:
                ys, xs = np.nonzero(original)
                nearest = int(np.argmin((ys - peak[0]) ** 2 + (xs - peak[1]) ** 2))
                label = int(labels[int(ys[nearest]), int(xs[nearest])])
            candidate = labels == label if label > 0 else original.copy()
    candidate_pixels = int(candidate.sum())
    retained = (
        float(candidate_pixels) / float(original_pixels)
        if original_pixels
        else 1.0
    )
    accepted = retained >= float(policy.minimum_retained_fraction)
    mask = candidate if accepted else original.copy()
    return ExtentPosterior(
        mask=mask,
        peak_yx=peak,
        original_pixels=original_pixels,
        candidate_pixels=candidate_pixels,
        retained_fraction=retained,
        candidate_accepted=accepted,
        policy=policy,
    )


__all__ = [
    "ExtentPosterior",
    "PeakAnchoredExtentPolicy",
    "apply_peak_anchored_extent",
]
