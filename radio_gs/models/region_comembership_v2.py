"""Capability-conditioned, query-independent canonical-region relation head."""

from __future__ import annotations

import torch
from torch import nn

from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES as V1_PAIR_FEATURE_NAMES,
)


CAPABILITY_PAIR_FEATURE_NAMES = (
    "appearance_direction_cosine",
    "boundary_direction_cosine",
    "minimum_appearance_concentration",
    "minimum_boundary_concentration",
    "absolute_appearance_concentration_difference",
    "absolute_boundary_concentration_difference",
)
PAIR_FEATURE_NAMES = V1_PAIR_FEATURE_NAMES + CAPABILITY_PAIR_FEATURE_NAMES
HIDDEN_DIMENSIONS = (64, 32)


class RegionCoMembershipV2(nn.Module):
    """A small nonlinear relation head over symmetric pair evidence.

    The final affine layer is exactly zero initialized.  Thus epoch zero emits
    probability 0.5 for every pair and remains a deterministic singleton
    fallback after source-validation readout selection.
    """

    def __init__(self, median: torch.Tensor, robust_scale: torch.Tensor) -> None:
        super().__init__()
        center = torch.as_tensor(median).detach().float().cpu().contiguous()
        scale = torch.as_tensor(robust_scale).detach().float().cpu().contiguous()
        dimension = len(PAIR_FEATURE_NAMES)
        if (
            center.shape != (dimension,)
            or scale.shape != (dimension,)
            or not bool(torch.isfinite(center).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
        ):
            raise ValueError("V2 co-membership normalization must be finite [21]")
        self.register_buffer("feature_median", center)
        self.register_buffer("feature_robust_scale", scale)
        self.network = nn.Sequential(
            nn.Linear(dimension, HIDDEN_DIMENSIONS[0]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMENSIONS[0], HIDDEN_DIMENSIONS[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMENSIONS[1], 1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("V2 final relation layer differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(pair_features)
        if (
            values.ndim != 2
            or values.shape[1] != len(PAIR_FEATURE_NAMES)
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
            or values.device != self.feature_median.device
            or values.dtype != self.network[0].weight.dtype
        ):
            raise ValueError(
                "V2 co-membership pair features must be finite float [P,21]"
            )
        normalized = (values - self.feature_median) / self.feature_robust_scale
        return self.network(normalized).squeeze(-1)

    def probability(self, pair_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(pair_features))


__all__ = [
    "CAPABILITY_PAIR_FEATURE_NAMES",
    "HIDDEN_DIMENSIONS",
    "PAIR_FEATURE_NAMES",
    "RegionCoMembershipV2",
]
