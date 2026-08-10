"""Opt-in RegionCoMembership candidate with factorized-native relations."""

from __future__ import annotations

import torch
from torch import nn

from radio_gs.interfaces.factorized_native_region_relation import (
    FEATURE_NAMES as NATIVE_PAIR_FEATURE_NAMES,
)
from radio_gs.models.region_comembership_v2 import (
    HIDDEN_DIMENSIONS,
    PAIR_FEATURE_NAMES as V2_PAIR_FEATURE_NAMES,
)


PAIR_FEATURE_NAMES = V2_PAIR_FEATURE_NAMES + NATIVE_PAIR_FEATURE_NAMES


class RegionCoMembershipNativeV3(nn.Module):
    """The frozen V2 MLP width with nine appended native pair channels.

    The final layer is zero initialized, preserving the exact 0.5 epoch-zero
    candidate.  This class is independent of and never substituted for V2.
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
            raise ValueError("native V3 co-membership normalization differs")
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
            raise AssertionError("native V3 final relation layer differs")
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
                f"native V3 pair features must be finite float [P,{len(PAIR_FEATURE_NAMES)}]"
            )
        normalized = (values - self.feature_median) / self.feature_robust_scale
        return self.network(normalized).squeeze(-1)

    def probability(self, pair_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(pair_features))


__all__ = [
    "NATIVE_PAIR_FEATURE_NAMES",
    "PAIR_FEATURE_NAMES",
    "V2_PAIR_FEATURE_NAMES",
    "RegionCoMembershipNativeV3",
]
