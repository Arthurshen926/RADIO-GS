"""Compact multiresolution spatial codes evaluated once per Gaussian primitive."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


class PrimitiveSpatialHash(nn.Module):
    """Pure-PyTorch hash grid for normalized, row-aligned primitive positions.

    Unlike the legacy screen-space hybrid path, this module is evaluated at a
    Gaussian's fixed 3-D position before rasterization.  Its output therefore
    belongs to the same canonical primitive descriptor read by every query
    interface.
    """

    def __init__(
        self,
        output_dim: int,
        *,
        num_levels: int = 8,
        features_per_level: int = 2,
        log2_hashmap_size: int = 15,
        base_resolution: int = 8,
        max_resolution: int = 512,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.max_resolution = int(max_resolution)
        self.hidden_dim = int(hidden_dim)
        if min(
            self.output_dim,
            self.num_levels,
            self.features_per_level,
            self.base_resolution,
            self.max_resolution,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("spatial hash dimensions and resolutions must be positive")
        if self.log2_hashmap_size <= 0:
            raise ValueError("log2_hashmap_size must be positive")
        if self.max_resolution < self.base_resolution:
            raise ValueError("max_resolution must be at least base_resolution")

        growth = (
            math.exp(
                math.log(self.max_resolution / self.base_resolution)
                / (self.num_levels - 1)
            )
            if self.num_levels > 1
            else 1.0
        )
        resolutions = [
            max(1, int(math.floor(self.base_resolution * growth**level)))
            for level in range(self.num_levels)
        ]
        self.register_buffer(
            "resolutions", torch.tensor(resolutions, dtype=torch.long)
        )
        maximum_table_size = 2**self.log2_hashmap_size
        table_sizes = [
            min(maximum_table_size, int(resolution + 1) ** 3)
            for resolution in resolutions
        ]
        self.hash_tables = nn.ParameterList(
            [
                nn.Parameter(torch.empty(size, self.features_per_level))
                for size in table_sizes
            ]
        )
        for table in self.hash_tables:
            nn.init.normal_(table, mean=0.0, std=0.01)
        self.register_buffer(
            "corner_offsets",
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "hash_primes",
            torch.tensor([1, 2_654_435_761, 805_459_861], dtype=torch.long),
        )
        encoded_dim = self.num_levels * self.features_per_level
        self.mlp = nn.Sequential(
            nn.Linear(encoded_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def architecture(self) -> dict[str, int]:
        return {
            "output_dim": self.output_dim,
            "num_levels": self.num_levels,
            "features_per_level": self.features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "max_resolution": self.max_resolution,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PrimitiveSpatialHash":
        return cls(
            output_dim=int(values["output_dim"]),
            num_levels=int(values["num_levels"]),
            features_per_level=int(values["features_per_level"]),
            log2_hashmap_size=int(values["log2_hashmap_size"]),
            base_resolution=int(values["base_resolution"]),
            max_resolution=int(values["max_resolution"]),
            hidden_dim=int(values["hidden_dim"]),
        )

    def _hash(self, corners: torch.Tensor, table_size: int) -> torch.Tensor:
        hashed = corners[..., 0] * self.hash_primes[0]
        hashed = hashed ^ (corners[..., 1] * self.hash_primes[1])
        hashed = hashed ^ (corners[..., 2] * self.hash_primes[2])
        return torch.remainder(hashed, int(table_size))

    def _level(self, positions: torch.Tensor, level: int) -> torch.Tensor:
        resolution = self.resolutions[level].to(positions.dtype)
        scaled = positions * resolution
        lower = torch.floor(scaled).long()
        fraction = scaled - lower.to(scaled.dtype)
        corners = lower[:, None, :] + self.corner_offsets[None]
        table = self.hash_tables[level]
        values = table[self._hash(corners, table.shape[0])]

        wx = fraction[:, 0:1]
        wy = fraction[:, 1:2]
        wz = fraction[:, 2:3]
        c00 = values[:, 0] * (1.0 - wz) + values[:, 1] * wz
        c01 = values[:, 2] * (1.0 - wz) + values[:, 3] * wz
        c10 = values[:, 4] * (1.0 - wz) + values[:, 5] * wz
        c11 = values[:, 6] * (1.0 - wz) + values[:, 7] * wz
        c0 = c00 * (1.0 - wy) + c01 * wy
        c1 = c10 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wx) + c1 * wx

    def forward(self, normalized_positions: torch.Tensor) -> torch.Tensor:
        positions = torch.as_tensor(
            normalized_positions,
            device=self.hash_tables[0].device,
            dtype=self.hash_tables[0].dtype,
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("normalized_positions must be [N,3]")
        if positions.numel() and not bool(torch.isfinite(positions).all()):
            raise ValueError("normalized_positions contain NaN or infinity")
        # Geometry fingerprints enforce the training geometry.  The clamp only
        # absorbs half-precision storage roundoff at the unit-cube boundary.
        positions = positions.clamp(0.0, 1.0)
        encoded = torch.cat(
            [self._level(positions, level) for level in range(self.num_levels)],
            dim=-1,
        )
        return self.mlp(encoded)
