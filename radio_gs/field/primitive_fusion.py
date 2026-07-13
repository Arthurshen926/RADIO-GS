"""Primitive-local coefficient fusion without screen or batch dependence."""

from __future__ import annotations

import torch
from torch import nn


class PrimitiveFusion(nn.Module):
    """Fuse local, coarse, and reliability codes independently per primitive."""

    def __init__(
        self,
        local_dim: int,
        coarse_dim: int,
        reliability_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 192,
        residual_local: bool = True,
    ) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        self.coarse_dim = int(coarse_dim)
        self.reliability_dim = int(reliability_dim)
        self.output_dim = int(output_dim)
        self.residual_local = bool(residual_local and local_dim == output_dim)
        input_dim = self.local_dim + self.coarse_dim + self.reliability_dim
        if input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("fusion dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, self.output_dim), nn.Sigmoid())
        if self.residual_local:
            # The PCA code is already a strong reconstruction.  Start from an
            # exact identity field and let reliability/coarse evidence learn
            # only a residual correction.
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        local: torch.Tensor,
        coarse: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local.ndim != 2 or local.shape[1] != self.local_dim:
            raise ValueError(f"local must be [N,{self.local_dim}]")
        count = local.shape[0]
        parts = [local]
        for value, dim, name in (
            (coarse, self.coarse_dim, "coarse"),
            (reliability, self.reliability_dim, "reliability"),
        ):
            if dim == 0:
                continue
            if value is None or value.shape != (count, dim):
                raise ValueError(f"{name} must be [N,{dim}]")
            parts.append(value)
        inputs = torch.cat(parts, dim=-1)
        update = self.network(inputs) * self.gate(inputs)
        return local + update if self.residual_local else update
