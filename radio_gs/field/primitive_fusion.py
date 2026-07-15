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
        self.residual_local = bool(residual_local)
        input_dim = self.local_dim + self.coarse_dim + self.reliability_dim
        if input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("fusion dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, self.output_dim), nn.Sigmoid())
        self.base_projection = (
            None
            if self.local_dim == self.output_dim
            else nn.Linear(self.local_dim, self.output_dim)
        )
        # PCA (or its analytical cross-basis projection) is already a strong
        # reconstruction.  Fusion starts as an exact residual-free baseline.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @torch.no_grad()
    def initialize_base_projection(
        self, weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        """Install an analytical local-code -> output-coefficient baseline."""

        if self.base_projection is None:
            if weight.shape != (self.output_dim, self.local_dim):
                raise ValueError("base projection weight has incompatible shape")
            identity = torch.eye(
                self.output_dim, dtype=weight.dtype, device=weight.device
            )
            if not torch.allclose(weight, identity) or not torch.allclose(
                bias, torch.zeros_like(bias)
            ):
                raise ValueError("equal-dimensional fusion baseline must be identity")
            return
        if weight.shape != self.base_projection.weight.shape:
            raise ValueError("base projection weight has incompatible shape")
        if bias.shape != self.base_projection.bias.shape:
            raise ValueError("base projection bias has incompatible shape")
        self.base_projection.weight.copy_(weight.to(self.base_projection.weight))
        self.base_projection.bias.copy_(bias.to(self.base_projection.bias))

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
        base = local if self.base_projection is None else self.base_projection(local)
        return base + update if self.residual_local else update
