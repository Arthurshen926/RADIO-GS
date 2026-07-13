"""A single query-independent canonical RADIO descriptor per Gaussian."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .basis_decoder import AffineBasisDecoder
from .field_signature import FeatureSpaceSignature
from .primitive_fusion import PrimitiveFusion


class CanonicalGaussianField(nn.Module):
    """Compact coefficient field whose only decoded truth is primitive RADIO."""

    def __init__(
        self,
        num_gaussians: int,
        decoder: AffineBasisDecoder,
        signature: FeatureSpaceSignature,
        *,
        local_dim: int | None = None,
        coarse_dim: int = 0,
        reliability: torch.Tensor | None = None,
        hidden_dim: int = 192,
        use_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")
        self.decoder = decoder
        self.signature = signature
        local_dim = decoder.coefficient_dim if local_dim is None else int(local_dim)
        self.local_codes = nn.Parameter(torch.zeros(self.num_gaussians, local_dim))
        nn.init.normal_(self.local_codes, mean=0.0, std=0.01)
        self.coarse_dim = int(coarse_dim)
        reliability_dim = 0 if reliability is None else int(reliability.shape[1])
        if reliability is None:
            self.register_buffer("reliability", torch.empty(self.num_gaussians, 0))
        else:
            reliability = torch.as_tensor(reliability).float()
            if reliability.ndim != 2 or reliability.shape[0] != self.num_gaussians:
                raise ValueError("reliability must be [num_gaussians,R]")
            self.register_buffer("reliability", reliability)
        self.use_fusion = bool(use_fusion)
        if not self.use_fusion and self.coarse_dim:
            raise ValueError("coarse codes require primitive fusion")
        self.fusion = (
            PrimitiveFusion(
                local_dim=local_dim,
                coarse_dim=self.coarse_dim,
                reliability_dim=reliability_dim,
                output_dim=decoder.coefficient_dim,
                hidden_dim=hidden_dim,
            )
            if self.use_fusion
            else None
        )
        self.coarse_provider: Callable[[torch.Tensor], torch.Tensor] | None = None

    def set_coarse_provider(
        self, provider: Callable[[torch.Tensor], torch.Tensor] | None
    ) -> None:
        """Attach a primitive-position provider; it must return row-aligned codes."""

        self.coarse_provider = provider

    def _indices(self, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return torch.arange(self.num_gaussians, device=self.local_codes.device)
        values = torch.as_tensor(indices, dtype=torch.long, device=self.local_codes.device)
        if values.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.num_gaussians):
            raise IndexError("Gaussian index out of range")
        return values

    def coefficients(
        self,
        indices: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = self._indices(indices)
        local = self.local_codes[rows]
        if self.fusion is None:
            if local.shape[1] != self.decoder.coefficient_dim:
                raise RuntimeError("direct local code dimension must equal coefficient dimension")
            return local
        coarse = None
        if self.coarse_dim:
            if self.coarse_provider is None or positions is None:
                raise RuntimeError("coarse field requires provider and row-aligned positions")
            coarse = self.coarse_provider(positions)
            if coarse.shape != (rows.numel(), self.coarse_dim):
                raise ValueError("coarse provider returned incompatible rows")
        reliability = self.reliability[rows] if self.reliability.shape[1] else None
        return self.fusion(local, coarse, reliability)

    def radio_features(
        self,
        indices: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(self.coefficients(indices, positions=positions))

    def get_features(self) -> torch.Tensor:
        """Renderer-compatible compact coefficient rows."""

        return self.coefficients()

    def primitive_confidence(self) -> torch.Tensor | None:
        """Training-only observation reliability for confidence-aware splatting."""

        if self.reliability.shape[1] == 0:
            return None
        return self.reliability.amax(dim=-1).clamp(0.0, 1.0)
