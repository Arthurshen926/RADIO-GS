"""A single query-independent canonical RADIO descriptor per Gaussian."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .basis_decoder import AffineBasisDecoder
from .field_signature import FeatureSpaceSignature
from .primitive_fusion import PrimitiveFusion
from .spatial_hash import PrimitiveSpatialHash


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
        primitive_positions: torch.Tensor | None = None,
        spatial_hash: Mapping[str, object] | None = None,
        reliability: torch.Tensor | None = None,
        fusion_reliability: bool = True,
        hidden_dim: int = 192,
        fusion_residual_blocks: int = 0,
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
        if self.coarse_dim < 0:
            raise ValueError("coarse_dim cannot be negative")
        self.fusion_reliability = bool(fusion_reliability)
        reliability_dim = (
            0
            if reliability is None or not self.fusion_reliability
            else int(reliability.shape[1])
        )
        if reliability is None:
            self.register_buffer("reliability", torch.empty(self.num_gaussians, 0))
        else:
            reliability = torch.as_tensor(reliability).float()
            if reliability.ndim != 2 or reliability.shape[0] != self.num_gaussians:
                raise ValueError("reliability must be [num_gaussians,R]")
            self.register_buffer("reliability", reliability)
        self.use_fusion = bool(use_fusion)
        self.fusion_residual_blocks = int(fusion_residual_blocks)
        if self.fusion_residual_blocks < 0:
            raise ValueError("fusion_residual_blocks cannot be negative")
        if not self.use_fusion and self.coarse_dim:
            raise ValueError("coarse codes require primitive fusion")
        if not self.use_fusion and self.fusion_residual_blocks:
            raise ValueError("fusion residual blocks require primitive fusion")
        if self.coarse_dim:
            if spatial_hash is None:
                raise ValueError("coarse codes require a checkpointed spatial hash")
            spatial_values = dict(spatial_hash)
            spatial_values["output_dim"] = self.coarse_dim
            self.spatial_encoder = PrimitiveSpatialHash.from_mapping(spatial_values)
            if primitive_positions is None:
                normalized = torch.zeros(self.num_gaussians, 3)
                minimum = torch.zeros(3)
                extent = torch.ones(3)
            else:
                positions = torch.as_tensor(primitive_positions).float()
                if positions.shape != (self.num_gaussians, 3):
                    raise ValueError("primitive_positions must be [num_gaussians,3]")
                minimum = positions.amin(dim=0)
                extent = (positions.amax(dim=0) - minimum).clamp_min(1e-6)
                normalized = ((positions - minimum) / extent).clamp(0.0, 1.0)
            # FP16 geometry codes cost only three scalars per primitive.  They
            # make primitive/map reads self-contained and are included in all
            # reported checkpoint storage numbers.
            self.register_buffer("normalized_positions", normalized.half())
            self.register_buffer("position_minimum", minimum)
            self.register_buffer("position_extent", extent)
        else:
            if spatial_hash is not None:
                raise ValueError("spatial_hash requires coarse_dim > 0")
            self.spatial_encoder = None
        self.fusion = (
            PrimitiveFusion(
                local_dim=local_dim,
                coarse_dim=self.coarse_dim,
                reliability_dim=reliability_dim,
                output_dim=decoder.coefficient_dim,
                hidden_dim=hidden_dim,
                residual_blocks=self.fusion_residual_blocks,
            )
            if self.use_fusion
            else None
        )
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
    ) -> torch.Tensor:
        rows = self._indices(indices)
        local = self.local_codes[rows]
        if self.fusion is None:
            if local.shape[1] != self.decoder.coefficient_dim:
                raise RuntimeError("direct local code dimension must equal coefficient dimension")
            return local
        coarse = None
        if self.coarse_dim:
            if self.spatial_encoder is None:
                raise RuntimeError("coarse field lacks its checkpointed spatial encoder")
            coarse = self.spatial_encoder(self.normalized_positions[rows])
            if coarse.shape != (rows.numel(), self.coarse_dim):
                raise ValueError("spatial encoder returned incompatible rows")
        reliability = (
            self.reliability[rows]
            if self.fusion_reliability and self.reliability.shape[1]
            else None
        )
        return self.fusion(local, coarse, reliability)

    def radio_features(
        self,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(self.coefficients(indices))

    def get_features(self) -> torch.Tensor:
        """Renderer-compatible compact coefficient rows."""

        return self.coefficients()

    def primitive_confidence(self) -> torch.Tensor | None:
        """Training-only joint reliability for confidence-aware splatting.

        Reliability channels are conjunctive evidence (coverage, agreement,
        etc.), so a geometric mean is appropriate.  ``amax`` allowed one good
        channel to hide a failed channel and systematically over-trusted
        boundary/occlusion primitives.
        """

        if self.reliability.shape[1] == 0:
            return None
        values = self.reliability.clamp(0.0, 1.0)
        confidence = values.clamp_min(1e-6).log().mean(dim=-1).exp()
        return confidence.masked_fill((values <= 0).any(dim=-1), 0.0)
