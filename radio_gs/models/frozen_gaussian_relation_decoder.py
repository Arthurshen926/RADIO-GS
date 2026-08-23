"""Shared same-object posterior over pairs of frozen Gaussian latents."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenGaussianRelationDecoder(nn.Module):
    """Predict a symmetric Bernoulli relation without primitive-local state."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 96) -> None:
        super().__init__()
        if min(int(latent_dim), int(hidden_dim)) <= 0:
            raise ValueError("relation decoder dimensions must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * int(hidden_dim) + 4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        left_latent: torch.Tensor,
        right_latent: torch.Tensor,
        left_xyz: torch.Tensor,
        right_xyz: torch.Tensor,
        scene_extent: torch.Tensor,
    ) -> torch.Tensor:
        if (
            left_latent.ndim != 2
            or right_latent.shape != left_latent.shape
            or left_xyz.shape != (left_latent.shape[0], 3)
            or right_xyz.shape != left_xyz.shape
        ):
            raise ValueError("Gaussian relation pair axes differ")
        extent = torch.as_tensor(
            scene_extent, device=left_latent.device, dtype=torch.float32
        ).reshape(-1)
        if extent.shape != (3,) or not bool(torch.isfinite(extent).all()):
            raise ValueError("scene extent must be a finite 3-vector")
        left = F.normalize(self.encoder(left_latent.float()), dim=-1, eps=1e-8)
        right = F.normalize(self.encoder(right_latent.float()), dim=-1, eps=1e-8)
        delta = (left_xyz.float() - right_xyz.float()).abs() / extent.clamp_min(1e-6)
        distance = delta.square().sum(-1, keepdim=True).sqrt()
        features = torch.cat(
            (torch.abs(left - right), left * right, delta, distance), dim=-1
        )
        return self.head(features).squeeze(-1)


__all__ = ["FrozenGaussianRelationDecoder"]
