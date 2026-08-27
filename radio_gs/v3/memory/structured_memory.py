"""Global functional projections over the sole persistent D512 latent."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StructuredMemoryHeads(nn.Module):
    """Visual, scale-conditioned instance, and boundary views of one latent."""

    def __init__(
        self,
        latent_dim: int = 512,
        visual_dim: int = 256,
        instance_dim: int = 32,
        boundary_dim: int = 16,
        scale_frequencies: int = 4,
    ) -> None:
        super().__init__()
        if min(latent_dim, visual_dim, instance_dim, boundary_dim, scale_frequencies) <= 0:
            raise ValueError("structured memory dimensions must be positive")
        self.visual = nn.Linear(latent_dim, visual_dim, bias=False)
        self.instance = nn.Linear(latent_dim, instance_dim, bias=False)
        self.boundary = nn.Linear(latent_dim, boundary_dim, bias=False)
        self.scale_adapter = nn.Sequential(
            nn.Linear(2 * scale_frequencies, 2 * instance_dim),
            nn.GELU(),
            nn.Linear(2 * instance_dim, 2 * instance_dim),
        )
        self.scale_frequencies = int(scale_frequencies)

    def visual_view(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(_validate_latent(latent)), dim=-1, eps=1e-8)

    def instance_view(self, latent: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        z = _validate_latent(latent)
        scale_tensor = torch.as_tensor(scale, device=z.device, dtype=z.dtype).reshape(-1)
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.expand(z.shape[0])
        if scale_tensor.shape != (z.shape[0],) or not bool(torch.isfinite(scale_tensor).all()):
            raise ValueError("mask scale must be finite and scalar or per Gaussian")
        frequency = torch.arange(1, self.scale_frequencies + 1, device=z.device, dtype=z.dtype)
        phase = math.pi * scale_tensor[:, None].clamp(0, 1) * frequency[None]
        gamma, beta = self.scale_adapter(torch.cat((phase.sin(), phase.cos()), dim=-1)).chunk(2, dim=-1)
        value = self.instance(z) * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def boundary_view(self, latent: torch.Tensor) -> torch.Tensor:
        return self.boundary(_validate_latent(latent))

    def orthogonality_loss(self) -> torch.Tensor:
        visual = F.normalize(self.visual.weight, dim=1)
        instance = F.normalize(self.instance.weight, dim=1)
        boundary = F.normalize(self.boundary.weight, dim=1)
        return (visual @ instance.T).square().mean() + (visual @ boundary.T).square().mean()


class ExtraInstanceCodeOracle(nn.Module):
    """Temporary Gaussian-indexed D16 upper bound; forbidden at deployment."""

    deployment_eligible = False

    def __init__(self, num_gaussians: int, instance_dim: int = 16, seed: int = 20260826) -> None:
        super().__init__()
        if num_gaussians <= 0 or instance_dim <= 0:
            raise ValueError("oracle dimensions must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(num_gaussians, instance_dim, generator=generator) / instance_dim**0.5
        self.code = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.code, dim=-1, eps=1e-8)


def _validate_latent(latent: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(latent)
    if value.ndim != 2 or value.shape[1] != 512 or not bool(torch.isfinite(value).all()):
        raise ValueError("structured memory input must be finite [N,512]")
    return value.float()


__all__ = ["ExtraInstanceCodeOracle", "StructuredMemoryHeads"]
