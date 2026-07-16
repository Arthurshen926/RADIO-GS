"""Tiny field-independent residual conditioned only on rendered discontinuities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


class BoundaryConditionedScreenResidual(nn.Module):
    """Predict a low-rank RADIO correction from RGB/depth/alpha gradients.

    The module has no primitive rows, scene embedding, query input, or camera
    identity.  It is applied only after canonical rendering, so every direct
    primitive query is exactly unchanged.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        rank: int = 8,
        hidden_dim: int = 16,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if min(feature_dim, rank, hidden_dim) <= 0 or residual_scale <= 0:
            raise ValueError("boundary residual dimensions/scale must be positive")
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.conditioner = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, rank, 1),
        )
        self.output_basis = nn.Parameter(torch.empty(rank, feature_dim))
        nn.init.orthogonal_(self.output_basis)
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    @staticmethod
    def _gradient_magnitude(values: torch.Tensor) -> torch.Tensor:
        dx = F.pad(values[..., 1:] - values[..., :-1], (0, 1, 0, 0))
        dy = F.pad(values[..., 1:, :] - values[..., :-1, :], (0, 0, 0, 1))
        return (dx.square() + dy.square() + 1e-12).sqrt()

    def conditions(
        self, rgb: torch.Tensor, depth: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        if rgb.ndim == 3:
            rgb = rgb[None]
        if depth.ndim == 2:
            depth = depth[None, None]
        elif depth.ndim == 3:
            depth = depth[:, None]
        if alpha.ndim == 2:
            alpha = alpha[None, None]
        elif alpha.ndim == 3:
            alpha = alpha[:, None]
        if rgb.ndim != 4 or rgb.shape[1] != 3 or depth.shape != alpha.shape:
            raise ValueError("rgb/depth/alpha must align as [B,3,H,W]/[B,1,H,W]")
        luminance = (
            0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
        )
        rgb_edge = self._gradient_magnitude(luminance)
        relative_depth = depth / depth.flatten(2).median(dim=-1).values[..., None].clamp_min(1e-6)
        depth_edge = self._gradient_magnitude(relative_depth).clamp(max=2.0) / 2.0
        alpha_edge = self._gradient_magnitude(alpha).clamp(max=1.0)
        return torch.cat([rgb_edge.clamp(max=1.0), depth_edge, alpha_edge], dim=1)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        condition = self.conditions(rgb, depth, alpha)
        # Structural support constraint: even after training, a bias cannot turn
        # this into an unconstrained full-screen correction.  The residual is
        # exactly zero wherever all observable discontinuity channels are zero.
        boundary_gate = condition.amax(dim=1, keepdim=True)
        latent = self.conditioner(condition) * boundary_gate
        delta = torch.einsum("brhw,rc->bchw", latent, self.output_basis)
        return delta * self.residual_scale

    def regularization(self) -> dict[str, torch.Tensor]:
        gram = self.output_basis @ self.output_basis.transpose(0, 1)
        identity = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)
        return {
            "delta_head_l2": self.conditioner[-1].weight.square().mean(),
            "basis_orthogonality": (gram - identity).square().mean(),
        }


def load_boundary_screen_residual_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[BoundaryConditionedScreenResidual, Mapping[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("not a boundary-screen-residual schema-v1 checkpoint")
    architecture = dict(payload["architecture"])
    module = BoundaryConditionedScreenResidual(**architecture)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, payload
