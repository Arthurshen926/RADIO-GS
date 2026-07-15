"""Small zero-mean view residual layered over an invariant canonical field."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


class ZeroMeanViewResidual(nn.Module):
    """Low-rank coefficient residual centered per primitive over training views.

    For primitive ``i`` and view direction ``v`` the residual is

    ``((u_i * ((v - mean_i) @ A)) @ B) * gate_i * scale``.

    Because the mapping after ``v - mean_i`` is linear, its weighted mean over
    the MPR training observations is exactly zero.  The canonical coefficient
    and every primitive-domain query therefore remain view invariant.
    """

    def __init__(
        self,
        num_gaussians: int,
        coefficient_dim: int,
        rank: int,
        mean_view_direction: torch.Tensor,
        *,
        row_gate: torch.Tensor | None = None,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.coefficient_dim = int(coefficient_dim)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        if min(self.num_gaussians, self.coefficient_dim, self.rank) <= 0:
            raise ValueError("view-residual dimensions must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        mean = torch.as_tensor(mean_view_direction).float()
        if mean.shape != (self.num_gaussians, 3):
            raise ValueError("mean_view_direction must be [num_gaussians,3]")
        gate = (
            torch.ones(self.num_gaussians)
            if row_gate is None
            else torch.as_tensor(row_gate).float()
        )
        if gate.shape != (self.num_gaussians,) or bool((gate < 0).any()):
            raise ValueError("row_gate must be non-negative [num_gaussians]")
        self.register_buffer("mean_view_direction", mean)
        self.register_buffer("row_gate", gate)
        self.local_codes = nn.Parameter(torch.zeros(self.num_gaussians, self.rank))
        self.direction_projection = nn.Parameter(torch.empty(3, self.rank))
        self.output_basis = nn.Parameter(torch.empty(self.rank, self.coefficient_dim))
        nn.init.orthogonal_(self.direction_projection)
        nn.init.orthogonal_(self.output_basis)

    @staticmethod
    def camera_center_from_w2c(viewmat: torch.Tensor) -> torch.Tensor:
        view = torch.as_tensor(viewmat).float()
        if view.shape != (4, 4):
            raise ValueError("viewmat must be [4,4]")
        return torch.linalg.inv(view)[:3, 3]

    def delta_from_directions(
        self,
        directions: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        direction = torch.as_tensor(
            directions, device=self.local_codes.device, dtype=self.local_codes.dtype
        )
        if indices is None:
            rows = torch.arange(self.num_gaussians, device=self.local_codes.device)
        else:
            rows = torch.as_tensor(indices, device=self.local_codes.device).long()
        if direction.shape != (rows.numel(), 3):
            raise ValueError("directions must be row-aligned [M,3]")
        centered = direction - self.mean_view_direction[rows]
        direction_latent = centered @ self.direction_projection
        latent = self.local_codes[rows] * direction_latent
        return (
            latent @ self.output_basis
        ) * self.row_gate[rows, None] * self.residual_scale

    def forward(
        self,
        positions: torch.Tensor,
        viewmat: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xyz = torch.as_tensor(
            positions, device=self.local_codes.device, dtype=self.local_codes.dtype
        )
        if indices is not None:
            rows = torch.as_tensor(indices, device=xyz.device).long()
            if xyz.shape == (self.num_gaussians, 3):
                xyz = xyz[rows]
        else:
            rows = None
        center = self.camera_center_from_w2c(viewmat).to(xyz)
        directions = F.normalize(center[None] - xyz, dim=-1, eps=1e-8)
        return self.delta_from_directions(directions, rows)

    def regularization(self) -> dict[str, torch.Tensor]:
        gram = self.output_basis @ self.output_basis.transpose(0, 1)
        identity = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)
        return {
            "local_l2": self.local_codes.square().mean(),
            "basis_orthogonality": (gram - identity).square().mean(),
        }


def load_view_residual_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ZeroMeanViewResidual, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("not a zero-mean view-residual schema-v1 checkpoint")
    architecture = dict(payload.get("architecture", {}))
    module = ZeroMeanViewResidual(
        num_gaussians=int(architecture["num_gaussians"]),
        coefficient_dim=int(architecture["coefficient_dim"]),
        rank=int(architecture["rank"]),
        mean_view_direction=payload["mean_view_direction"],
        row_gate=payload["row_gate"],
        residual_scale=float(architecture["residual_scale"]),
    )
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, payload

