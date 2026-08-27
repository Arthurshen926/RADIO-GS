"""Jointly reorganized single-D512 source-mapping arm for SUGM-v3."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class JointD512Arm(nn.Module):
    """Open the sole canonical D512 while keeping its RADIO decoder frozen."""

    deployment_eligible = True

    def __init__(
        self,
        latent: torch.Tensor,
        *,
        radio_basis: torch.Tensor,
        radio_mean: torch.Tensor,
        radio_scale: torch.Tensor,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        base = torch.as_tensor(latent).detach().float()
        basis = torch.as_tensor(radio_basis).detach().float()
        mean = torch.as_tensor(radio_mean).detach().float()
        scale = torch.as_tensor(radio_scale).detach().float()
        if base.ndim != 2 or base.shape[1] != 512:
            raise ValueError("joint source mapping requires exactly one D512")
        if basis.ndim != 2 or basis.shape[1] != 512:
            raise ValueError("RADIO basis must be [F,512]")
        if mean.shape != (basis.shape[0],) or scale.shape != mean.shape:
            raise ValueError("RADIO decoder statistics differ")
        self.register_buffer("base_latent", base.clone(), persistent=False)
        self.register_buffer("radio_basis", basis, persistent=False)
        self.register_buffer("radio_mean", mean, persistent=False)
        self.register_buffer("radio_scale", scale, persistent=False)
        self.latent = nn.Parameter(base.clone())
        self.projection = nn.Linear(512, int(output_dim), bias=False)
        self.scale_adapter = nn.Linear(2, 2 * int(output_dim))

    def coefficients(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        if rows is None:
            return self.latent
        indices = torch.as_tensor(rows, device=self.latent.device, dtype=torch.long)
        return self.latent[indices]

    def projected_latent(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        return self.projection(self.coefficients(rows))

    def scale_embedding(self, projected: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        phase = projected.new_tensor([float(scale)]).clamp(0, 1) * torch.pi
        gamma, beta = self.scale_adapter(
            torch.cat((phase.sin(), phase.cos()))
        ).chunk(2)
        value = projected * (1 + 0.1 * gamma.tanh()) + 0.1 * beta
        return F.normalize(value, dim=-1, eps=1e-8)

    def forward(self, scale: float = 0.5, rows: torch.Tensor | None = None) -> torch.Tensor:
        return self.scale_embedding(self.projected_latent(rows), scale)

    def decode_radio(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.radio_mean + (coefficients @ self.radio_basis.T) * self.radio_scale

    def radio_anchor_loss(self, rows: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(rows, device=self.latent.device, dtype=torch.long)
        return self.radio_anchor_loss_from_coefficients(
            indices, self.latent[indices]
        )

    def radio_anchor_loss_from_coefficients(
        self, rows: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the RADIO anchor for an explicit (possibly leaf) row block."""

        indices = torch.as_tensor(rows, device=self.latent.device, dtype=torch.long)
        if coefficients.shape != (indices.numel(), self.latent.shape[1]):
            raise ValueError("explicit anchor coefficient rows differ")
        before = self.decode_radio(self.base_latent[indices]).detach()
        after = self.decode_radio(coefficients)
        return (1.0 - F.cosine_similarity(after, before, dim=-1, eps=1e-8)).mean()

    @torch.no_grad()
    def radio_cosine(self, rows: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(rows, device=self.latent.device, dtype=torch.long)
        return F.cosine_similarity(
            self.decode_radio(self.latent[indices]),
            self.decode_radio(self.base_latent[indices]),
            dim=-1,
            eps=1e-8,
        )

    @torch.no_grad()
    def deployment_latent(self) -> torch.Tensor:
        return self.latent.detach().clone()


__all__ = ["JointD512Arm"]
