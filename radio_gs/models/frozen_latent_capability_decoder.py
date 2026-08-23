"""Query-independent capability decoder over a frozen per-Gaussian latent."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenLatentCapabilityDecoder(nn.Module):
    """Predict a normalized capability residual without changing field state.

    The existing typed descriptor is an explicit skip connection.  Zero
    initialization therefore reproduces the deployed readout exactly, while
    the learned scene-global mapping can restore capability statistics lost by
    applying a nonlinear head after multi-view aggregation.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        capability_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.capability_dim = int(capability_dim)
        self.norm = nn.LayerNorm(self.latent_dim)
        self.hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.capability_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, latent: torch.Tensor, baseline_capability: torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [N,latent_dim]")
        if baseline_capability.shape != (latent.shape[0], self.capability_dim):
            raise ValueError("baseline capability rows do not align")
        residual = self.output(F.gelu(self.hidden(self.norm(latent))))
        return F.normalize(baseline_capability + residual, dim=-1, eps=1e-8)

