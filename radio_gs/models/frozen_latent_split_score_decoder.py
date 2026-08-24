"""Compact categorical score residuals over a frozen Gaussian latent."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenLatentSplitScoreDecoder(nn.Module):
    """Scene-global residual decoder with independent normalized score blocks."""

    def __init__(
        self, latent_dim: int = 512, hidden_dim: int = 256,
        split_dims: tuple[int, ...] = (19, 15, 10),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.split_dims = tuple(int(value) for value in split_dims)
        self.score_dim = sum(self.split_dims)
        self.norm = nn.LayerNorm(self.latent_dim)
        self.hidden = nn.Linear(self.latent_dim, int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), self.score_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor, baseline_scores: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent score-decoder input differs")
        if baseline_scores.shape != (latent.shape[0], self.score_dim):
            raise ValueError("baseline score blocks differ")
        residual = self.output(F.gelu(self.hidden(self.norm(latent.float()))))
        raw = baseline_scores.float() + residual
        return torch.cat(
            [F.normalize(value, dim=-1, eps=1e-8) for value in raw.split(self.split_dims, dim=-1)],
            dim=-1,
        )


class FrozenReliabilityEligibilityGate(nn.Module):
    """Predict three eligibility bits from field state and transient scores."""

    def __init__(
        self, latent_dim: int = 512, reliability_dim: int = 5,
        score_dim: int = 44, hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.score_dim = int(score_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.reliability_norm = nn.LayerNorm(self.reliability_dim)
        self.score_norm = nn.LayerNorm(self.score_dim)
        self.hidden = nn.Linear(
            self.latent_dim + self.reliability_dim + self.score_dim,
            int(hidden_dim),
        )
        self.output = nn.Linear(int(hidden_dim), 3)

    def forward(
        self, latent: torch.Tensor, reliability: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("eligibility latent input differs")
        if reliability.shape != (latent.shape[0], self.reliability_dim):
            raise ValueError("eligibility reliability input differs")
        if baseline_scores.shape != (latent.shape[0], self.score_dim):
            raise ValueError("eligibility baseline-score input differs")
        value = torch.cat(
            (
                self.latent_norm(latent.float()),
                self.reliability_norm(reliability.float()),
                self.score_norm(baseline_scores.float()),
            ), dim=-1,
        )
        return self.output(F.gelu(self.hidden(value)))


__all__ = ["FrozenLatentSplitScoreDecoder", "FrozenReliabilityEligibilityGate"]
