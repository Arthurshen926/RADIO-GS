"""Query-conditioned object membership over a frozen canonical latent.

The decoder adds no primitive-local state.  A mask/text descriptor identifies
the object, while the frozen L512 code identifies one Gaussian.  Their
interaction predicts membership; invisible source rows are never converted
into negative labels.
"""

from __future__ import annotations

import torch
from torch import nn


class FrozenLatentMembershipDecoder(nn.Module):
    """Shared text/region-to-Gaussian Bernoulli membership decoder."""

    def __init__(
        self,
        latent_dim: int = 512,
        query_dim: int = 1536,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(int(latent_dim), int(query_dim), int(hidden_dim)) <= 0:
            raise ValueError("membership decoder dimensions must be positive")
        self.gaussian_encoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(int(query_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.membership_head = nn.Sequential(
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def encode_gaussian(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must be one matrix")
        return self.gaussian_encoder(latent.float())

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2:
            raise ValueError("query must be one matrix")
        return self.query_encoder(query.float())

    def score_encoded(
        self, gaussian: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        if (
            gaussian.ndim != 2
            or identity.ndim != 2
            or gaussian.shape != identity.shape
        ):
            raise ValueError("encoded Gaussian and query matrices must align")
        interaction = torch.cat(
            (torch.abs(gaussian - identity), gaussian * identity), dim=-1
        )
        return self.membership_head(interaction).squeeze(-1)

    def forward(
        self, latent: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim != 2 or query.ndim != 2 or latent.shape[0] != query.shape[0]:
            raise ValueError("latent and query must be aligned matrices")
        return self.score_encoded(
            self.encode_gaussian(latent), self.encode_query(query)
        )


__all__ = ["FrozenLatentMembershipDecoder"]
