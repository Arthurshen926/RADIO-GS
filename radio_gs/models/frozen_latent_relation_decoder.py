"""Scene-independent object-relation decoder over a frozen canonical latent.

The decoder has no per-Gaussian parameters.  It consumes proposal-pooled
canonical latent values plus relative 3-D geometry and predicts a proper
Bernoulli same-object posterior.  A shared language head keeps the proposal
representation aligned with source-mask SigLIP2 teachers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def symmetric_pair_features(
    embedding: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    centroids: torch.Tensor,
    area_fraction: torch.Tensor,
    scene_extent: torch.Tensor,
) -> torch.Tensor:
    """Construct swap-invariant appearance and relative-geometry features."""

    left = torch.as_tensor(left, device=embedding.device, dtype=torch.long)
    right = torch.as_tensor(right, device=embedding.device, dtype=torch.long)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("pair indices must be aligned vectors")
    if centroids.shape != (embedding.shape[0], 3):
        raise ValueError("proposal centroids must have shape [P,3]")
    area = torch.as_tensor(area_fraction, device=embedding.device).float().reshape(-1)
    if area.shape != (embedding.shape[0],):
        raise ValueError("proposal areas must have shape [P]")
    extent = torch.as_tensor(scene_extent, device=embedding.device).float().reshape(-1)
    if extent.shape != (3,) or not bool(torch.isfinite(extent).all()):
        raise ValueError("scene extent must be a finite 3-vector")
    a, b = embedding[left], embedding[right]
    delta = (centroids[left] - centroids[right]).abs() / extent.clamp_min(1e-6)
    distance = delta.square().sum(-1, keepdim=True).sqrt()
    log_area_ratio = (
        area[left].clamp_min(1e-8).log() - area[right].clamp_min(1e-8).log()
    ).abs()[:, None]
    return torch.cat((torch.abs(a - b), a * b, delta, distance, log_area_ratio), dim=-1)


class FrozenLatentRelationDecoder(nn.Module):
    """Global proposal encoder, relation posterior, and mask-language decoder."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 96, language_dim: int = 1536) -> None:
        super().__init__()
        if min(int(latent_dim), int(hidden_dim), int(language_dim)) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.proposal_encoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.relation_decoder = nn.Sequential(
            nn.Linear(2 * int(hidden_dim) + 5, int(hidden_dim)),
            nn.GELU(),
            # Categorical order: different, same, unknown.
            nn.Linear(int(hidden_dim), 3),
        )
        self.language_decoder = nn.Linear(int(hidden_dim), int(language_dim), bias=False)

    def encode(self, pooled_latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proposal_encoder(pooled_latent.float()), dim=-1, eps=1e-8)

    def relation_logits(
        self,
        embedding: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        centroids: torch.Tensor,
        area_fraction: torch.Tensor,
        scene_extent: torch.Tensor,
    ) -> torch.Tensor:
        features = symmetric_pair_features(
            embedding,
            left,
            right,
            centroids,
            area_fraction,
            scene_extent,
        )
        return self.relation_decoder(features)

    def decode_language(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.language_decoder(embedding), dim=-1, eps=1e-8)


__all__ = ["FrozenLatentRelationDecoder", "symmetric_pair_features"]
