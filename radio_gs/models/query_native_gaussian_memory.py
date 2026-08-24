"""Direct query-to-posterior decoders over one frozen Gaussian memory."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.query_packet import QueryPacket


class ModalityQueryAdapter(nn.Module):
    """Small adapter from one frozen encoder space into shared query tokens."""

    def __init__(self, input_dim: int, query_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.query_dim = int(query_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.query_dim),
            nn.GELU(),
            nn.LayerNorm(self.query_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != self.input_dim:
            raise ValueError("modality query adapter input differs")
        return self.network(value.float())


class QuerySetCategoricalDecoder(nn.Module):
    """Class-set-equivariant residual decoder for arbitrary text queries.

    The module has no class-indexed parameter. Query competition enters only
    through symmetric set statistics, so reordering a query set reorders the
    output and changing its cardinality needs no architectural change.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        reliability_dim: int = 5,
        query_dim: int = 1536,
        hidden_dim: int = 192,
        pair_hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.query_dim = int(query_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.reliability_norm = nn.LayerNorm(self.reliability_dim)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.latent_projection = nn.Linear(
            self.latent_dim + self.reliability_dim, int(hidden_dim)
        )
        self.query_projection = nn.Linear(self.query_dim, int(hidden_dim))
        self.pair = nn.Sequential(
            nn.Linear(7, int(pair_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pair_hidden_dim), 1),
        )
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(
        self,
        latent: torch.Tensor,
        reliability: torch.Tensor,
        query_tokens: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("categorical latent input differs")
        if reliability.shape != (latent.shape[0], self.reliability_dim):
            raise ValueError("categorical reliability input differs")
        if query_tokens.ndim != 2 or query_tokens.shape[1] != self.query_dim:
            raise ValueError("categorical query token input differs")
        if baseline_scores.shape != (latent.shape[0], query_tokens.shape[0]):
            raise ValueError("categorical baseline score input differs")
        if query_tokens.shape[0] < 2:
            raise ValueError("categorical query set requires at least two entries")

        gaussian = F.normalize(
            self.latent_projection(torch.cat((
                self.latent_norm(latent.float()),
                self.reliability_norm(reliability.float()),
            ), dim=-1)), dim=-1,
        )
        query = F.normalize(
            self.query_projection(self.query_norm(query_tokens.float())), dim=-1
        )
        interaction = gaussian @ query.T
        baseline = baseline_scores.float()
        mean = baseline.mean(1, keepdim=True)
        centered = baseline - mean
        std = baseline.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        maximum = baseline.max(1, keepdim=True).values
        query_context = query @ query.mean(0)
        pair_features = torch.stack((
            interaction, baseline, centered, mean.expand_as(baseline),
            std.expand_as(baseline), maximum.expand_as(baseline),
            query_context[None, :].expand_as(baseline),
        ), dim=-1)
        residual = self.pair(pair_features).squeeze(-1)
        return F.normalize(baseline + residual, dim=-1, eps=1e-8)


class QuerySetEligibilityGate(nn.Module):
    """Permutation-invariant source-authority gate for one query set."""

    def __init__(
        self,
        latent_dim: int = 512,
        reliability_dim: int = 5,
        query_dim: int = 1536,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.query_dim = int(query_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.reliability_norm = nn.LayerNorm(self.reliability_dim)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.query_projection = nn.Linear(self.query_dim, 32)
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim + self.reliability_dim + 32 + 6, int(hidden_dim)),
            nn.GELU(), nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        reliability: torch.Tensor,
        query_tokens: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("eligibility latent input differs")
        if reliability.shape != (latent.shape[0], self.reliability_dim):
            raise ValueError("eligibility reliability input differs")
        if query_tokens.ndim != 2 or query_tokens.shape[1] != self.query_dim:
            raise ValueError("eligibility query token input differs")
        if baseline_scores.shape != (latent.shape[0], query_tokens.shape[0]):
            raise ValueError("eligibility baseline score input differs")
        query_context = self.query_projection(self.query_norm(query_tokens.float())).mean(0)
        score = baseline_scores.float()
        top2 = score.topk(k=min(2, score.shape[1]), dim=1).values
        margin = top2[:, :1] - top2[:, -1:]
        statistics = torch.cat((
            score.mean(1, keepdim=True), score.std(1, keepdim=True, unbiased=False),
            score.min(1, keepdim=True).values, score.max(1, keepdim=True).values,
            margin, score.abs().mean(1, keepdim=True),
        ), dim=1)
        value = torch.cat((
            self.latent_norm(latent.float()), self.reliability_norm(reliability.float()),
            query_context[None, :].expand(latent.shape[0], -1), statistics,
        ), dim=1)
        return self.network(value).squeeze(-1)


class QueryNativeGaussianPosteriorDecoder(nn.Module):
    """Shared dense reference decoder from a QueryPacket to Gaussian logits."""

    def __init__(
        self,
        latent_dim: int = 512,
        reliability_dim: int = 5,
        query_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.query_dim = int(query_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.reliability_norm = nn.LayerNorm(self.reliability_dim)
        self.gaussian_key = nn.Linear(self.latent_dim, self.query_dim)
        self.identity_residual_scale = nn.Parameter(torch.zeros(()))
        self.extent = nn.Sequential(
            nn.Linear(self.query_dim * 2 + self.reliability_dim + 1, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.extent[-1].weight)
        nn.init.zeros_(self.extent[-1].bias)

    def forward(
        self,
        latent: torch.Tensor,
        reliability: torch.Tensor,
        packet: QueryPacket,
        identity_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("posterior latent input differs")
        if reliability.shape != (latent.shape[0], self.reliability_dim):
            raise ValueError("posterior reliability input differs")
        if packet.tokens.shape[1] != self.query_dim:
            raise ValueError("posterior query packet dimension differs")
        if packet.seed_probability is not None and packet.seed_probability.shape != (latent.shape[0],):
            raise ValueError("posterior prompt seed row domain differs")
        if identity_prior is not None and identity_prior.shape != (latent.shape[0],):
            raise ValueError("posterior identity prior row domain differs")

        key = F.normalize(self.gaussian_key(self.latent_norm(latent.float())), dim=-1)
        query = F.normalize(packet.tokens.float(), dim=-1)
        token_score = key @ query.T / math.sqrt(float(self.query_dim))
        if packet.confidence is None:
            learned_identity = torch.logsumexp(token_score, dim=1) - math.log(query.shape[0])
        else:
            weight = packet.confidence.float().clamp_min(1e-8)
            learned_identity = torch.logsumexp(
                token_score + weight.log()[None, :], dim=1
            ) - weight.sum().log()
        identity = (
            learned_identity
            if identity_prior is None
            else identity_prior.float() + self.identity_residual_scale * learned_identity
        )
        anchor_weight = torch.softmax(identity.detach() / 0.05, dim=0)
        anchor = (anchor_weight[:, None] * key).sum(0, keepdim=True).expand_as(key)
        extent_input = torch.cat((
            key, anchor, self.reliability_norm(reliability.float()), identity[:, None],
        ), dim=1)
        logits = identity + self.extent(extent_input).squeeze(-1)
        if packet.seed_probability is not None:
            seed = packet.seed_probability.float()
            known = torch.isfinite(seed)
            logits = logits.clone()
            logits[known] = torch.logit(seed[known].clamp(1e-4, 1.0 - 1e-4))
        return logits, identity


__all__ = [
    "ModalityQueryAdapter",
    "QueryNativeGaussianPosteriorDecoder",
    "QuerySetCategoricalDecoder",
    "QuerySetEligibilityGate",
]
