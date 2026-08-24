"""Direct query-to-posterior decoders over one frozen Gaussian memory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.query_packet import QueryPacket


@dataclass(frozen=True)
class GaussianGeometry:
    """Finite geometry state aligned with the decoder's Gaussian rows."""

    xyz: torch.Tensor
    scales: torch.Tensor | None = None
    opacity: torch.Tensor | None = None

    def validate(self, rows: int) -> None:
        if self.xyz.shape != (rows, 3) or not bool(torch.isfinite(self.xyz).all()):
            raise ValueError("Gaussian geometry xyz differs")
        if self.scales is not None and (
            self.scales.shape != (rows, 3)
            or not bool(torch.isfinite(self.scales).all())
            or bool((self.scales <= 0).any())
        ):
            raise ValueError("Gaussian geometry scales differ")
        if self.opacity is not None and (
            self.opacity.reshape(-1).shape != (rows,)
            or not bool(torch.isfinite(self.opacity).all())
        ):
            raise ValueError("Gaussian geometry opacity differs")


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


class LowRankSceneCanonicalizer(nn.Module):
    """Constant-size scene FiLM for aligning scene-gauged latent coordinates.

    The module owns only ``O(num_scenes * rank + latent_dim * rank)`` state,
    never a Gaussian-indexed table.  Both low-rank up projections start at
    exact zero, so enabling it initially replays the frozen memory bitwise.
    """

    def __init__(self, num_scenes: int, latent_dim: int = 512, rank: int = 8) -> None:
        super().__init__()
        self.num_scenes = int(num_scenes)
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        if min(self.num_scenes, self.latent_dim, self.rank) <= 0:
            raise ValueError("scene canonicalizer dimensions must be positive")
        self.scene_code = nn.Embedding(self.num_scenes, self.rank * 2)
        self.scale_up = nn.Linear(self.rank, self.latent_dim, bias=False)
        self.shift_up = nn.Linear(self.rank, self.latent_dim, bias=False)
        nn.init.normal_(self.scene_code.weight, std=0.02)
        nn.init.zeros_(self.scale_up.weight)
        nn.init.zeros_(self.shift_up.weight)

    def forward(self, latent: torch.Tensor, scene_index: int | torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("scene canonicalizer latent input differs")
        index = torch.as_tensor(scene_index, device=latent.device, dtype=torch.long)
        if index.ndim == 0:
            index = index.expand(latent.shape[0])
        if index.shape != (latent.shape[0],):
            raise ValueError("scene canonicalizer index row domain differs")
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= self.num_scenes):
            raise IndexError("scene canonicalizer index is out of range")
        scale_code, shift_code = self.scene_code(index).chunk(2, dim=-1)
        scale = self.scale_up(scale_code)
        shift = self.shift_up(shift_code)
        return latent.float() * (1.0 + scale) + shift


class CounterfactualSelectiveRiskEstimator(nn.Module):
    """Predict beneficial/harmful/neutral risk for a frozen score candidate.

    This module never changes semantic scores.  It only estimates the
    counterfactual risk of adopting a separately frozen candidate, keeping
    scoring and selective deployment as identifiable components.
    """

    def __init__(
        self, latent_dim: int = 512, reliability_dim: int = 5,
        decision_feature_dim: int = 9, hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.decision_feature_dim = int(decision_feature_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.reliability_dim + self.decision_feature_dim),
            nn.Linear(self.latent_dim + self.reliability_dim + self.decision_feature_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3),
        )

    def forward(self, latent: torch.Tensor, reliability: torch.Tensor, decision_features: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("risk-estimator latent input differs")
        if reliability.shape != (latent.shape[0], self.reliability_dim):
            raise ValueError("risk-estimator reliability input differs")
        if decision_features.shape != (latent.shape[0], self.decision_feature_dim):
            raise ValueError("risk-estimator decision feature input differs")
        return self.network(torch.cat((latent.float(), reliability.float(), decision_features.float()), dim=1))


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
        factorized_identity_competition: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.query_dim = int(query_dim)
        self.factorized_identity_competition = bool(factorized_identity_competition)
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
        self.identity_pair = None
        if self.factorized_identity_competition:
            self.identity_pair = nn.Sequential(
                nn.Linear(1, int(pair_hidden_dim)), nn.GELU(),
                nn.Linear(int(pair_hidden_dim), 1),
            )
            nn.init.zeros_(self.identity_pair[-1].weight)
            nn.init.zeros_(self.identity_pair[-1].bias)

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
        if self.identity_pair is not None:
            # Pair-local identity cannot change when unrelated alternatives
            # enter the query set. Only centered competition is set-dependent.
            identity = self.identity_pair(interaction.unsqueeze(-1)).squeeze(-1)
            residual = identity + residual - residual.mean(1, keepdim=True)
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
    """Multi-anchor geometry-aware decoder from a query to Gaussian logits."""

    def __init__(
        self,
        latent_dim: int = 512,
        reliability_dim: int = 5,
        query_dim: int = 128,
        hidden_dim: int = 128,
        topk_anchors: int = 6,
        initial_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_dim = int(reliability_dim)
        self.query_dim = int(query_dim)
        self.topk_anchors = int(topk_anchors)
        if self.topk_anchors <= 0:
            raise ValueError("topk_anchors must be positive")
        if not 0.02 <= float(initial_temperature) <= 0.2:
            raise ValueError("initial identity temperature must be in [0.02,0.2]")
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.reliability_norm = nn.LayerNorm(self.reliability_dim)
        self.gaussian_key = nn.Linear(self.latent_dim, self.query_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(initial_temperature)), dtype=torch.float32)
        )
        self.identity_residual_scale = nn.Parameter(torch.zeros(()))
        # key, multi-anchor key, reliability, identity, and finite 3D relation:
        # delta xyz (3), Euclidean/Mahalanobis distance, scale ratio,
        # feature relation, anchor confidence, and opacity (9 values).
        self.extent = nn.Sequential(
            nn.Linear(self.query_dim * 2 + self.reliability_dim + 1 + 9, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.extent[-1].weight)
        nn.init.zeros_(self.extent[-1].bias)

    def identity_temperature(self) -> torch.Tensor:
        """Return the bounded learned cosine temperature."""

        return self.log_temperature.exp().clamp(0.02, 0.2)

    def forward(
        self,
        latent: torch.Tensor,
        reliability: torch.Tensor,
        packet: QueryPacket,
        identity_prior: torch.Tensor | None = None,
        geometry: GaussianGeometry | None = None,
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
        if geometry is not None:
            geometry.validate(latent.shape[0])

        key = F.normalize(self.gaussian_key(self.latent_norm(latent.float())), dim=-1)
        query = F.normalize(packet.tokens.float(), dim=-1)
        temperature = self.identity_temperature()
        token_score = key @ query.T / temperature
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
        anchor_count = min(self.topk_anchors, latent.shape[0])
        anchor_score, anchor_rows = identity.detach().topk(anchor_count)
        anchor_weight = torch.softmax(anchor_score / 0.05, dim=0)
        anchor_keys = key[anchor_rows]
        relation_weight = torch.softmax(key @ anchor_keys.T / temperature.detach(), dim=1)
        anchor = relation_weight @ anchor_keys

        if geometry is None:
            geometry_relation = key.new_zeros((key.shape[0], 9))
        else:
            xyz = geometry.xyz.to(device=key.device, dtype=key.dtype)
            anchor_xyz = xyz[anchor_rows]
            delta = xyz[:, None, :] - anchor_xyz[None, :, :]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            scale = (
                geometry.scales.to(device=key.device, dtype=key.dtype).clamp_min(1e-6)
                if geometry.scales is not None
                else torch.ones_like(xyz)
            )
            anchor_scale = scale[anchor_rows]
            mahalanobis = torch.sqrt(
                (delta.square() / anchor_scale[None].square()).sum(-1).clamp_min(1e-8)
            )
            scale_ratio = (
                scale.log().mean(-1, keepdim=True)
                - anchor_scale.log().mean(-1)[None, :]
            ).abs()
            feature_relation = key @ anchor_keys.T
            confidence = torch.sigmoid(anchor_score)[None, :].expand_as(distance)
            opacity = (
                geometry.opacity.to(device=key.device, dtype=key.dtype).reshape(-1, 1)
                if geometry.opacity is not None
                else torch.ones((key.shape[0], 1), device=key.device, dtype=key.dtype)
            )
            relation = torch.cat((
                delta,
                distance[..., None],
                mahalanobis[..., None],
                scale_ratio[..., None],
                feature_relation[..., None],
                confidence[..., None],
            ), dim=-1)
            geometry_relation = (relation * relation_weight[..., None]).sum(1)
            geometry_relation = torch.cat((geometry_relation, opacity), dim=-1)
        extent_input = torch.cat((
            key, anchor, self.reliability_norm(reliability.float()), identity[:, None],
            geometry_relation,
        ), dim=1)
        logits = identity + self.extent(extent_input).squeeze(-1)
        # Extent propagation may add support, but cannot erase the strongest
        # identity evidence. This keeps LocAcc/identity peaks invariant.
        logits = logits.clone()
        logits[anchor_rows] = torch.maximum(logits[anchor_rows], identity[anchor_rows])
        if packet.seed_probability is not None:
            seed = packet.seed_probability.float()
            known = torch.isfinite(seed)
            logits = logits.clone()
            logits[known] = torch.logit(seed[known].clamp(1e-4, 1.0 - 1e-4))
        return logits, identity


__all__ = [
    "GaussianGeometry",
    "LowRankSceneCanonicalizer",
    "ModalityQueryAdapter",
    "QueryNativeGaussianPosteriorDecoder",
    "QuerySetCategoricalDecoder",
    "QuerySetEligibilityGate",
]
