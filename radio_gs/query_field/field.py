"""Native-aligned subspaces and one scale-conditioned Gaussian member decoder."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.models.native_decoder_readout import affine_decoder_cosine
from .contracts import QueryPacket, QueryResult, Support


class NativeCoordinates(nn.Module):
    """An affine codec with exactly paired native query scoring."""

    def __init__(self, codes, weight=None, bias=None):
        super().__init__()
        self.register_buffer("codes", codes.detach().contiguous())
        if weight is not None and (weight.ndim != 2 or weight.shape[1] != codes.shape[1]):
            raise ValueError("invalid affine codec")
        self.register_buffer("weight", None if weight is None else weight.detach().float())
        self.register_buffer("bias", None if bias is None else bias.detach().float())
        if (weight is None) != (bias is None):
            raise ValueError("affine codec requires both weight and bias")

    @property
    def native_dim(self):
        return self.codes.shape[1] if self.weight is None else self.weight.shape[0]

    def decode(self, rows):
        code = self.codes[rows].float()
        return code if self.weight is None else F.linear(code, self.weight, self.bias)

    def score(self, query, chunk_size=8192, *, patch_max=False):
        query = query.to(self.codes.device).float().reshape(-1, self.native_dim)
        values = []
        for codes in self.codes.split(chunk_size):
            codes = codes.float()
            if self.weight is None:
                scores = F.normalize(codes, dim=-1) @ F.normalize(query, dim=-1).T
            else:
                scores = affine_decoder_cosine(codes, query, self.weight, self.bias)
            values.append(scores.max(dim=1).values if patch_max else scores)
        return torch.cat(values)


class CapabilityField(nn.Module):
    """One carrier, protected native channels, and scene-specific grouping codes.

    Membership is a learned score, not a calibrated probability. Semantic and
    local coordinates are immutable during grouping optimization. Candidate
    identity remains explicit; it is not overwritten by member optimization.
    """

    def __init__(self, geometry, semantic, local, group, observed, *,
                 semantic_codec=None, local_codec=None, default_scale=0.5,
                 threshold=0.5, candidate_count=8, candidate_radius=0.05):
        super().__init__()
        count = len(geometry["xyz"])
        if any(len(x) != count for x in [semantic, local, group, observed]):
            raise ValueError("all subspaces must share the same Gaussian rows")
        if not math.isfinite(default_scale) or default_scale <= 0:
            raise ValueError("invalid default physical scale")
        if not 0 < threshold < 1 or candidate_count < 1 or candidate_radius <= 0:
            raise ValueError("invalid readout policy")
        for key in ["xyz", "normals", "quaternions", "scales", "opacities"]:
            value = geometry[key].detach().float()
            if len(value) != count or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid frozen Gaussian geometry")
            self.register_buffer(key, value)
        self.register_buffer("observed", observed.detach().bool())
        self.semantic = NativeCoordinates(semantic, **(semantic_codec or {}))
        self.local = NativeCoordinates(local, **(local_codec or {}))
        self.group = nn.Parameter(group.detach().float())
        self.scale_cutoffs = nn.Parameter(torch.linspace(math.log(default_scale)-2,
                                                        math.log(default_scale)+2, group.shape[1]))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(12.)))
        self.affinity_bias = nn.Parameter(torch.tensor(0.5))
        self.default_scale = float(default_scale)
        self.threshold = float(threshold)
        self.candidate_count = int(candidate_count)
        self.candidate_radius = float(candidate_radius)

    def grouping_features(self, scale):
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("physical scale must be positive")
        gate = torch.sigmoid(2 * (math.log(float(scale)) - self.scale_cutoffs))
        return F.normalize(self.group * gate, dim=-1)

    def member_logits(self, support: Support, scale, rows=None):
        support.validate(len(self.xyz))
        features = self.grouping_features(scale)
        ids = support.indices.to(features.device)
        weights = support.weights.to(features).reshape(-1, 1)
        prototype = F.normalize((features[ids] * weights).sum(0), dim=0)
        selected = features if rows is None else features[rows]
        return self.log_temperature.exp().clamp(1, 50) * (selected @ prototype - self.affinity_bias)

    def _candidates(self, score):
        # Examine the full observed carrier; no fixed top-k proposal truncation.
        order = torch.argsort(score.masked_fill(~self.observed, -torch.inf), descending=True, stable=True)
        order = order[self.observed[order]]
        chosen = []
        while len(order) and len(chosen) < self.candidate_count:
            index = order[0]
            chosen.append(index)
            distance = (self.xyz[order] - self.xyz[index]).norm(dim=-1)
            order = order[distance >= self.candidate_radius]
        return torch.stack(chosen) if chosen else torch.empty(0, dtype=torch.long, device=score.device)

    @torch.no_grad()
    def query(self, packet: QueryPacket):
        packet.validate(len(self.xyz), self.semantic.native_dim, self.local.native_dim)
        scale = self.default_scale if packet.scale is None else packet.scale
        native = None
        identity = None
        if packet.semantic is not None:
            native = self.semantic.score(packet.semantic).flatten()
            identity = native.clone()
        if packet.local is not None:
            # Local query patches are compared independently before reduction.
            patches = packet.local.reshape(-1, self.local.native_dim)
            local = self.local.score(patches, patch_max=True)
            identity = local if identity is None else (identity + local) * 0.5
        if packet.positive is not None:
            support = packet.positive
            ids = support.indices.to(self.xyz.device)
            supported = self.observed[ids]
            weights = support.weights.to(self.xyz.device).float()
            positive_coverage = float(weights[supported].sum()/weights.sum())
            if positive_coverage == 0 or (packet.exact_positive and positive_coverage < 1):
                raise ValueError("positive support lacks trained observation support")
            candidates = ids
            candidate_weights = weights / weights.sum()
            membership = self.member_logits(Support(ids[supported], weights[supported]), scale).sigmoid()
        else:
            positive_coverage = None
            candidates = self._candidates(identity)
            if not len(candidates):
                raise ValueError("field has no observed identity candidates")
            # These weights expose uncertainty; they are not calibrated identity.
            candidate_weights = torch.softmax(10 * identity[candidates], dim=0)
            alternatives = torch.stack([self.member_logits(
                Support(i.reshape(1), torch.ones(1, device=self.xyz.device)), scale).sigmoid()
                for i in candidates], dim=1)
            if packet.task == "instance":
                # Return one coherent hypothesis. Preserve all alternatives in
                # the receipt; do not average incompatible physical instances.
                membership = alternatives[:, candidate_weights.argmax()]
            else:
                # Union includes hypotheses supported near the best identity.
                # Global policy; not a query/benchmark-selected threshold.
                admitted = identity[candidates] >= identity[candidates].max() - 0.02
                membership = alternatives[:, admitted].max(dim=1).values
        if packet.negative is not None:
            ids = packet.negative.indices.to(self.xyz.device)
            weights = packet.negative.weights.to(self.xyz.device).float()
            supported = self.observed[ids] & (weights > 0)
            if bool(supported.any()):
                exclusion = self.member_logits(Support(ids[supported], weights[supported]), scale).sigmoid()
                membership = membership * (1 - exclusion)
        membership = membership.masked_fill(~self.observed, 0)
        if packet.exact_positive:
            ids = packet.positive.indices[packet.positive.weights > 0].to(membership.device)
            membership[ids] = 1
            if packet.negative is not None:
                ids = packet.negative.indices[packet.negative.weights > 0].to(membership.device)
                membership[ids] = 0
        return QueryResult(native, membership, membership > self.threshold,
                           candidates, candidate_weights, float(scale), dict(
                               shared_member_core=True, source_access=False,
                               identity_rule="native evidence with spatially separated hypotheses",
                               uncertainty_resolved=False, semantic_scores_calibrated=False,
                               membership_scores_calibrated=False,
                               observed_positive_mass_fraction=positive_coverage,
                               hard_positive_constraint=packet.exact_positive))

    @torch.no_grad()
    def category_partition(self, native_queries):
        """Explicit complete-menu semantic competition; no independent masks."""
        scores = self.semantic.score(native_queries)
        labels = scores.argmax(dim=1)
        labels[~self.observed] = -1
        return scores, labels

    def export_spec(self):
        return dict(default_scale=self.default_scale, threshold=self.threshold,
                    candidate_count=self.candidate_count, candidate_radius=self.candidate_radius)
