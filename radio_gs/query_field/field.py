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
                 threshold=0.5, candidate_count=8, candidate_radius=0.05,
                 anchor_eligible=None, spatial_metric="none", world_units_per_metre=None,
                 local_metric="none", support_aggregation="normalized_prototype", negative_rule="product",
                 relation_decoder="none", relation_confidence=None,
                 evidence_kind="none", evidence_state=None):
        super().__init__()
        count = len(geometry["xyz"])
        if any(len(x) != count for x in [semantic, local, group, observed]):
            raise ValueError("all subspaces must share the same Gaussian rows")
        if not math.isfinite(default_scale) or default_scale <= 0:
            raise ValueError("invalid default physical scale")
        if not 0 < threshold < 1 or candidate_count < 1 or candidate_radius <= 0:
            raise ValueError("invalid readout policy")
        if spatial_metric not in {"none", "scale_normalized"}:
            raise ValueError("unknown spatial member metric")
        if local_metric not in {"none", "native_cosine"}:
            raise ValueError("unknown native local member metric")
        if support_aggregation not in {"normalized_prototype", "expected_energy"}:
            raise ValueError("unknown support aggregation")
        if negative_rule not in {"product", "competitive_pooled", "competitive_rays"}:
            raise ValueError("unknown signed evidence rule")
        if negative_rule != "product" and support_aggregation != "normalized_prototype":
            raise ValueError("competitive evidence requires the accepted prototype kernel")
        if world_units_per_metre is not None and (
                not math.isfinite(world_units_per_metre) or world_units_per_metre <= 0):
            raise ValueError("invalid geometry world units per metre")
        if spatial_metric != "none" and world_units_per_metre is None:
            raise ValueError("spatial membership requires explicit world units per metre")
        for key in ["xyz", "normals", "quaternions", "scales", "opacities"]:
            value = geometry[key].detach().float()
            if len(value) != count or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid frozen Gaussian geometry")
            self.register_buffer(key, value)
        self.register_buffer("observed", observed.detach().bool())
        if anchor_eligible is not None:
            if anchor_eligible.shape != (count,) or anchor_eligible.dtype != torch.bool:
                raise ValueError("anchor eligibility must be a Boolean Gaussian-row mask")
            if not bool(anchor_eligible.any()) or bool((anchor_eligible & ~observed.bool()).any()):
                raise ValueError("inferred anchors require nonempty observed training support")
            anchor_eligible = anchor_eligible.detach().clone()
        # Optional, source-only exposure summary. This limits inferred input
        # anchors, never the decoded members or explicitly registered prompts.
        self.register_buffer("anchor_eligible", anchor_eligible)
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
        self.spatial_metric = spatial_metric
        self.world_units_per_metre = world_units_per_metre
        self.local_metric = local_metric
        self.support_aggregation = support_aggregation
        self.negative_rule = negative_rule
        self.relation_decoder = "none"
        self.relation = None
        self.register_buffer("relation_confidence", None)
        if relation_decoder != "none":
            if relation_decoder != "native_geometry_v1" or relation_confidence is None:
                raise ValueError("unknown or incomplete conditional relation decoder")
            from .conditional_relation import enable
            enable(self, relation_confidence)
        self.evidence_kind = 'none'
        self.evidence = None
        if evidence_kind != 'none':
            from .paired_evidence import KIND, enable
            if evidence_kind != KIND or evidence_state is None:
                raise ValueError('unknown or incomplete paired-evidence field')
            enable(self, evidence_state)

    def grouping_features(self, scale):
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("physical scale must be positive")
        gate = torch.sigmoid(2 * (math.log(float(scale)) - self.scale_cutoffs))
        return F.normalize(self.group * gate, dim=-1)

    def member_logits(self, support: Support, scale, rows=None):
        support.validate(len(self.xyz))
        if self.support_aggregation == "expected_energy" and len(support.indices) > 1:
            from .support_energy import expected_logits
            return expected_logits(self, support, scale, rows)
        features = self.grouping_features(scale)
        ids = support.indices.to(features.device)
        weights = support.weights.to(features).reshape(-1, 1)
        prototype = F.normalize((features[ids] * weights).sum(0), dim=0)
        selected = features if rows is None else features[rows]
        cosine = selected @ prototype
        affinity = cosine - self.affinity_bias
        if self.local_metric == "native_cosine":
            # Protected native directions contribute a common local metric,
            # including for registered rays and negative supports. Decoding and
            # scoring remain paired for full-native and affine-codec exports.
            native_support = F.normalize(self.local.decode(ids), dim=-1)
            native_prototype = (native_support * weights).sum(0)
            if rows is None:
                local_cosine = self.local.score(native_prototype).flatten()
            else:
                local_cosine = F.normalize(self.local.decode(rows), dim=-1) @ F.normalize(native_prototype, dim=0)
            affinity = affinity - (1 - local_cosine.clamp(-1, 1))
        logits = self.log_temperature.exp().clamp(1, 50) * affinity
        if self.spatial_metric == "scale_normalized":
            # A physical-scale log prior on the same conditional member energy.
            # A ray keeps its weighted support; no contributor is hard selected.
            centre = (self.xyz[ids] * weights).sum(0) / weights.sum()
            xyz = self.xyz if rows is None else self.xyz[rows]
            distance = (xyz - centre) / (float(scale) * self.world_units_per_metre)
            logits = logits - 0.5 * distance.square().sum(-1)
        if self.relation is not None:
            from .conditional_relation import residual
            logits = logits + residual(self, support, scale, cosine, rows)
        if self.evidence is not None:
            logits = self.evidence.refine(logits, support, scale, rows)
        return logits

    def _candidates(self, score):
        eligible = self.observed if self.anchor_eligible is None else self.anchor_eligible
        # Examine the complete declared support domain before spatial NMS.
        # Membership output always retains the full observed Gaussian carrier.
        order = torch.argsort(score.masked_fill(~eligible, -torch.inf), descending=True, stable=True)
        order = order[eligible[order]]
        chosen = []
        while len(order) and len(chosen) < self.candidate_count:
            index = order[0]
            chosen.append(index)
            distance = (self.xyz[order] - self.xyz[index]).norm(dim=-1)
            order = order[distance >= self.candidate_radius]
        return torch.stack(chosen) if chosen else torch.empty(0, dtype=torch.long, device=score.device)

    def exclude_membership(self, membership, negative, scale, positive_logits=None):
        """Shared differentiable signed rule for deployed queries and source fit."""
        if negative is not None:
            negative.validate(len(self.xyz))
            ids = negative.indices.to(self.xyz.device)
            weights = negative.weights.to(self.xyz.device).float()
            supported = self.observed[ids] & (weights > 0)
            if bool(supported.any()):
                components = None if negative.components is None else negative.components.to(ids.device)[supported]
                support = Support(ids[supported], weights[supported], components)
                if self.negative_rule != "product":
                    from .negative_evidence import competitive_membership, ray_log_mixture
                    if positive_logits is None:
                        raise ValueError("competition requires the original unsaturated positive logits")
                    logits = (ray_log_mixture(self, support, scale) if self.negative_rule == "competitive_rays"
                              else self.member_logits(support, scale))
                    return competitive_membership(positive_logits, logits)
                if self.support_aggregation == "expected_energy" and components is not None:
                    from .support_energy import independent_exclusion
                    exclusion = independent_exclusion(self, support, scale)
                else:
                    exclusion = self.member_logits(support, scale).sigmoid()
                membership = membership * (1 - exclusion)
        return membership

    def support_logits(self, positive, scale):
        """Common observed-support filtering before any signed conditioning."""
        positive.validate(len(self.xyz))
        ids = positive.indices.to(self.xyz.device)
        weights = positive.weights.to(self.xyz.device).float()
        supported = self.observed[ids]
        if not bool(weights[supported].sum() > 0):
            raise ValueError("positive support lacks trained observation support")
        return self.member_logits(Support(ids[supported], weights[supported]), scale)

    def support_membership(self, positive, scale, negative=None):
        """Uncertain registered supports; exact constraints remain query-only."""
        logits = self.support_logits(positive, scale)
        return self.exclude_membership(logits.sigmoid(), negative, scale, logits)

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
            positive_logits = self.support_logits(Support(ids[supported], weights[supported]), scale)
            membership = positive_logits.sigmoid()
        else:
            positive_coverage = None
            candidates = self._candidates(identity)
            if not len(candidates):
                raise ValueError("field has no observed identity candidates")
            # These weights expose uncertainty; they are not calibrated identity.
            candidate_weights = torch.softmax(10 * identity[candidates], dim=0)
            competing = packet.negative is not None and self.negative_rule != "product"
            values = [self.member_logits(Support(i.reshape(1), torch.ones(1, device=self.xyz.device)), scale)
                      if competing else self.member_logits(
                          Support(i.reshape(1), torch.ones(1, device=self.xyz.device)), scale).sigmoid()
                      for i in candidates]
            alternatives = torch.stack([value.sigmoid() for value in values] if competing else values, dim=1)
            positive_logits = None
            if packet.task == "instance":
                # Return one coherent hypothesis. Preserve all alternatives in
                # the receipt; do not average incompatible physical instances.
                membership = alternatives[:, candidate_weights.argmax()]
                if competing:
                    positive_logits = values[int(candidate_weights.argmax())]
            else:
                # Union includes hypotheses supported near the best identity.
                # Global policy; not a query/benchmark-selected threshold.
                admitted = identity[candidates] >= identity[candidates].max() - 0.02
                membership = alternatives[:, admitted].max(dim=1).values
                if competing:
                    positive_logits = torch.stack(values, dim=1)[:, admitted].max(dim=1).values
        membership = self.exclude_membership(membership, packet.negative, scale, positive_logits)
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
                               inferred_anchor_domain="all_observed" if self.anchor_eligible is None else "training_eligible",
                               spatial_member_metric=self.spatial_metric,
                               native_local_member_metric=self.local_metric,
                               conditional_relation_decoder=self.relation_decoder,
                               paired_evidence=self.evidence_kind,
                               support_aggregation=self.support_aggregation,
                               negative_rule=self.negative_rule,
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
        spec = dict(default_scale=self.default_scale, threshold=self.threshold,
                    candidate_count=self.candidate_count, candidate_radius=self.candidate_radius,
                    spatial_metric=self.spatial_metric, world_units_per_metre=self.world_units_per_metre,
                    local_metric=self.local_metric, support_aggregation=self.support_aggregation,
                    negative_rule=self.negative_rule)
        if self.relation_decoder != "none":
            spec["relation_decoder"] = self.relation_decoder
        if self.evidence is not None:
            spec['evidence_kind'] = self.evidence_kind
        return spec
