"""Experimental multi-teacher student with exactly one learned vector per row.

Native readout heads are shared over all Gaussians. They decode only the current
chunk and never own a second per-Gaussian feature tensor. This first student is
a capacity/control experiment, not a claim of preserving native multimodal measures.
"""
from __future__ import annotations

import math
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import QueryResult, Support

GEOMETRY = ("xyz", "normals", "quaternions", "scales", "opacities")
ARCHITECTURE = "unified_compact_v1"


class NativeReadout:
    """Non-owning compatibility interface; no cached decoded scene state."""
    def __init__(self, field, channel):
        self._owner = weakref.ref(field)
        self.channel = channel

    @property
    def native_dim(self):
        return self._owner().heads[self.channel].out_features

    def decode(self, rows):
        owner = self._owner()
        return owner.heads[self.channel](owner.latent[rows].float())

    def score(self, query, chunk_size=4096, *, patch_max=False):
        owner = self._owner()
        query = F.normalize(query.to(owner.latent).reshape(-1, self.native_dim), dim=-1)
        scores = []
        for start in range(0, len(owner.latent), chunk_size):
            decoded = F.normalize(self.decode(slice(start, start + chunk_size)), dim=-1)
            value = decoded @ query.T
            scores.append(value.max(1).values if patch_max else value)
        return torch.cat(scores)


class UnifiedCompactField(nn.Module):
    def __init__(self, geometry, observed, latent, *, semantic_dim, local_dim,
                 hidden_dim=64, default_scale=0.5, world_units_per_metre=1.,
                 threshold=0.5, candidate_count=4):
        super().__init__()
        n = len(observed)
        if latent.ndim != 2 or len(latent) != n or not torch.isfinite(latent).all():
            raise ValueError("invalid single Gaussian latent")
        if observed.dtype != torch.bool or observed.shape != (n,):
            raise ValueError("invalid observed support")
        if min(semantic_dim, local_dim, hidden_dim, candidate_count) < 1:
            raise ValueError("invalid decoder dimensions")
        if any(not math.isfinite(float(x)) or x <= 0 for x in (default_scale, world_units_per_metre)):
            raise ValueError("invalid spatial units")
        if not 0 < threshold < 1:
            raise ValueError("invalid membership threshold")
        for key in GEOMETRY:
            value = geometry[key].detach()
            if len(value) != n or not torch.isfinite(value).all():
                raise ValueError("geometry and latent row identity differ")
            self.register_buffer(key, value)
        self.register_buffer("observed", observed.detach())
        self.latent = nn.Parameter(latent.detach().clone())
        d = latent.shape[1]
        self.heads = nn.ModuleDict({"semantic": nn.Linear(d, semantic_dim),
                                    "local": nn.Linear(d, local_dim)})
        # All modalities share this conditional spatial decoder. Prototype,
        # differences and relative geometry are transient query state.
        self.member = nn.Sequential(nn.Linear(3 * d + 5, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, 1))
        self.semantic = NativeReadout(self, "semantic")
        self.local = NativeReadout(self, "local")
        self.default_scale = float(default_scale)
        self.world_units_per_metre = float(world_units_per_metre)
        self.threshold = float(threshold)
        self.candidate_count = int(candidate_count)
        self.semantic_measure = None
        # Typed public raster adapters must use the actual compact decoder's
        # support semantics; these are not the legacy ray-energy programs.
        self.support_aggregation = "weighted_prototype"
        self.negative_rule = "member_exclusion"

    @property
    def member_available(self):
        return self.observed

    def member_logits(self, support, scale, rows=None, *, explicit=False):
        support.validate(len(self.latent))
        if not math.isfinite(float(scale)) or scale <= 0:
            raise ValueError("invalid query scale")
        ids = support.indices.to(self.latent.device)
        weights = support.weights.to(self.latent)
        weights = weights * self.observed[ids]
        if not bool(weights.sum() > 0):
            raise ValueError("support has no observed mass")
        weights = weights / weights.sum()
        prototype = (F.normalize(self.latent[ids], dim=-1) * weights[:, None]).sum(0)
        centre = (self.xyz[ids] * weights[:, None]).sum(0)
        rows = slice(None) if rows is None else rows
        code = F.normalize(self.latent[rows], dim=-1)
        relative = (self.xyz[rows] - centre) / (float(scale) * self.world_units_per_metre)
        features = torch.cat([code, prototype.expand_as(code), code * prototype,
                              relative.clamp(-10, 10), relative.norm(dim=1, keepdim=True).log1p(),
                              code.new_full((len(code), 1), float(explicit))], dim=1)
        return self.member(features).flatten()

    def support_membership(self, positive, scale, negative=None, *, explicit=False):
        result = []
        for start in range(0, len(self.latent), 4096):
            rows = slice(start, start + 4096)
            value = self.member_logits(positive, scale, rows, explicit=explicit).sigmoid()
            if negative is not None:
                value = value * (1 - self.member_logits(negative, scale, rows, explicit=True).sigmoid())
            result.append(value)
        return torch.cat(result).masked_fill(~self.observed, 0)

    @torch.no_grad()
    def query(self, packet):
        packet.validate(len(self.xyz), self.semantic.native_dim, self.local.native_dim)
        scale = self.default_scale if packet.scale is None else packet.scale
        native = None if packet.semantic is None else self.semantic.score(packet.semantic).flatten()
        identity = native
        if packet.local is not None:
            # Respect valid patch mass; unavailable/padded patches are not evidence.
            patch = self.local.score(packet.local)
            weights = (patch.new_ones(patch.shape[1]) if packet.local_weights is None
                       else packet.local_weights.to(patch))
            local = (patch * (weights / weights.sum())[None]).sum(1)
            identity = local if identity is None else .5 * (identity + local)
        if packet.positive is not None:
            support = packet.positive
            members = self.support_membership(support, scale, packet.negative, explicit=True)
            candidates = support.indices.to(self.xyz.device)
            weights = support.weights.to(self.xyz)
            weights = weights / weights.sum()
        else:
            valid = torch.where(self.observed)[0]
            if not len(valid):
                raise ValueError("no observed identity candidates")
            candidates = valid[identity[valid].topk(min(self.candidate_count, len(valid))).indices]
            weights = (10 * identity[candidates]).softmax(0)
            if packet.task == "instance":
                candidates = candidates[:1]
                weights = weights[:1] * 0 + 1
            support = Support(candidates, weights)
            members = self.support_membership(support, scale, packet.negative)
        if packet.exact_positive:
            ids = packet.positive.indices[packet.positive.weights > 0].to(members.device)
            if not bool(self.observed[ids].all()):
                raise ValueError("exact positive outside observed support")
            members[ids] = 1
            if packet.negative is not None:
                members[packet.negative.indices[packet.negative.weights > 0].to(members.device)] = 0
        return QueryResult(native, members, members > self.threshold, candidates, weights, scale,
                           dict(architecture=ARCHITECTURE, shared_member_core=True, source_access=False,
                                compact_student=True, single_latent=True, uncertainty_resolved=False,
                                native_multimode_preservation_verified=False,
                                hard_positive_constraint=packet.exact_positive))

    def category_partition(self, queries):
        scores = self.semantic.score(queries)
        return scores, scores.argmax(1).masked_fill(~self.observed, -1)

    def export_spec(self):
        return dict(architecture=ARCHITECTURE, latent_dim=self.latent.shape[1],
                    semantic_dim=self.semantic.native_dim, local_dim=self.local.native_dim,
                    hidden_dim=self.member[0].out_features, default_scale=self.default_scale,
                    world_units_per_metre=self.world_units_per_metre, threshold=self.threshold,
                    candidate_count=self.candidate_count)


def from_state(state, spec, device="cpu"):
    arguments = {k: v for k, v in spec.items() if k not in {"architecture", "latent_dim"}}
    if state["latent"].shape[1] != spec["latent_dim"]:
        raise ValueError("latent dimension differs from specification")
    field = UnifiedCompactField({k: state[k] for k in GEOMETRY}, state["observed"],
                                state["latent"], **arguments)
    field.load_state_dict(state, strict=True)
    return field.to(device)
