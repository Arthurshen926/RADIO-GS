"""Permutation-invariant text router for a frozen SurfaceRegion codebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


SURFACE_REGION_QUERY_ROUTER_V1 = "surface_region_query_router_v1"


class SurfaceRegionQueryRouterOutput(NamedTuple):
    response: torch.Tensor
    slot_weights: torch.Tensor
    slot_scores: torch.Tensor
    residual_gate: torch.Tensor


class SurfaceRegionQueryRouterV1(nn.Module):
    """Globally route queries between exact fallback and residual slots.

    The same scorer is applied to every residual slot.  Its inputs contain no
    slot identity, preserving permutation invariance of the persistent latent
    set.  The fallback logit is fixed at zero and remains available for every
    region-query pair.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        residual_slots: int = 3,
        codebook_sha256: str,
    ) -> None:
        super().__init__()
        self.input_dim = 7
        self.hidden_dim = int(hidden_dim)
        self.residual_slots = int(residual_slots)
        self.total_slots = 1 + self.residual_slots
        self.codebook_sha256 = str(codebook_sha256)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.residual_slots != 3:
            raise ValueError("query router V1 requires three residual slots")
        if len(self.codebook_sha256) != 64:
            raise ValueError("codebook_sha256 must bind the frozen codebook")
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        slot_descriptors: torch.Tensor,
        slot_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        *,
        logit_scale: float = 10.0,
    ) -> SurfaceRegionQueryRouterOutput:
        descriptors = F.normalize(
            torch.as_tensor(slot_descriptors).float(), dim=-1, eps=1e-8
        )
        tokens = torch.as_tensor(slot_tokens, device=descriptors.device).float()
        text = F.normalize(
            torch.as_tensor(text_embeddings, device=descriptors.device).float(),
            dim=-1,
            eps=1e-8,
        )
        negative_text = F.normalize(
            torch.as_tensor(
                negative_text_embeddings, device=descriptors.device
            ).float(),
            dim=-1,
            eps=1e-8,
        )
        if descriptors.ndim != 3 or descriptors.shape[1] != self.total_slots:
            raise ValueError("slot_descriptors must be [B,4,D]")
        if (
            tokens.ndim != 3
            or tokens.shape[:2] != descriptors.shape[:2]
            or text.ndim != 2
            or text.shape[-1] != descriptors.shape[-1]
            or negative_text.ndim != 2
            or negative_text.shape[-1] != descriptors.shape[-1]
            or negative_text.shape[0] == 0
        ):
            raise ValueError("slot tokens, descriptors and text do not align")
        if not 0.0 < float(logit_scale) <= 100.0:
            raise ValueError("logit_scale must lie in (0,100]")
        positive_cosine = torch.einsum("bkd,qd->bkq", descriptors, text)
        negative_cosine = torch.einsum(
            "bkd,nd->bkn", descriptors, negative_text
        )
        hardest_negative = negative_cosine.amax(dim=-1, keepdim=True)
        scores = torch.sigmoid(
            float(logit_scale) * (positive_cosine - hardest_negative)
        )
        fallback = scores[:, :1, :]
        residual = scores[:, 1:, :]
        fallback_expanded = fallback.expand(-1, self.residual_slots, -1)
        descriptor_agreement = torch.einsum(
            "bkd,bd->bk", descriptors[:, 1:], descriptors[:, 0]
        )
        token_norm = tokens.norm(dim=-1).clamp_min(1e-8)
        log_norm_ratio = (
            token_norm[:, 1:].log() - token_norm[:, :1].log()
        )
        residual_best = residual.amax(dim=1, keepdim=True)
        residual_std = residual.std(dim=1, unbiased=False, keepdim=True)
        batch, slots, queries = residual.shape
        features = torch.stack(
            [
                fallback_expanded,
                residual,
                residual - fallback_expanded,
                descriptor_agreement[:, :, None].expand(-1, -1, queries),
                log_norm_ratio[:, :, None].expand(-1, -1, queries),
                residual - residual_best,
                residual_std.expand(-1, slots, -1),
            ],
            dim=-1,
        )
        residual_logits = self.scorer(features).squeeze(-1)
        residual_attention = torch.softmax(residual_logits, dim=1)
        aggregate = (residual_attention[..., None] * features).sum(dim=1)
        raw_gate = self.gate(aggregate).squeeze(-1)
        residual_gate = (
            F.softplus(raw_gate) - torch.log(
                torch.tensor(2.0, device=raw_gate.device, dtype=raw_gate.dtype)
            )
        ).clamp(0.0, 1.0)
        residual_weights = residual_gate[:, None, :] * residual_attention
        weights = torch.cat(
            [1.0 - residual_gate[:, None, :], residual_weights], dim=1
        )
        response = (weights * scores).sum(dim=1)
        return SurfaceRegionQueryRouterOutput(
            response=response,
            slot_weights=weights,
            slot_scores=scores,
            residual_gate=residual_gate,
        )

    def architecture(self) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": SURFACE_REGION_QUERY_ROUTER_V1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "residual_slots": self.residual_slots,
            "total_slots": self.total_slots,
            "codebook_sha256": self.codebook_sha256,
            "score_contract": "canonical_negative_bernoulli_logit_scale_10",
            "slot_parameterization": "shared_permutation_equivariant_attention",
            "gate_parameterization": "exact_zero_softplus_offset_clamped",
            "initial_fallback_mass": "1.0_exact",
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["SurfaceRegionQueryRouterV1", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("invalid SurfaceRegion query-router checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        model = cls(
            hidden_dim=int(architecture["hidden_dim"]),
            residual_slots=int(architecture["residual_slots"]),
            codebook_sha256=str(architecture["codebook_sha256"]),
        )
        if model.architecture()["digest"] != expected:
            raise ValueError("SurfaceRegion query-router architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        return model, payload
