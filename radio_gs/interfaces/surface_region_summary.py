"""Global 3-D surface-region readout for official RADIO summary tokens.

The readout is intentionally query-free and scene-independent.  It consumes
only canonical primitive state and predicts a genuine 1280-D RADIO summary
token, which must subsequently be passed through the frozen official
``siglip2-g`` summary head.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
from torch import nn

from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    stable_descriptor_load,
)


SURFACE_GEOMETRY_DIM = 12
SURFACE_GEOMETRY_V2_DIM = 14
SURFACE_GEOMETRY_V3_DIM = 16
SURFACE_GEOMETRY_V3_LEARNED_DIM = 15
JOINT_CONTEXT_POOLING = "joint_attention_v1"
SEPARATE_CONTEXT_POOLING = "core_context_separate_attention_v1"
SURFACE_CODEBOOK_V3 = "surface_region_summary_codebook_v3"
SURFACE_RESIDUAL_CODEBOOK_V1 = "surface_region_frozen_v2_residual_codebook_v1"
SURFACE_SUMMARY_READOUT_V3 = "surface_region_summary_readout_v3"
SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION = 7
SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION = 8
SURFACE_REGION_V3_FEATURE_GAUGE = "l2_direction_plus_log_raw_norm_v1"
SURFACE_REGION_V3_LEGACY_RAW_BASE = "fixed_raw_base_with_anchor_mix_v1"
SURFACE_REGION_V3_GATED_RAW_PRIOR = "learned_residual_plus_gated_raw_pool_v1"
SURFACE_REGION_V3_GATED_RAW_PRIOR_INITIAL_WEIGHT = 0.05
SURFACE_SUMMARY_READOUT_V4 = "surface_region_summary_readout_v4"
SURFACE_SUMMARY_READOUT_V4_SCHEMA_VERSION = 9
SURFACE_REGION_V4_IMMUTABLE_V2_FALLBACK = (
    "accepted_v2_raw_gauge_immutable_fallback_v1"
)
SURFACE_REGION_V4_RESIDUAL_DISABLED = "disabled_exact_fallback_v1"

# These four digests are the immutable authority for the accepted global V2
# readout.  A V4 loader may accept a different explicitly supplied authority
# for isolated tests, but its default path cannot silently substitute another
# file, architecture, tensor state, or provenance record.
ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256 = (
    "5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb"
)
ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256 = (
    "1c608eb736074484ae42f5b51fad9a8f15b945bb9bf9f6e6304c861ce5bef05d"
)
ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256 = (
    "9e2118589fb13b9834ebaebf8409fa5761702aefbbc16d24496ef8447d0762fa"
)
ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256 = (
    "d3359edcf2d3e0e2c531205ecd450aeca93ec1a6c9ae046aaeaa5f5697fb6d0d"
)
ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256 = (
    "ac77e31694ebe796befcc725ea60685ad6f97978a9a903e1029aa7a7a05abc07"
)


class SurfaceRegionCodebookOutput(NamedTuple):
    """Query-independent canonical token plus latent region hypotheses."""

    canonical_token: torch.Tensor
    slot_tokens: torch.Tensor
    slot_priors: torch.Tensor


def surface_region_geometry(
    xyz: torch.Tensor,
    primitive_scale: torch.Tensor,
    opacity: torch.Tensor,
    reliability: torch.Tensor,
    region_radius: torch.Tensor | float,
    *,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build dimensionless, translation-invariant per-token geometry.

    Layout: relative xyz/radius (3), log primitive scale/radius (3), opacity,
    reliability, and a four-frequency physical-scale embedding (4).
    """

    points = torch.as_tensor(xyz).float()
    scales = torch.as_tensor(primitive_scale).float()
    alpha = torch.as_tensor(opacity).float()
    confidence = torch.as_tensor(reliability).float()
    squeeze = points.ndim == 2
    if squeeze:
        points, scales = points[None], scales[None]
        alpha, confidence = alpha[None], confidence[None]
    if points.ndim != 3 or points.shape[-1] != 3 or scales.shape != points.shape:
        raise ValueError("xyz and primitive_scale must align as [B,T,3]")
    if alpha.shape != (*points.shape[:2], 1) or confidence.shape != alpha.shape:
        raise ValueError("opacity and reliability must align as [B,T,1]")
    mask = (
        torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if token_mask is None
        else torch.as_tensor(token_mask, device=points.device).bool()
    )
    if mask.shape != points.shape[:2] or not bool(mask.any(dim=1).all()):
        raise ValueError("token_mask must keep at least one token per region")
    radius = torch.as_tensor(region_radius, device=points.device).float()
    if radius.ndim == 0:
        radius = radius.expand(points.shape[0])
    radius = radius.reshape(-1)
    if radius.shape != (points.shape[0],) or bool((radius <= 0).any()):
        raise ValueError("region_radius must contain one positive value per region")
    weights = mask.float()
    center = (points * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)
    normalized_xyz = (points - center[:, None]) / radius[:, None, None]
    normalized_scale = torch.log(
        scales.clamp_min(1e-6) / radius[:, None, None]
    ).clamp(-8.0, 4.0)
    log_radius = torch.log(radius.clamp_min(1e-6))[:, None]
    scale_embedding = torch.cat(
        [torch.sin(log_radius), torch.cos(log_radius),
         torch.sin(2.0 * log_radius), torch.cos(2.0 * log_radius)], dim=-1
    )[:, None].expand(-1, points.shape[1], -1)
    result = torch.cat(
        [normalized_xyz, normalized_scale, alpha.clamp(0, 1),
         confidence.clamp(0, 1), scale_embedding], dim=-1
    )
    result = result.masked_fill(~mask[..., None], 0.0)
    return result[0] if squeeze else result


def surface_region_geometry_v2(
    xyz: torch.Tensor,
    primitive_scale: torch.Tensor,
    reliability: torch.Tensor,
    region_radius: torch.Tensor | float,
    *,
    anchor_index: torch.Tensor | int,
    core_mask: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build anchor-relative geometry for ``SurfaceRegionContractV2``.

    Layout: relative xyz/radius (3), log local graph sigma/radius (3),
    reliability (1), anchor/core/context flags (3), and a four-frequency
    physical-scale embedding (4).  Opacity is absent because v1 supplied a
    constant-one channel in both domains.
    """

    points = torch.as_tensor(xyz).float()
    scales = torch.as_tensor(primitive_scale, device=points.device).float()
    confidence = torch.as_tensor(reliability, device=points.device).float()
    squeeze = points.ndim == 2
    if squeeze:
        points, scales, confidence = points[None], scales[None], confidence[None]
        core_mask = torch.as_tensor(core_mask)[None]
        if token_mask is not None:
            token_mask = torch.as_tensor(token_mask)[None]
    if points.ndim != 3 or points.shape[-1] != 3 or scales.shape != points.shape:
        raise ValueError("xyz and primitive_scale must align as [B,T,3]")
    if confidence.shape != (*points.shape[:2], 1):
        raise ValueError("reliability must align as [B,T,1]")
    mask = (
        torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if token_mask is None
        else torch.as_tensor(token_mask, device=points.device).bool()
    )
    core = torch.as_tensor(core_mask, device=points.device).bool()
    if core.shape != mask.shape or not bool((core & mask).any(dim=1).all()):
        raise ValueError("core_mask must keep a core token in every region")
    anchor = torch.as_tensor(anchor_index, device=points.device).long().reshape(-1)
    if anchor.numel() == 1:
        anchor = anchor.expand(points.shape[0])
    if anchor.shape != (points.shape[0],) or bool((anchor < 0).any()) or bool(
        (anchor >= points.shape[1]).any()
    ):
        raise ValueError("anchor_index must identify one token per region")
    batch = torch.arange(points.shape[0], device=points.device)
    if not bool(mask[batch, anchor].all()) or not bool(core[batch, anchor].all()):
        raise ValueError("the anchor must be a valid core token")
    radius = torch.as_tensor(region_radius, device=points.device).float().reshape(-1)
    if radius.numel() == 1:
        radius = radius.expand(points.shape[0])
    if radius.shape != (points.shape[0],) or bool((radius <= 0).any()):
        raise ValueError("region_radius must contain one positive value per region")
    center = points[batch, anchor]
    normalized_xyz = (points - center[:, None]) / radius[:, None, None]
    normalized_scale = torch.log(
        scales.clamp_min(1e-6) / radius[:, None, None]
    ).clamp(-8.0, 4.0)
    log_radius = torch.log(radius.clamp_min(1e-6))[:, None]
    scale_embedding = torch.cat(
        [torch.sin(log_radius), torch.cos(log_radius),
         torch.sin(2.0 * log_radius), torch.cos(2.0 * log_radius)], dim=-1
    )[:, None].expand(-1, points.shape[1], -1)
    anchor_flag = torch.zeros_like(mask)
    anchor_flag[batch, anchor] = True
    context = mask & ~core
    result = torch.cat(
        [normalized_xyz, normalized_scale, confidence.clamp(0, 1),
         anchor_flag[..., None].float(), core[..., None].float(),
         context[..., None].float(), scale_embedding], dim=-1,
    ).masked_fill(~mask[..., None], 0.0)
    return result[0] if squeeze else result


def surface_region_effective_reliability_v3(
    primitive_reliability: torch.Tensor,
    recovery_distance: torch.Tensor,
    region_radius: torch.Tensor | float,
    *,
    support_fill_mask: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the parameter-free V3 support-fill reliability policy.

    Real semantic members keep their primitive reliability exactly.  Selected
    support-fill rows receive ``r * exp(-recovery_distance / radius)`` and
    tensor padding receives exact zero.  Recovery support remains a real token
    (its token mask is true); this helper never conflates it with padding.
    """

    confidence = torch.as_tensor(primitive_reliability).float()
    squeeze = confidence.ndim == 2
    if squeeze:
        confidence = confidence[None]
        support_fill_mask = torch.as_tensor(support_fill_mask)[None]
        recovery_distance = torch.as_tensor(recovery_distance)[None]
        if token_mask is not None:
            token_mask = torch.as_tensor(token_mask)[None]
    if confidence.ndim != 3 or confidence.shape[-1] != 1:
        raise ValueError("primitive_reliability must align as [B,T,1]")
    mask = (
        torch.ones(confidence.shape[:2], dtype=torch.bool, device=confidence.device)
        if token_mask is None
        else torch.as_tensor(token_mask, device=confidence.device).bool()
    )
    support_fill = torch.as_tensor(
        support_fill_mask, device=confidence.device
    ).bool()
    recovery = torch.as_tensor(
        recovery_distance, device=confidence.device
    ).float()
    if recovery.ndim == 3 and recovery.shape[-1] == 1:
        recovery = recovery[..., 0]
    if mask.shape != confidence.shape[:2] or support_fill.shape != mask.shape:
        raise ValueError("V3 reliability masks must align with region tokens")
    if recovery.shape != mask.shape:
        raise ValueError("recovery_distance must align with region tokens")
    if bool((support_fill & ~mask).any()):
        raise ValueError("support_fill_mask must be a subset of token_mask")
    active_confidence = confidence[..., 0][mask]
    if (
        not bool(torch.isfinite(active_confidence).all())
        or bool((active_confidence < 0).any())
        or bool((active_confidence > 1).any())
    ):
        raise ValueError("active primitive reliability must be finite and lie in [0,1]")
    if bool(support_fill.any()):
        selected_recovery = recovery[support_fill]
        if not bool(torch.isfinite(selected_recovery).all()) or bool(
            (selected_recovery < 0).any()
        ):
            raise ValueError("support-fill recovery distance must be finite and non-negative")
    radius = torch.as_tensor(region_radius, device=confidence.device).float().reshape(-1)
    if radius.numel() == 1:
        radius = radius.expand(confidence.shape[0])
    if (
        radius.shape != (confidence.shape[0],)
        or not bool(torch.isfinite(radius).all())
        or bool((radius <= 0).any())
    ):
        raise ValueError("region_radius must contain one positive finite value per region")
    attenuation = torch.ones_like(recovery)
    if bool(support_fill.any()):
        normalized_distance = recovery / radius[:, None]
        attenuation[support_fill] = torch.exp(-normalized_distance[support_fill])
    result = confidence * attenuation[..., None]
    result = result.masked_fill(~mask[..., None], 0.0)
    return result[0] if squeeze else result


def surface_region_geometry_v3(
    xyz: torch.Tensor,
    primitive_scale: torch.Tensor,
    reliability: torch.Tensor,
    region_radius: torch.Tensor | float,
    *,
    raw_radio_l2_norm: torch.Tensor,
    anchor_index: torch.Tensor | int,
    core_mask: torch.Tensor,
    context_mask: torch.Tensor,
    support_fill_mask: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the 16-D geometry for ``SurfaceRegionContractV3``.

    Indices 0--13 preserve geometry-v2 exactly: relative xyz (0:3), log local
    scale (3:6), effective reliability (6), anchor/core/context flags (7:10),
    and the physical-radius embedding (10:14).  Index 14 is the explicit
    support-fill flag and index 15 is ``log(raw RADIO L2 norm)``.  Raw RADIO
    norms must be positive and finite on every selected token; tensor padding
    is ignored as input and emitted as an exact all-zero row.
    """

    points = torch.as_tensor(xyz).float()
    scales = torch.as_tensor(primitive_scale, device=points.device).float()
    confidence = torch.as_tensor(reliability, device=points.device).float()
    raw_norm = torch.as_tensor(raw_radio_l2_norm, device=points.device).float()
    squeeze = points.ndim == 2
    if squeeze:
        points, scales, confidence = points[None], scales[None], confidence[None]
        raw_norm = raw_norm[None]
        core_mask = torch.as_tensor(core_mask)[None]
        context_mask = torch.as_tensor(context_mask)[None]
        support_fill_mask = torch.as_tensor(support_fill_mask)[None]
        if token_mask is not None:
            token_mask = torch.as_tensor(token_mask)[None]
    if points.ndim != 3 or points.shape[-1] != 3 or scales.shape != points.shape:
        raise ValueError("xyz and primitive_scale must align as [B,T,3]")
    if confidence.shape != (*points.shape[:2], 1):
        raise ValueError("reliability must align as [B,T,1]")
    if raw_norm.shape == points.shape[:2]:
        raw_norm = raw_norm[..., None]
    if raw_norm.shape != (*points.shape[:2], 1):
        raise ValueError("raw_radio_l2_norm must align as [B,T,1]")
    mask = (
        torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if token_mask is None
        else torch.as_tensor(token_mask, device=points.device).bool()
    )
    core = torch.as_tensor(core_mask, device=points.device).bool()
    context = torch.as_tensor(context_mask, device=points.device).bool()
    support_fill = torch.as_tensor(
        support_fill_mask, device=points.device
    ).bool()
    if (
        mask.shape != points.shape[:2]
        or core.shape != mask.shape
        or context.shape != mask.shape
        or support_fill.shape != mask.shape
    ):
        raise ValueError("V3 geometry masks must align with region tokens")
    memberships = core.to(torch.int8) + context.to(torch.int8) + support_fill.to(torch.int8)
    if not bool((memberships[mask] == 1).all()) or bool(memberships[~mask].any()):
        raise ValueError(
            "core/context/support-fill masks must be mutually exclusive and unite to token_mask"
        )
    anchor = torch.as_tensor(anchor_index, device=points.device).long().reshape(-1)
    if anchor.numel() == 1:
        anchor = anchor.expand(points.shape[0])
    if anchor.shape != (points.shape[0],) or bool((anchor < 0).any()) or bool(
        (anchor >= points.shape[1]).any()
    ):
        raise ValueError("anchor_index must identify one token per region")
    batch = torch.arange(points.shape[0], device=points.device)
    if not bool(mask[batch, anchor].all()) or not bool(core[batch, anchor].all()):
        raise ValueError("the anchor must be a valid core token")
    selected_norm = raw_norm[..., 0][mask]
    if not bool(torch.isfinite(selected_norm).all()) or bool((selected_norm <= 0).any()):
        raise ValueError("active raw RADIO L2 norms must be positive and finite")
    radius = torch.as_tensor(region_radius, device=points.device).float().reshape(-1)
    if radius.numel() == 1:
        radius = radius.expand(points.shape[0])
    if (
        radius.shape != (points.shape[0],)
        or not bool(torch.isfinite(radius).all())
        or bool((radius <= 0).any())
    ):
        raise ValueError("region_radius must contain one positive finite value per region")
    center = points[batch, anchor]
    normalized_xyz = (points - center[:, None]) / radius[:, None, None]
    normalized_scale = torch.log(
        scales.clamp_min(1e-6) / radius[:, None, None]
    ).clamp(-8.0, 4.0)
    log_radius = torch.log(radius)[:, None]
    scale_embedding = torch.cat(
        [torch.sin(log_radius), torch.cos(log_radius),
         torch.sin(2.0 * log_radius), torch.cos(2.0 * log_radius)], dim=-1
    )[:, None].expand(-1, points.shape[1], -1)
    anchor_flag = torch.zeros_like(mask)
    anchor_flag[batch, anchor] = True
    result = torch.cat(
        [normalized_xyz, normalized_scale, confidence.clamp(0, 1),
         anchor_flag[..., None].float(), core[..., None].float(),
         context[..., None].float(), scale_embedding,
         support_fill[..., None].float(), torch.log(raw_norm)],
        dim=-1,
    ).masked_fill(~mask[..., None], 0.0)
    return result[0] if squeeze else result


class SurfaceRegionSummaryReadout(nn.Module):
    """Low-capacity permutation-invariant 3-D region summary readout."""

    def __init__(
        self,
        feature_dim: int = 1280,
        geometry_dim: int = SURFACE_GEOMETRY_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = int(geometry_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            nn.Linear(self.geometry_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(radio_features).float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
            if reliability is not None:
                reliability = torch.as_tensor(reliability)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the region token set")
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else torch.as_tensor(token_mask, device=values.device).bool()
        )
        if mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("token_mask must keep at least one token per region")
        hidden = self.feature_encoder(values) + self.geometry_encoder(geom)
        logits = self.attention(hidden).squeeze(-1)
        if reliability is not None:
            confidence = torch.as_tensor(reliability, device=values.device).float()
            if confidence.ndim == 3 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            if confidence.shape != values.shape[:2]:
                raise ValueError("reliability must align with region tokens")
            logits = logits + confidence.clamp_min(1e-4).log()
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        raw_mean = torch.einsum("bt,btc->bc", weights, values)
        pooled = torch.einsum("bt,bth->bh", weights, hidden)
        output = raw_mean + self.residual(pooled)
        return output[0] if squeeze else output

    def architecture(self) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": "surface_region_summary_readout_v1",
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "hidden_dim": self.hidden_dim,
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> tuple["SurfaceRegionSummaryReadout", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
            raise ValueError("invalid surface-region summary checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        model = cls(
            feature_dim=int(architecture["feature_dim"]),
            geometry_dim=int(architecture["geometry_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
        )
        if model.architecture()["digest"] != expected:
            raise ValueError("surface-region architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, payload


class SurfaceRegionSummaryReadoutV2(nn.Module):
    """Anchor/core-context conditioned low-capacity region readout."""

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 128,
        reliability_attention_mode: str = "log_prior",
        context_pooling_mode: str = JOINT_CONTEXT_POOLING,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = SURFACE_GEOMETRY_V2_DIM
        self.hidden_dim = int(hidden_dim)
        self.reliability_attention_mode = str(reliability_attention_mode)
        self.context_pooling_mode = str(context_pooling_mode)
        if self.reliability_attention_mode not in {"log_prior", "input_only"}:
            raise ValueError(
                "reliability_attention_mode must be log_prior or input_only"
            )
        if self.context_pooling_mode not in {
            JOINT_CONTEXT_POOLING,
            SEPARATE_CONTEXT_POOLING,
        }:
            raise ValueError(
                "context_pooling_mode must be joint_attention_v1 or "
                "core_context_separate_attention_v1"
            )
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, self.hidden_dim)
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.geometry_dim), nn.Linear(self.geometry_dim, self.hidden_dim),
            nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.geometry_dim),
            nn.Linear(self.feature_dim + self.geometry_dim, self.hidden_dim),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Linear(self.hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_context(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        return output

    def forward_with_context(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the official summary token and its pooled context.

        The first element is bitwise identical to :meth:`forward`.  Exposing
        the existing pooled hidden state lets optional downstream heads add a
        descriptor-space residual without changing the summary-token path or
        adding parameters to this checkpoint-compatible module.
        """
        values = torch.as_tensor(radio_features).float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
            if reliability is not None:
                reliability = torch.as_tensor(reliability)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the region token set")
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else torch.as_tensor(token_mask, device=values.device).bool()
        )
        anchor = torch.as_tensor(anchor_index, device=values.device).long().reshape(-1)
        if anchor.numel() == 1:
            anchor = anchor.expand(values.shape[0])
        batch = torch.arange(values.shape[0], device=values.device)
        if anchor.shape != (values.shape[0],) or not bool(mask[batch, anchor].all()):
            raise ValueError("anchor_index must identify a valid token per region")
        hidden = self.feature_encoder(values) + self.geometry_encoder(geom)
        query = self.query_encoder(
            torch.cat([values[batch, anchor], geom[batch, anchor]], dim=-1)
        )
        logits = torch.einsum("bh,bth->bt", query, self.key(hidden)) / self.hidden_dim**0.5
        if (
            reliability is not None
            and self.reliability_attention_mode == "log_prior"
        ):
            confidence = torch.as_tensor(reliability, device=values.device).float()
            if confidence.ndim == 3 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            if confidence.shape != mask.shape:
                raise ValueError("reliability must align with region tokens")
            logits = logits + confidence.clamp_min(1e-4).log()
        anchor_feature = values[batch, anchor]
        if self.context_pooling_mode == JOINT_CONTEXT_POOLING:
            weights = self._masked_attention(logits, mask)
            raw_mean = torch.einsum("bt,btc->bc", weights, values)
            base = raw_mean + 0.25 * (anchor_feature - raw_mean)
            pooled = torch.einsum("bt,bth->bh", weights, hidden) + query
        else:
            # Geometry-v2 channels 8/9 are the mutually exclusive core and
            # context flags.  Normalizing their attention streams separately
            # prevents a larger candidate pool from changing the core base
            # merely by adding context tokens.  Context remains an O(1)
            # conditioning stream through the learned residual.
            core = mask & (geom[..., 8] > 0.5)
            context = mask & (geom[..., 9] > 0.5)
            if (
                not bool(core.any(dim=1).all())
                or bool((core & context).any())
                or not torch.equal(core | context, mask)
                or not bool(core[batch, anchor].all())
            ):
                raise ValueError(
                    "separate context pooling requires valid core/context flags "
                    "and a core anchor"
                )
            core_weights = self._masked_attention(logits, core)
            context_weights = self._masked_attention(logits, context)
            core_mean = torch.einsum("bt,btc->bc", core_weights, values)
            base = core_mean + 0.25 * (anchor_feature - core_mean)
            core_hidden = torch.einsum("bt,bth->bh", core_weights, hidden)
            context_hidden = torch.einsum(
                "bt,bth->bh", context_weights, hidden
            )
            pooled = core_hidden + context_hidden + query
        output = base + self.residual(pooled)
        if squeeze:
            return output[0], pooled[0]
        return output, pooled

    @staticmethod
    def _masked_attention(
        logits: torch.Tensor,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """Return row-normalized weights, or an exact zero row if empty."""

        active = selection.any(dim=1, keepdim=True)
        masked = logits.masked_fill(
            ~selection,
            torch.finfo(logits.dtype).min,
        )
        return torch.softmax(masked, dim=1) * active.to(logits.dtype)

    def architecture(self, contract_sha256: str) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": "surface_region_summary_readout_v2",
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "hidden_dim": self.hidden_dim,
            "anchor_conditioned": "true",
            "core_context_conditioned": "true",
            "contract_sha256": str(contract_sha256),
        }
        if self.reliability_attention_mode != "log_prior":
            payload["reliability_attention_mode"] = (
                self.reliability_attention_mode
            )
        if self.context_pooling_mode != JOINT_CONTEXT_POOLING:
            payload["context_pooling_mode"] = self.context_pooling_mode
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> tuple["SurfaceRegionSummaryReadoutV2", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
            raise ValueError("invalid v2 surface-region summary checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        model = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            reliability_attention_mode=str(
                architecture.get("reliability_attention_mode", "log_prior")
            ),
            context_pooling_mode=str(
                architecture.get(
                    "context_pooling_mode",
                    JOINT_CONTEXT_POOLING,
                )
            ),
        )
        if model.architecture(str(architecture["contract_sha256"]))["digest"] != expected:
            raise ValueError("v2 surface-region architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        return model, payload


class SurfaceRegionSummaryReadoutV3(nn.Module):
    """Gauge-explicit joint-attention readout for contract V3.

    Active ``radio_features`` are required to be unit L2 directions.  Their
    original magnitude is a separate geometry-v3 side channel used only to
    reconstruct the raw-gauge base.  The learned geometry/query encoders see
    indices ``0:15`` and therefore cannot use magnitude a second time.  V3
    reliability is likewise input-only through geometry index 6; no log prior
    is accepted or applied by attention.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 128,
        reliability_attention_mode: str = "input_only",
        context_pooling_mode: str = JOINT_CONTEXT_POOLING,
        base_output_mode: str = SURFACE_REGION_V3_LEGACY_RAW_BASE,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = SURFACE_GEOMETRY_V3_DIM
        self.learned_geometry_dim = SURFACE_GEOMETRY_V3_LEARNED_DIM
        self.hidden_dim = int(hidden_dim)
        self.reliability_attention_mode = str(reliability_attention_mode)
        self.context_pooling_mode = str(context_pooling_mode)
        self.base_output_mode = str(base_output_mode)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if self.reliability_attention_mode != "input_only":
            raise ValueError("V3 reliability_attention_mode is fixed to input_only")
        if self.context_pooling_mode != JOINT_CONTEXT_POOLING:
            raise ValueError("V3 context_pooling_mode is fixed to joint_attention_v1")
        if self.base_output_mode not in {
            SURFACE_REGION_V3_LEGACY_RAW_BASE,
            SURFACE_REGION_V3_GATED_RAW_PRIOR,
        }:
            raise ValueError("unsupported V3 base_output_mode")
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.learned_geometry_dim),
            nn.Linear(self.learned_geometry_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.learned_geometry_dim),
            nn.Linear(
                self.feature_dim + self.learned_geometry_dim,
                self.hidden_dim,
            ),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        if self.base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR:
            initial = SURFACE_REGION_V3_GATED_RAW_PRIOR_INITIAL_WEIGHT
            self.raw_prior_gate_logit = nn.Parameter(
                torch.tensor(math.log(initial / (1.0 - initial)))
            )

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_context(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        return output

    def forward_with_context(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The argument is retained only for interface compatibility.  The
        # architecture contract guarantees it cannot become a second prior.
        del reliability
        source_values = torch.as_tensor(radio_features)
        values = source_values.float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the V3 region token set")
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else torch.as_tensor(token_mask, device=values.device).bool()
        )
        if mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("token_mask must keep at least one token per region")
        if not bool(torch.isfinite(values[mask]).all()):
            raise ValueError("active RADIO directions must be finite")
        active_norm = torch.linalg.vector_norm(values, dim=-1)[mask]
        # Training and deployment caches intentionally store RADIO directions
        # in fp16.  Re-quantizing a unit vector can move its reconstructed L2
        # norm by up to about 5e-4 (and by 2.15e-4 in the Figurines field)
        # without changing the declared direction gauge.  Keep float32 callers
        # as strict as before while admitting only this bounded storage error.
        direction_atol = (
            5e-4
            if source_values.dtype in {torch.float16, torch.bfloat16}
            else 2e-4
        )
        if not torch.allclose(
            active_norm,
            torch.ones_like(active_norm),
            rtol=0.0,
            atol=direction_atol,
        ):
            raise ValueError("active radio_features must use the unit L2 direction gauge")
        if not bool(torch.isfinite(geom[mask]).all()) or bool(
            torch.count_nonzero(geom[~mask])
        ):
            raise ValueError("V3 geometry must be finite on tokens and exactly zero on padding")
        core = geom[..., 8] > 0.5
        context = geom[..., 9] > 0.5
        support_fill = geom[..., 14] > 0.5
        membership_count = (
            core.to(torch.int8) + context.to(torch.int8) + support_fill.to(torch.int8)
        )
        if not bool((membership_count[mask] == 1).all()) or bool(
            membership_count[~mask].any()
        ):
            raise ValueError(
                "V3 geometry requires disjoint core/context/support-fill flags"
            )
        anchor = torch.as_tensor(anchor_index, device=values.device).long().reshape(-1)
        if anchor.numel() == 1:
            anchor = anchor.expand(values.shape[0])
        if anchor.shape != (values.shape[0],) or bool((anchor < 0).any()) or bool(
            (anchor >= values.shape[1]).any()
        ):
            raise ValueError("anchor_index must identify one token per region")
        batch = torch.arange(values.shape[0], device=values.device)
        if not bool(mask[batch, anchor].all()) or not bool(core[batch, anchor].all()):
            raise ValueError("the V3 anchor must be a valid core token")
        direction = values.masked_fill(~mask[..., None], 0.0)
        amplitude = torch.exp(geom[..., 15])
        if not bool(torch.isfinite(amplitude[mask]).all()) or bool(
            (amplitude[mask] <= 0).any()
        ):
            raise ValueError("V3 log raw RADIO norm must reconstruct a positive finite amplitude")
        gauge_values = direction * amplitude[..., None]
        gauge_values = gauge_values.masked_fill(~mask[..., None], 0.0)
        learned_geometry = geom[..., :self.learned_geometry_dim]
        hidden = (
            self.feature_encoder(direction)
            + self.geometry_encoder(learned_geometry)
        )
        query = self.query_encoder(
            torch.cat(
                [
                    direction[batch, anchor],
                    learned_geometry[batch, anchor],
                ],
                dim=-1,
            )
        )
        logits = torch.einsum("bh,bth->bt", query, self.key(hidden))
        logits = logits / math.sqrt(float(self.hidden_dim))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        raw_mean = torch.einsum("bt,btc->bc", weights, gauge_values)
        pooled = torch.einsum("bt,bth->bh", weights, hidden) + query
        learned_residual = self.residual(pooled)
        if self.base_output_mode == SURFACE_REGION_V3_LEGACY_RAW_BASE:
            anchor_feature = gauge_values[batch, anchor]
            base = raw_mean + 0.25 * (anchor_feature - raw_mean)
            output = base + learned_residual
        else:
            gate = torch.sigmoid(self.raw_prior_gate_logit).to(
                device=raw_mean.device,
                dtype=raw_mean.dtype,
            )
            output = learned_residual + gate * raw_mean
        if squeeze:
            return output[0], pooled[0]
        return output, pooled

    def architecture(self, contract_sha256: str) -> dict[str, float | int | str]:
        payload: dict[str, float | int | str] = {
            "name": SURFACE_SUMMARY_READOUT_V3,
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "learned_geometry_dim": self.learned_geometry_dim,
            "hidden_dim": self.hidden_dim,
            "anchor_conditioned": "true",
            "core_context_support_fill_conditioned": "true",
            "feature_normalization": SURFACE_REGION_V3_FEATURE_GAUGE,
            "raw_radio_l2_norm": "geometry_index_15_log_raw_l2_norm",
            "raw_radio_l2_norm_usage": "base_reconstruction_only_v1",
            "base_gauge_reconstruction": "direction_times_exp_log_raw_norm_v1",
            "reliability_attention_mode": "input_only",
            "context_pooling_mode": JOINT_CONTEXT_POOLING,
            "contract_sha256": str(contract_sha256),
        }
        if self.base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR:
            payload.update(
                {
                    "base_output_mode": self.base_output_mode,
                    "raw_radio_l2_norm_usage": (
                        "gated_pooled_raw_amplitude_prior_only_v1"
                    ),
                    "base_gauge_reconstruction": (
                        "attention_pooled_direction_times_exp_log_raw_norm_v1"
                    ),
                    "raw_amplitude_prior": "attention_weighted_raw_pool_v1",
                    "raw_amplitude_prior_anchor_mix": "none",
                    "raw_amplitude_prior_gate": (
                        "global_learned_sigmoid_scalar_v1"
                    ),
                    "raw_amplitude_prior_gate_initial_weight": (
                        SURFACE_REGION_V3_GATED_RAW_PRIOR_INITIAL_WEIGHT
                    ),
                    "output_composition": (
                        "learned_residual_plus_gated_raw_amplitude_prior_v1"
                    ),
                }
            )
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
    ) -> tuple["SurfaceRegionSummaryReadoutV3", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") not in {
            SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
            SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
        }:
            raise ValueError("invalid v3 surface-region summary checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        schema_version = int(payload["schema_version"])
        base_output_mode = str(
            architecture.get(
                "base_output_mode",
                SURFACE_REGION_V3_LEGACY_RAW_BASE,
            )
        )
        if (
            schema_version == SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION
            and base_output_mode != SURFACE_REGION_V3_LEGACY_RAW_BASE
        ) or (
            schema_version == SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION
            and base_output_mode != SURFACE_REGION_V3_GATED_RAW_PRIOR
        ):
            raise ValueError("V3 checkpoint schema/base-output mode mismatch")
        model = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            reliability_attention_mode=str(
                architecture["reliability_attention_mode"]
            ),
            context_pooling_mode=str(architecture["context_pooling_mode"]),
            base_output_mode=base_output_mode,
        )
        if model.architecture(str(architecture["contract_sha256"]))["digest"] != expected:
            raise ValueError("v3 surface-region architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        return model, payload


def _surface_region_tensor_sha256(value: torch.Tensor) -> str:
    """Hash checkpoint tensors independently of ``torch.save`` encoding."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("surface-region state tensor is non-finite")
        array = tensor.to(torch.float32).numpy().astype("<f4", copy=False)
        dtype = "float32-le"
    elif tensor.dtype == torch.bool:
        array = tensor.to(torch.uint8).numpy()
        dtype = "bool-u8"
    elif tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        array = tensor.to(torch.int64).numpy().astype("<i8", copy=False)
        dtype = "int64-le"
    else:
        raise ValueError(f"unsupported surface-region state dtype: {tensor.dtype}")
    header = json.dumps(
        {"dtype": dtype, "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def surface_region_state_dict_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:
    """Return a name-, shape-, dtype-, and value-bound state digest."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("surface-region state_dict must be a non-empty mapping")
    records: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise ValueError("surface-region state_dict fields differ")
        tensor = value.detach().cpu().contiguous()
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": _surface_region_tensor_sha256(tensor),
            }
        )
    return canonical_json_sha256(records)


def _validate_frozen_v2_provenance(
    provenance: object,
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("V4 base checkpoint lacks V2 provenance")
    record = dict(provenance)
    exact = {
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "custom_text_projection": False,
    }
    if any(record.get(key) is not value for key, value in exact.items()):
        raise ValueError("V4 base checkpoint V2 provenance is not frozen and clean")
    if record.get("training_scope") != "global_cross_scene_3d_surface_v2":
        raise ValueError("V4 base checkpoint has the wrong V2 training scope")
    if record.get("region_contract_sha256") != contract_sha256:
        raise ValueError("V4 base checkpoint provenance/contract digest mismatch")
    if record.get("official_summary_head") != "c-radio_v4 siglip2-g":
        raise ValueError("V4 base checkpoint has the wrong official summary head")
    split_scenes: list[set[str]] = []
    for split_name in ("train", "validation"):
        split = record.get(split_name)
        if not isinstance(split, Mapping):
            raise ValueError(f"V4 base checkpoint lacks {split_name} provenance")
        if split.get("region_contract_sha256") != contract_sha256:
            raise ValueError(
                f"V4 base checkpoint {split_name} contract digest mismatch"
            )
        scenes = split.get("scenes")
        if (
            not isinstance(scenes, (list, tuple))
            or not scenes
            or any(not isinstance(scene, str) or not scene for scene in scenes)
        ):
            raise ValueError(f"V4 base checkpoint has invalid {split_name} scenes")
        split_scenes.append(set(scenes))
    if split_scenes[0] & split_scenes[1]:
        raise ValueError("V4 base checkpoint train/validation scenes overlap")
    return record


def _surface_region_v2_from_payload(
    payload: object,
    *,
    expected_architecture_sha256: str,
    expected_state_dict_sha256: str,
    expected_provenance_sha256: str,
    expected_contract_sha256: str,
) -> tuple[SurfaceRegionSummaryReadoutV2, dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
        raise ValueError("invalid V4 base V2 checkpoint")
    raw_architecture = payload.get("architecture")
    if not isinstance(raw_architecture, Mapping):
        raise ValueError("V4 base checkpoint lacks a V2 architecture")
    architecture = dict(raw_architecture)
    if architecture.get("digest") != expected_architecture_sha256:
        raise ValueError("V4 base V2 architecture authority differs")
    if architecture.get("name") != "surface_region_summary_readout_v2":
        raise ValueError("V4 base checkpoint is not a V2 readout")
    if architecture.get("geometry_dim") != SURFACE_GEOMETRY_V2_DIM:
        raise ValueError("V4 base checkpoint has the wrong V2 geometry dimension")
    if architecture.get("contract_sha256") != expected_contract_sha256:
        raise ValueError("V4 base V2 contract authority differs")
    model = SurfaceRegionSummaryReadoutV2(
        feature_dim=int(architecture["feature_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        reliability_attention_mode=str(
            architecture.get("reliability_attention_mode", "log_prior")
        ),
        context_pooling_mode=str(
            architecture.get("context_pooling_mode", JOINT_CONTEXT_POOLING)
        ),
    )
    if model.architecture(expected_contract_sha256) != architecture:
        raise ValueError("V4 base V2 architecture digest mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("V4 base checkpoint lacks a V2 state_dict")
    if surface_region_state_dict_sha256(state) != expected_state_dict_sha256:
        raise ValueError("V4 base V2 state_dict authority differs")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    provenance = _validate_frozen_v2_provenance(
        payload.get("provenance"),
        contract_sha256=expected_contract_sha256,
    )
    if canonical_json_sha256(provenance) != expected_provenance_sha256:
        raise ValueError("V4 base V2 provenance authority differs")
    return model, architecture, provenance


class SurfaceRegionSummaryReadoutV4(nn.Module):
    """Exact immutable V2 fallback over the explicit V3/V4 input gauge.

    The wrapper contains no residual parameters and no OOD threshold.  It
    reconstructs raw RADIO values from unit directions and geometry index 15,
    removes support-fill rows from the V2 token mask, and delegates unchanged
    geometry indices 0:14 plus reliability index 6 to a frozen SHA-bound V2
    readout.  Consequently its output is exactly the accepted V2 output on
    that reconstructed token set; the wrapper has no mechanism that can make
    the accepted readout worse.

    A trainable residual is deliberately absent.  Introducing one requires a
    separately versioned architecture with a distribution-derived query-free
    gate and a proved norm bound; inventing a benchmark-tuned OOD threshold in
    this fail-safe class would violate its contract.
    """

    def __init__(
        self,
        base_readout: SurfaceRegionSummaryReadoutV2,
        *,
        base_checkpoint_sha256: str,
        base_architecture: Mapping[str, Any],
        base_state_dict_sha256: str,
        base_provenance: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(base_readout, SurfaceRegionSummaryReadoutV2):
            raise TypeError("V4 base_readout must be SurfaceRegionSummaryReadoutV2")
        if any(parameter.requires_grad for parameter in base_readout.parameters()):
            raise ValueError("V4 base_readout must already be frozen")
        architecture = dict(base_architecture)
        if base_readout.architecture(
            str(architecture.get("contract_sha256", ""))
        ) != architecture:
            raise ValueError("V4 base_readout and architecture differ")
        if surface_region_state_dict_sha256(
            base_readout.state_dict()
        ) != str(base_state_dict_sha256):
            raise ValueError("V4 base_readout and state authority differ")
        provenance = _validate_frozen_v2_provenance(
            base_provenance,
            contract_sha256=str(architecture["contract_sha256"]),
        )
        digests = (
            str(base_checkpoint_sha256),
            str(base_state_dict_sha256),
            str(architecture["digest"]),
            canonical_json_sha256(provenance),
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("V4 base authority must use lowercase SHA-256 digests")
        self.feature_dim = base_readout.feature_dim
        self.geometry_dim = SURFACE_GEOMETRY_V3_DIM
        self.hidden_dim = base_readout.hidden_dim
        self.base_readout = base_readout.eval().requires_grad_(False)
        self.base_checkpoint_sha256 = str(base_checkpoint_sha256)
        self.base_architecture = architecture
        self.base_state_dict_sha256 = str(base_state_dict_sha256)
        self.base_provenance = provenance
        self.base_provenance_sha256 = canonical_json_sha256(provenance)

    def train(self, mode: bool = True) -> "SurfaceRegionSummaryReadoutV4":
        super().train(mode)
        # LayerNorm currently makes train/eval numerically equivalent, but
        # keeping the immutable authority in eval mode prevents future V2
        # implementation changes from altering the fallback implicitly.
        self.base_readout.eval()
        return self

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.forward_with_context(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        return output

    def forward_with_context(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_values = torch.as_tensor(radio_features)
        values = source_values.float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
            if reliability is not None:
                reliability = torch.as_tensor(reliability)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("V4 radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the V4 region token set")
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else torch.as_tensor(token_mask, device=values.device).bool()
        )
        if mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("V4 token_mask must keep at least one token per region")
        if not bool(torch.isfinite(values[mask]).all()) or bool(
            torch.count_nonzero(values[~mask])
        ):
            raise ValueError("V4 directions must be finite and exactly zero on padding")
        direction_atol = (
            5e-4
            if source_values.dtype in {torch.float16, torch.bfloat16}
            else 2e-4
        )
        active_norm = torch.linalg.vector_norm(values, dim=-1)[mask]
        if not torch.allclose(
            active_norm,
            torch.ones_like(active_norm),
            rtol=0.0,
            atol=direction_atol,
        ):
            raise ValueError("V4 radio_features must use the unit L2 direction gauge")
        if not bool(torch.isfinite(geom[mask]).all()) or bool(
            torch.count_nonzero(geom[~mask])
        ):
            raise ValueError("V4 geometry must be finite and exactly zero on padding")
        core = geom[..., 8] > 0.5
        context = geom[..., 9] > 0.5
        support_fill = geom[..., 14] > 0.5
        membership_count = (
            core.to(torch.int8)
            + context.to(torch.int8)
            + support_fill.to(torch.int8)
        )
        if not bool((membership_count[mask] == 1).all()) or bool(
            membership_count[~mask].any()
        ):
            raise ValueError(
                "V4 geometry requires disjoint core/context/support-fill flags"
            )
        base_mask = mask & ~support_fill
        if not bool(base_mask.any(dim=1).all()):
            raise ValueError("V4 support-fill removal left an empty V2 region")
        anchor = torch.as_tensor(anchor_index, device=values.device).long().reshape(-1)
        if anchor.numel() == 1:
            anchor = anchor.expand(values.shape[0])
        if anchor.shape != (values.shape[0],) or bool((anchor < 0).any()) or bool(
            (anchor >= values.shape[1]).any()
        ):
            raise ValueError("anchor_index must identify one V4 token per region")
        batch = torch.arange(values.shape[0], device=values.device)
        if not bool(base_mask[batch, anchor].all()) or not bool(
            core[batch, anchor].all()
        ):
            raise ValueError("the V4 anchor must be a non-fill core token")
        amplitude = torch.exp(geom[..., 15])
        if not bool(torch.isfinite(amplitude[mask]).all()) or bool(
            (amplitude[mask] <= 0).any()
        ):
            raise ValueError("V4 log raw RADIO norm must reconstruct a positive finite amplitude")
        raw_values = (values * amplitude[..., None]).masked_fill(
            ~base_mask[..., None], 0.0
        )
        base_geometry = geom[..., :SURFACE_GEOMETRY_V2_DIM].masked_fill(
            ~base_mask[..., None], 0.0
        )
        base_reliability = geom[..., 6:7].masked_fill(
            ~base_mask[..., None], 0.0
        )
        active_reliability = base_reliability[..., 0][base_mask]
        if bool((active_reliability < 0).any()) or bool(
            (active_reliability > 1).any()
        ):
            raise ValueError("V4 reliability geometry must lie in [0,1]")
        if reliability is not None:
            supplied = torch.as_tensor(
                reliability, device=values.device
            ).float()
            if supplied.ndim == 2:
                supplied = supplied[..., None]
            if supplied.shape != base_reliability.shape or not torch.allclose(
                supplied[..., 0][base_mask],
                base_reliability[..., 0][base_mask],
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("V4 reliability must equal authoritative geometry index 6")
        output, pooled = self.base_readout.forward_with_context(
            raw_values,
            base_geometry,
            anchor_index=anchor,
            token_mask=base_mask,
            reliability=base_reliability,
        )
        if squeeze:
            return output[0], pooled[0]
        return output, pooled

    def architecture(self, contract_sha256: str) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": SURFACE_SUMMARY_READOUT_V4,
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "base_geometry_dim": SURFACE_GEOMETRY_V2_DIM,
            "hidden_dim": self.hidden_dim,
            "feature_normalization": SURFACE_REGION_V3_FEATURE_GAUGE,
            "raw_gauge_reconstruction": "direction_times_exp_geometry_index_15_v1",
            "support_fill_policy": "exclude_geometry_index_14_from_v2_base_v1",
            "reliability_authority": "geometry_index_6_v1",
            "output_mode": SURFACE_REGION_V4_IMMUTABLE_V2_FALLBACK,
            "residual_mode": SURFACE_REGION_V4_RESIDUAL_DISABLED,
            "ood_gate": "not_applicable_residual_disabled",
            "trainable_parameter_count": 0,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_architecture_sha256": str(self.base_architecture["digest"]),
            "base_state_dict_sha256": self.base_state_dict_sha256,
            "base_provenance_sha256": self.base_provenance_sha256,
            "base_contract_sha256": str(self.base_architecture["contract_sha256"]),
            "contract_sha256": str(contract_sha256),
        }
        payload["digest"] = canonical_json_sha256(payload)
        return payload

    def checkpoint_payload(self, contract_sha256: str) -> dict[str, Any]:
        return {
            "schema_version": SURFACE_SUMMARY_READOUT_V4_SCHEMA_VERSION,
            "architecture": self.architecture(contract_sha256),
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "base_authority": {
                "schema_version": 3,
                "checkpoint_sha256": self.base_checkpoint_sha256,
                "architecture": dict(self.base_architecture),
                "state_dict_sha256": self.base_state_dict_sha256,
                "provenance": dict(self.base_provenance),
            },
        }

    @classmethod
    def from_v2_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_architecture_sha256: str,
        expected_state_dict_sha256: str,
        expected_provenance_sha256: str,
        expected_contract_sha256: str,
        map_location: str | torch.device = "cpu",
    ) -> tuple["SurfaceRegionSummaryReadoutV4", Mapping[str, Any]]:
        payload, actual_sha256, _ = stable_descriptor_load(
            path,
            lambda handle: torch.load(handle, map_location=map_location),
            expected_sha256=expected_checkpoint_sha256,
            label="V4 immutable V2 base checkpoint",
        )
        base, architecture, provenance = _surface_region_v2_from_payload(
            payload,
            expected_architecture_sha256=expected_architecture_sha256,
            expected_state_dict_sha256=expected_state_dict_sha256,
            expected_provenance_sha256=expected_provenance_sha256,
            expected_contract_sha256=expected_contract_sha256,
        )
        model = cls(
            base,
            base_checkpoint_sha256=actual_sha256,
            base_architecture=architecture,
            base_state_dict_sha256=expected_state_dict_sha256,
            base_provenance=provenance,
        )
        model.eval().requires_grad_(False)
        return model, payload

    @classmethod
    def from_accepted_v2_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["SurfaceRegionSummaryReadoutV4", Mapping[str, Any]]:
        return cls.from_v2_checkpoint(
            path,
            expected_checkpoint_sha256=ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
            expected_architecture_sha256=ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
            expected_state_dict_sha256=ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
            expected_provenance_sha256=ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
            expected_contract_sha256=ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
            map_location=map_location,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_checkpoint_sha256: str | None = None,
        expected_base_checkpoint_sha256: str = (
            ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256
        ),
        expected_base_architecture_sha256: str = (
            ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256
        ),
        expected_base_state_dict_sha256: str = (
            ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256
        ),
        expected_base_provenance_sha256: str = (
            ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256
        ),
        expected_base_contract_sha256: str = (
            ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256
        ),
        map_location: str | torch.device = "cpu",
    ) -> tuple["SurfaceRegionSummaryReadoutV4", Mapping[str, Any]]:
        payload, _, _ = stable_descriptor_load(
            path,
            lambda handle: torch.load(handle, map_location=map_location),
            expected_sha256=expected_checkpoint_sha256,
            label="V4 surface-region summary checkpoint",
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version")
            != SURFACE_SUMMARY_READOUT_V4_SCHEMA_VERSION
        ):
            raise ValueError("invalid V4 surface-region summary checkpoint")
        authority = payload.get("base_authority")
        if not isinstance(authority, Mapping):
            raise ValueError("V4 checkpoint lacks immutable base authority")
        if authority.get("checkpoint_sha256") != expected_base_checkpoint_sha256:
            raise ValueError("V4 base checkpoint authority differs")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("V4 checkpoint lacks state_dict")
        prefix = "base_readout."
        if not state or any(not str(name).startswith(prefix) for name in state):
            raise ValueError("V4 checkpoint contains non-base state")
        base_state = {
            str(name)[len(prefix):]: value for name, value in state.items()
        }
        embedded_payload = {
            "schema_version": authority.get("schema_version"),
            "architecture": authority.get("architecture"),
            "state_dict": base_state,
            "provenance": authority.get("provenance"),
        }
        base, base_architecture, provenance = _surface_region_v2_from_payload(
            embedded_payload,
            expected_architecture_sha256=expected_base_architecture_sha256,
            expected_state_dict_sha256=expected_base_state_dict_sha256,
            expected_provenance_sha256=expected_base_provenance_sha256,
            expected_contract_sha256=expected_base_contract_sha256,
        )
        if authority.get("state_dict_sha256") != expected_base_state_dict_sha256:
            raise ValueError("V4 embedded base state authority differs")
        model = cls(
            base,
            base_checkpoint_sha256=expected_base_checkpoint_sha256,
            base_architecture=base_architecture,
            base_state_dict_sha256=expected_base_state_dict_sha256,
            base_provenance=provenance,
        )
        raw_architecture = payload.get("architecture")
        if not isinstance(raw_architecture, Mapping):
            raise ValueError("V4 checkpoint lacks architecture")
        architecture = dict(raw_architecture)
        contract_sha256 = str(architecture.get("contract_sha256", ""))
        if model.architecture(contract_sha256) != architecture:
            raise ValueError("V4 surface-region architecture digest mismatch")
        model.eval().requires_grad_(False)
        return model, payload


class SurfaceRegionSummaryCodebookV3(nn.Module):
    """Direction-identified core/context region codebook.

    The module preserves several query-independent summary hypotheses instead
    of forcing all valid views through one conditional mean.  Text queries may
    read the frozen hypotheses later, but never alter the persistent region
    state.  RADIO direction, geometric/reliability channels, and slot priors
    remain separate throughout the readout.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 256,
        slots: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = SURFACE_GEOMETRY_V2_DIM
        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not 2 <= self.slots <= 8:
            raise ValueError("surface codebook requires 2 to 8 slots")
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            nn.Linear(self.geometry_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.anchor_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.geometry_dim),
            nn.Linear(self.feature_dim + self.geometry_dim, self.hidden_dim),
        )
        self.key = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.slot_queries = nn.Parameter(torch.empty(self.slots, self.hidden_dim))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        self.context_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.prior = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del reliability  # Reliability is already an explicit geometry channel.
        return self.forward_codebook(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
        ).canonical_token

    def forward_codebook(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
    ) -> SurfaceRegionCodebookOutput:
        values = torch.as_tensor(radio_features).float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the region token set")
        mask = (
            torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            if token_mask is None
            else torch.as_tensor(token_mask, device=values.device).bool()
        )
        if mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("token_mask must keep at least one token per region")
        anchor = torch.as_tensor(anchor_index, device=values.device).long().reshape(-1)
        if anchor.numel() == 1:
            anchor = anchor.expand(values.shape[0])
        batch = torch.arange(values.shape[0], device=values.device)
        if anchor.shape != (values.shape[0],) or not bool(mask[batch, anchor].all()):
            raise ValueError("anchor_index must identify a valid token per region")

        # The V3 gauge is part of the model rather than a caller convention.
        direction = torch.nn.functional.normalize(values, dim=-1, eps=1e-8)
        direction = direction.masked_fill(~mask[..., None], 0.0)
        hidden = self.feature_encoder(direction) + self.geometry_encoder(geom)
        anchor_hidden = self.anchor_encoder(
            torch.cat([direction[batch, anchor], geom[batch, anchor]], dim=-1)
        )
        query = anchor_hidden[:, None, :] + self.slot_queries[None, :, :]
        logits = torch.einsum("bkh,bth->bkt", query, self.key(hidden))
        logits = logits / math.sqrt(float(self.hidden_dim))

        core = mask & (geom[..., 8] > 0.5)
        context = mask & (geom[..., 9] > 0.5)
        if (
            not bool(core.any(dim=1).all())
            or bool((core & context).any())
            or not torch.equal(core | context, mask)
            or not bool(core[batch, anchor].all())
        ):
            raise ValueError(
                "V3 requires disjoint core/context flags and a core anchor"
            )
        core_weights = self._slot_attention(logits, core)
        context_weights = self._slot_attention(logits, context)
        core_direction = torch.einsum("bkt,btd->bkd", core_weights, direction)
        anchor_direction = direction[batch, anchor][:, None, :]
        base = core_direction + 0.25 * (anchor_direction - core_direction)
        core_hidden = torch.einsum("bkt,bth->bkh", core_weights, hidden)
        context_hidden = torch.einsum("bkt,bth->bkh", context_weights, hidden)
        pooled = (
            core_hidden
            + self.context_gate(query) * context_hidden
            + query
        )
        slot_tokens = base + self.residual(pooled)
        slot_priors = torch.softmax(self.prior(pooled).squeeze(-1), dim=-1)
        canonical_token = torch.einsum("bk,bkd->bd", slot_priors, slot_tokens)
        output = SurfaceRegionCodebookOutput(
            canonical_token=canonical_token,
            slot_tokens=slot_tokens,
            slot_priors=slot_priors,
        )
        if squeeze:
            return SurfaceRegionCodebookOutput(*(value[0] for value in output))
        return output

    @staticmethod
    def _slot_attention(
        logits: torch.Tensor,
        selection: torch.Tensor,
    ) -> torch.Tensor:
        active = selection.any(dim=1, keepdim=True)
        masked = logits.masked_fill(
            ~selection[:, None, :],
            torch.finfo(logits.dtype).min,
        )
        return torch.softmax(masked, dim=-1) * active[:, :, None].to(logits.dtype)

    def architecture(self, contract_sha256: str) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": SURFACE_CODEBOOK_V3,
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "hidden_dim": self.hidden_dim,
            "slots": self.slots,
            "feature_normalization": "l2_direction_inside_model",
            "reliability_attention": "input_only_geometry_channel",
            "context_pooling": "core_base_context_conditioning_per_slot",
            "contract_sha256": str(contract_sha256),
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
    ) -> tuple["SurfaceRegionSummaryCodebookV3", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 4:
            raise ValueError("invalid v3 surface-region codebook checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        model = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            slots=int(architecture["slots"]),
        )
        if model.architecture(str(architecture["contract_sha256"]))["digest"] != expected:
            raise ValueError("v3 surface-region architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        return model, payload


class SurfaceRegionSummaryResidualCodebookV1(nn.Module):
    """Frozen-V2 canonical path plus query-independent residual hypotheses.

    Slot zero and :attr:`canonical_token` are the exact frozen V2 output on
    the caller-provided feature gauge.  Three additional hypotheses are
    conditioned on an L2-direction copy of the same region, then expressed as
    a tangent direction update and an explicit log-norm ratio.  This keeps the
    deployed V2 fallback intact while exposing multiview capacity through a
    separate codebook route.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 128,
        residual_slots: int = 3,
        reliability_attention_mode: str = "log_prior",
        context_pooling_mode: str = JOINT_CONTEXT_POOLING,
        control_sha256: str = "",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = SURFACE_GEOMETRY_V2_DIM
        self.hidden_dim = int(hidden_dim)
        self.residual_slots = int(residual_slots)
        self.slots = 1 + self.residual_slots
        self.reliability_attention_mode = str(reliability_attention_mode)
        self.context_pooling_mode = str(context_pooling_mode)
        self.control_sha256 = str(control_sha256)
        if self.residual_slots != 3:
            raise ValueError("residual codebook V1 has exactly three learned slots")
        if not self.control_sha256:
            raise ValueError("control_sha256 must bind the frozen V2 authority")
        self.base = SurfaceRegionSummaryReadoutV2(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            reliability_attention_mode=self.reliability_attention_mode,
            context_pooling_mode=self.context_pooling_mode,
        )
        self.base.eval().requires_grad_(False)
        self.slot_embeddings = nn.Parameter(
            torch.empty(self.residual_slots, self.hidden_dim)
        )
        nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim + 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def train(self, mode: bool = True) -> "SurfaceRegionSummaryResidualCodebookV1":
        super().train(mode)
        self.base.eval()
        return self

    def load_frozen_base_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict, strict=True)
        self.base.eval().requires_grad_(False)

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_codebook(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        ).canonical_token

    def forward_codebook(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> SurfaceRegionCodebookOutput:
        values = torch.as_tensor(radio_features).float()
        geom = torch.as_tensor(geometry, device=values.device).float()
        squeeze = values.ndim == 2
        if squeeze:
            values, geom = values[None], geom[None]
            if token_mask is not None:
                token_mask = torch.as_tensor(token_mask)[None]
            if reliability is not None:
                reliability = torch.as_tensor(reliability)[None]
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError("radio_features must be [B,T,feature_dim]")
        if geom.shape != (*values.shape[:2], self.geometry_dim):
            raise ValueError("geometry must align with the region token set")

        # Preserve the caller's raw gauge exactly on the canonical route.
        canonical, _ = self.base.forward_with_context(
            values,
            geom,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        # Condition only the additive hypotheses on a stable direction gauge.
        direction = torch.nn.functional.normalize(values, dim=-1, eps=1e-8)
        _, pooled = self.base.forward_with_context(
            direction,
            geom,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        query = pooled[:, None, :] + self.slot_embeddings[None, :, :]
        raw = self.residual_head(query)
        raw_direction = raw[..., : self.feature_dim]
        raw_log_norm = raw[..., self.feature_dim]

        base_norm = canonical.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        base_direction = canonical / base_norm
        tangent = raw_direction - (
            raw_direction * base_direction[:, None, :]
        ).sum(dim=-1, keepdim=True) * base_direction[:, None, :]
        candidate_direction = torch.nn.functional.normalize(
            base_direction[:, None, :] + tangent,
            dim=-1,
            eps=1e-8,
        )
        candidate_norm = base_norm[:, None, :] * torch.exp(
            torch.tanh(raw_log_norm)[..., None]
        )
        candidates = candidate_norm * candidate_direction

        # At zero initialization this branch is exactly the fallback in the
        # forward pass, while the straight-through expression preserves the
        # candidate gradient needed to leave the symmetric starting point.
        inactive = (raw == 0).all(dim=-1, keepdim=True)
        exact_fallback_st = candidates + (
            canonical[:, None, :] - candidates
        ).detach()
        candidates = torch.where(inactive, exact_fallback_st, candidates)
        slot_tokens = torch.cat([canonical[:, None, :], candidates], dim=1)
        slot_priors = torch.zeros(
            canonical.shape[0],
            self.slots,
            device=canonical.device,
            dtype=canonical.dtype,
        )
        slot_priors[:, 0] = 1.0
        output = SurfaceRegionCodebookOutput(
            canonical_token=canonical,
            slot_tokens=slot_tokens,
            slot_priors=slot_priors,
        )
        if squeeze:
            return SurfaceRegionCodebookOutput(*(value[0] for value in output))
        return output

    def architecture(self, contract_sha256: str) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "name": SURFACE_RESIDUAL_CODEBOOK_V1,
            "feature_dim": self.feature_dim,
            "geometry_dim": self.geometry_dim,
            "hidden_dim": self.hidden_dim,
            "residual_slots": self.residual_slots,
            "total_slots": self.slots,
            "reliability_attention_mode": self.reliability_attention_mode,
            "context_pooling_mode": self.context_pooling_mode,
            "control_sha256": self.control_sha256,
            "canonical_gauge": "caller_provided_exact_frozen_v2",
            "residual_gauge": "l2_direction_tangent_plus_log_norm",
            "contract_sha256": str(contract_sha256),
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
    ) -> tuple["SurfaceRegionSummaryResidualCodebookV1", Mapping]:
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 5:
            raise ValueError("invalid residual surface-region codebook checkpoint")
        architecture = dict(payload["architecture"])
        expected = architecture.pop("digest")
        model = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            residual_slots=int(architecture["residual_slots"]),
            reliability_attention_mode=str(
                architecture["reliability_attention_mode"]
            ),
            context_pooling_mode=str(architecture["context_pooling_mode"]),
            control_sha256=str(architecture["control_sha256"]),
        )
        if (
            model.architecture(str(architecture["contract_sha256"]))["digest"]
            != expected
        ):
            raise ValueError("residual codebook architecture digest mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        return model, payload
