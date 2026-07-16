"""Global 3-D surface-region readout for official RADIO summary tokens.

The readout is intentionally query-free and scene-independent.  It consumes
only canonical primitive state and predicts a genuine 1280-D RADIO summary
token, which must subsequently be passed through the frozen official
``siglip2-g`` summary head.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


SURFACE_GEOMETRY_DIM = 12


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
