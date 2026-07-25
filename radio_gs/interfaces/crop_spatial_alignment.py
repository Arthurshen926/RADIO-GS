"""Frozen global alignment from a pose-free crop to a full-image DINO token."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CropSpatialAlignmentManifest:
    """Fail-closed provenance for the optional global query-side bridge."""

    checkpoint_sha256: str
    training_scope: str
    frozen: bool
    uses_benchmark_scenes: bool
    uses_benchmark_labels: bool
    scene_disjoint: bool
    training_manifest_sha256: str

    def validate(self) -> None:
        if self.training_scope != "global_cross_scene_crop_to_spatial_dino":
            raise ValueError("crop alignment must be global_cross_scene_crop_to_spatial_dino")
        if not self.frozen or not self.scene_disjoint:
            raise ValueError("crop alignment must be frozen and scene-disjoint")
        if self.uses_benchmark_scenes or self.uses_benchmark_labels:
            raise ValueError("crop alignment cannot use benchmark scenes or labels")
        if not self.checkpoint_sha256 or not self.training_manifest_sha256:
            raise ValueError("crop alignment provenance is incomplete")


@dataclass(frozen=True)
class CropContextAlignmentManifest:
    """Fail-closed provenance for the global crop-context query compiler."""

    checkpoint_sha256: str
    training_scope: str
    frozen: bool
    uses_benchmark_scenes: bool
    uses_benchmark_labels: bool
    scene_disjoint: bool
    training_manifest_sha256: str

    def validate(self) -> None:
        if self.training_scope != "global_cross_scene_crop_context_to_spatial_dino":
            raise ValueError(
                "crop context alignment must be "
                "global_cross_scene_crop_context_to_spatial_dino"
            )
        if not self.frozen or not self.scene_disjoint:
            raise ValueError("crop context alignment must be frozen and scene-disjoint")
        if self.uses_benchmark_scenes or self.uses_benchmark_labels:
            raise ValueError("crop context alignment cannot use benchmark scenes or labels")
        if not self.checkpoint_sha256 or not self.training_manifest_sha256:
            raise ValueError("crop context alignment provenance is incomplete")


class GlobalCropSpatialAdapter(nn.Module):
    """Low-rank residual correcting crop-context shift in official DINO space.

    It maps one official C-RADIO DINO crop-center descriptor to the descriptor
    that the same frozen official adaptor emits at that pixel in the full
    source image.  The adapter is global, query-side only, and initialized as
    an exact identity; it never changes a scene field or an official adaptor.
    """

    def __init__(self, *, feature_dim: int = 4096, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("crop adapter dimensions must be positive")
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(self.feature_dim, self.hidden_dim, bias=False)
        self.up = nn.Linear(self.hidden_dim, self.feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(descriptors).float()
        squeeze = values.ndim == 1
        if squeeze:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"crop adapter expects [B,{self.feature_dim}] official DINO descriptors"
            )
        base = F.normalize(values, dim=-1, eps=1e-8)
        residual = self.up(F.gelu(self.down(self.input_norm(base))))
        result = F.normalize(base + residual, dim=-1, eps=1e-8)
        return result[0] if squeeze else result

    def architecture(self) -> dict[str, int | str]:
        return {
            "name": "global_crop_spatial_adapter_v1",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GlobalCropSpatialAdapter", CropSpatialAlignmentManifest]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("invalid global crop-spatial adapter checkpoint")
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
        if not sidecar.is_file():
            raise FileNotFoundError("crop-spatial adapter manifest sidecar is required")
        manifest = CropSpatialAlignmentManifest(
            **json.loads(sidecar.read_text(encoding="utf-8"))
        )
        manifest.validate()
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != manifest.checkpoint_sha256:
            raise ValueError("crop-spatial adapter checkpoint hash mismatch")
        architecture = dict(payload.get("architecture", {}))
        if architecture.get("name") != "global_crop_spatial_adapter_v1":
            raise ValueError("unsupported crop-spatial adapter architecture")
        adapter = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
        )
        adapter.load_state_dict(payload["state_dict"], strict=True)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        return adapter, manifest


def checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GlobalCropContextAdapter(nn.Module):
    """Frozen residual bridge from crop center *and context* to a full-image token.

    The input uses only the official DINO spatial map produced from the
    method-visible crop: its centre descriptor and its spatial global mean.
    It therefore addresses crop/full-image context shift without adding a
    scene-specific field, camera pose, depth, or benchmark label. The zero
    initialized residual makes the checkpoint an exact centre-token baseline
    until cross-scene RGB-only training demonstrates an improvement.
    """

    def __init__(self, *, feature_dim: int = 4096, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("crop context adapter dimensions must be positive")
        self.center_norm = nn.LayerNorm(self.feature_dim)
        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(2 * self.feature_dim, self.hidden_dim, bias=False)
        self.up = nn.Linear(self.hidden_dim, self.feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        center_descriptors: torch.Tensor,
        context_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        center = torch.as_tensor(center_descriptors).float()
        context = torch.as_tensor(context_descriptors, device=center.device).float()
        squeeze = center.ndim == 1
        if squeeze:
            center = center[None]
            context = context[None] if context.ndim == 1 else context
        if (
            center.ndim != 2
            or context.ndim != 2
            or center.shape != context.shape
            or center.shape[1] != self.feature_dim
        ):
            raise ValueError(
                f"crop context adapter expects aligned [B,{self.feature_dim}] inputs"
            )
        base = F.normalize(center, dim=-1, eps=1e-8)
        context = F.normalize(context, dim=-1, eps=1e-8)
        hidden = F.gelu(
            self.down(
                torch.cat([self.center_norm(base), self.context_norm(context)], dim=-1)
            )
        )
        result = F.normalize(base + self.up(hidden), dim=-1, eps=1e-8)
        return result[0] if squeeze else result

    def architecture(self) -> dict[str, int | str]:
        return {
            "name": "global_crop_context_adapter_v1",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "context": "official_dino_crop_spatial_global_mean",
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GlobalCropContextAdapter", CropContextAlignmentManifest]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
            raise ValueError("invalid global crop-context adapter checkpoint")
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
        if not sidecar.is_file():
            raise FileNotFoundError("crop-context adapter manifest sidecar is required")
        manifest = CropContextAlignmentManifest(
            **json.loads(sidecar.read_text(encoding="utf-8"))
        )
        manifest.validate()
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != manifest.checkpoint_sha256:
            raise ValueError("crop-context adapter checkpoint hash mismatch")
        architecture = dict(payload.get("architecture", {}))
        if architecture.get("name") != "global_crop_context_adapter_v1":
            raise ValueError("unsupported crop-context adapter architecture")
        adapter = cls(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
        )
        adapter.load_state_dict(payload["state_dict"], strict=True)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        return adapter, manifest
