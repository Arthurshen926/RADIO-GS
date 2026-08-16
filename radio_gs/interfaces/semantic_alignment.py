"""Three-level, official-first semantic alignment decision contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


def align_full_extent_feature_grid(
    value: torch.Tensor,
    target_size: tuple[int, int],
    *,
    label: str,
) -> torch.Tensor:
    """Align a full-image feature grid across a one-cell rounding difference.

    RADIO's native token grid follows the rounded model input resolution, while
    a frozen full-extent teacher may use one shared grid across scenes.  A
    one-cell difference therefore describes the same image domain; anything
    larger remains a contract error.
    """

    if value.ndim not in (3, 4):
        raise ValueError(f"{label}: feature map must be [C,H,W] or [B,C,H,W]")
    target = tuple(int(size) for size in target_size)
    if len(target) != 2 or min(target) <= 0:
        raise ValueError(f"{label}: target grid must be positive HxW")
    source = tuple(int(size) for size in value.shape[-2:])
    if source == target:
        return value
    if max(abs(a - b) for a, b in zip(source, target)) > 1:
        raise ValueError(f"{label}: grid {source} vs {target}")
    batched = value.unsqueeze(0) if value.ndim == 3 else value
    aligned = F.interpolate(
        batched,
        size=target,
        mode="bilinear",
        align_corners=False,
    )
    return aligned[0] if value.ndim == 3 else aligned


class SemanticAlignmentStage(str, Enum):
    OFFICIAL_SPATIAL = "official_siglip2_spatial"
    OFFICIAL_CROP_SUMMARY = "official_siglip2_crop_summary"
    GLOBAL_FROZEN_BRIDGE = "global_frozen_region_summary_bridge"


@dataclass(frozen=True)
class SemanticOracleResult:
    stage: SemanticAlignmentStage
    dataset: str
    miou: float
    localization_accuracy: float
    sample_count: int
    protocol_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", SemanticAlignmentStage(self.stage))
        if self.sample_count <= 0:
            raise ValueError("oracle sample_count must be positive")
        if not 0.0 <= self.miou <= 1.0 or not 0.0 <= self.localization_accuracy <= 1.0:
            raise ValueError("oracle metrics must be fractions in [0,1]")
        if not self.protocol_hash:
            raise ValueError("oracle protocol_hash is required")


@dataclass(frozen=True)
class GlobalSemanticBridgeManifest:
    checkpoint_sha256: str
    training_scope: str
    frozen: bool
    uses_benchmark_test_vocabulary: bool
    uses_benchmark_scenes: bool
    training_dataset_manifest_sha256: str

    def validate(self) -> None:
        if self.training_scope != "global_cross_scene":
            raise ValueError("semantic bridge must be global_cross_scene")
        if not self.frozen:
            raise ValueError("semantic bridge must be frozen for every scene/query")
        if self.uses_benchmark_test_vocabulary or self.uses_benchmark_scenes:
            raise ValueError("semantic bridge cannot use benchmark scenes or test vocabulary")
        if not self.checkpoint_sha256 or not self.training_dataset_manifest_sha256:
            raise ValueError("semantic bridge provenance is incomplete")


@dataclass(frozen=True)
class SemanticAlignmentDecision:
    selected_stage: SemanticAlignmentStage
    reason: str
    stage1: SemanticOracleResult
    stage2: SemanticOracleResult | None = None
    bridge_manifest: GlobalSemanticBridgeManifest | None = None


@dataclass(frozen=True)
class SemanticAlignmentPolicy:
    """Select the simplest official interface that clears frozen oracle gates."""

    minimum_miou: float
    minimum_localization_accuracy: float

    def _sufficient(self, result: SemanticOracleResult) -> bool:
        return (
            result.miou >= float(self.minimum_miou)
            and result.localization_accuracy >= float(self.minimum_localization_accuracy)
        )

    def decide(
        self,
        stage1: SemanticOracleResult,
        *,
        stage2: SemanticOracleResult | None = None,
        bridge_manifest: GlobalSemanticBridgeManifest | None = None,
    ) -> SemanticAlignmentDecision:
        if stage1.stage is not SemanticAlignmentStage.OFFICIAL_SPATIAL:
            raise ValueError("the first oracle must evaluate official SigLIP2 spatial output")
        if self._sufficient(stage1):
            return SemanticAlignmentDecision(
                selected_stage=stage1.stage,
                reason="official spatial oracle clears the frozen quality gate",
                stage1=stage1,
            )
        if stage2 is None:
            raise RuntimeError(
                "official spatial oracle is insufficient; official crop-summary oracle is required"
            )
        if stage2.stage is not SemanticAlignmentStage.OFFICIAL_CROP_SUMMARY:
            raise ValueError("stage2 must evaluate official crop summaries")
        if self._sufficient(stage2):
            return SemanticAlignmentDecision(
                selected_stage=stage2.stage,
                reason="official crop-summary oracle clears the frozen quality gate",
                stage1=stage1,
                stage2=stage2,
            )
        if bridge_manifest is None:
            raise RuntimeError(
                "both official oracles are insufficient; a validated global frozen bridge is required"
            )
        bridge_manifest.validate()
        return SemanticAlignmentDecision(
            selected_stage=SemanticAlignmentStage.GLOBAL_FROZEN_BRIDGE,
            reason="official spatial and crop-summary oracles are insufficient",
            stage1=stage1,
            stage2=stage2,
            bridge_manifest=bridge_manifest,
        )


class GlobalRegionSummaryBridge(nn.Module):
    """Optional global, permutation-invariant region-to-summary aligner.

    The module predicts a 1280-D RADIO *summary token* from a set of RADIO
    spatial tokens.  It never predicts a SigLIP/text descriptor directly;
    callers must pass its output through the frozen official summary head.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 1280,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        if self.output_dim != self.input_dim:
            raise ValueError("region-summary residual requires output_dim == input_dim")
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.Linear(self.hidden_dim, 1)
        self.summary_residual = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        nn.init.zeros_(self.summary_residual[-1].weight)
        nn.init.zeros_(self.summary_residual[-1].bias)

    def forward(
        self,
        radio_features: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(radio_features).float()
        squeeze = values.ndim == 2
        if squeeze:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"expected bridge input dim {self.input_dim}")
        encoded, logits = self.encode_region_tokens(values)
        summary = self.summarize_preencoded_region(
            values, encoded, logits, token_mask=token_mask
        )
        return summary[0] if squeeze else summary

    def encode_region_tokens(
        self, radio_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode individual tokens once for reuse across overlapping regions."""

        values = torch.as_tensor(radio_features).float()
        if values.ndim not in (2, 3) or values.shape[-1] != self.input_dim:
            raise ValueError(f"expected token input dim {self.input_dim}")
        encoded = self.token_encoder(values)
        return encoded, self.attention(encoded).squeeze(-1)

    def summarize_preencoded_region(
        self,
        radio_features: torch.Tensor,
        encoded_tokens: torch.Tensor,
        attention_logits: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool a token set whose pointwise bridge encoding is already cached."""

        values = torch.as_tensor(radio_features).float()
        encoded = torch.as_tensor(encoded_tokens).float()
        logits = torch.as_tensor(attention_logits).float()
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"radio_features must be [B,T,{self.input_dim}]")
        if encoded.shape != (*values.shape[:2], self.hidden_dim):
            raise ValueError("encoded_tokens must align with radio_features")
        if logits.shape != values.shape[:2]:
            raise ValueError("attention_logits must align with radio_features")
        if token_mask is not None:
            mask = torch.as_tensor(token_mask, device=values.device).bool()
            if mask.shape != values.shape[:2] or not bool(mask.any(dim=1).all()):
                raise ValueError("token_mask must keep at least one token per region")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled_raw = torch.einsum("bt,btc->bc", weights, values)
        pooled_hidden = torch.einsum("bt,bth->bh", weights, encoded)
        summary = pooled_raw + self.summary_residual(pooled_hidden)
        return summary

    def dense_square_regions(
        self,
        feature_map: torch.Tensor,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
    ) -> torch.Tensor:
        """Predict a summary token for every square region in a 2-D token map.

        The attention numerator and denominator are box-filtered separately,
        which is exactly equivalent to calling :meth:`forward` on every local
        token set but avoids materializing ``H*W`` unfolded regions.
        """

        values = torch.as_tensor(feature_map).float()
        if values.ndim != 4 or values.shape[1] != self.input_dim:
            raise ValueError(f"feature_map must be [B,{self.input_dim},H,W]")
        batch, _channels, height, width = values.shape
        tokens = values.permute(0, 2, 3, 1)
        encoded = self.token_encoder(tokens)
        logits = self.attention(encoded).squeeze(-1)
        # A per-image constant shift leaves every local softmax unchanged.
        weights = torch.exp(logits - logits.amax(dim=(1, 2), keepdim=True))[:, None]
        weighted_raw = values * weights
        encoded_map = encoded.permute(0, 3, 1, 2)
        weighted_hidden = encoded_map * weights
        summaries: list[torch.Tensor] = []
        for raw_kernel in kernel_sizes:
            kernel = int(raw_kernel)
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError("dense region kernel sizes must be positive odd integers")
            padding = kernel // 2
            denominator = F.avg_pool2d(
                weights,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            ).clamp_min(1e-8)
            pooled_raw = F.avg_pool2d(
                weighted_raw,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            ) / denominator
            pooled_hidden = F.avg_pool2d(
                weighted_hidden,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            ) / denominator
            residual = self.summary_residual(
                pooled_hidden.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            summaries.append(pooled_raw + residual)
        return torch.stack(summaries, dim=1).reshape(
            batch, len(summaries), self.output_dim, height, width
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GlobalRegionSummaryBridge", GlobalSemanticBridgeManifest]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("invalid global semantic bridge checkpoint")
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")
        if not sidecar.is_file():
            raise FileNotFoundError("global semantic bridge manifest sidecar is required")
        manifest = GlobalSemanticBridgeManifest(
            **json.loads(sidecar.read_text(encoding="utf-8"))
        )
        manifest.validate()
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != manifest.checkpoint_sha256:
            raise ValueError("global semantic bridge checkpoint hash mismatch")
        architecture = payload["architecture"]
        bridge = cls(**dict(architecture))
        bridge.load_state_dict(payload["state_dict"], strict=True)
        bridge.eval()
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        return bridge, manifest


def project_dense_region_semantics(
    bridge: GlobalRegionSummaryBridge,
    official_summary_head: nn.Module,
    radio_map: torch.Tensor,
    *,
    kernel_sizes: tuple[int, ...] = (3, 7, 15),
    projection_batch_size: int = 2048,
) -> torch.Tensor:
    """Region-align a RADIO map, then use the frozen official summary head.

    Scales are projected one at a time.  This is algebraically identical to
    stacking every scale before projection, but avoids simultaneously holding
    all dense RADIO summaries and all projected descriptors on the GPU.  The
    latter is several GiB for a full-resolution SPIn frame.
    """

    values = torch.as_tensor(radio_map).float()
    if values.ndim != 4:
        raise ValueError("radio_map must be [B,1280,H,W]")
    kernels = tuple(int(kernel) for kernel in kernel_sizes)
    if not kernels:
        raise ValueError("kernel_sizes must contain at least one scale")
    chunk_size = int(projection_batch_size)
    if chunk_size <= 0:
        raise ValueError("projection_batch_size must be positive")

    batch, _channels, height, width = values.shape
    descriptor_sum: torch.Tensor | None = None
    for kernel in kernels:
        summaries = bridge.dense_square_regions(values, (kernel,))[:, 0]
        channels = int(summaries.shape[1])
        tokens = summaries.permute(0, 2, 3, 1).reshape(-1, channels)
        projected_chunks: list[torch.Tensor] = []
        for start in range(0, tokens.shape[0], chunk_size):
            projected = official_summary_head(tokens[start : start + chunk_size, None])[:, 0]
            projected_chunks.append(
                F.normalize(projected.float(), dim=-1, eps=1e-8)
            )
        scale_descriptor = torch.cat(projected_chunks).reshape(
            batch, height, width, -1
        )
        descriptor_sum = (
            scale_descriptor
            if descriptor_sum is None
            else descriptor_sum + scale_descriptor
        )

    assert descriptor_sum is not None
    fused = F.normalize(descriptor_sum / float(len(kernels)), dim=-1, eps=1e-8)
    return fused.permute(0, 3, 1, 2).contiguous()
