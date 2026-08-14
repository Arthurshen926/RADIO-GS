"""Shared SigLIP2 projection utilities for grounding-aware supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from radio_gs.utils.immutable_artifacts import (
    load_fixed_radio_checkpoint_payload,
    load_torch_payload,
)


OFFICIAL_C_RADIO_V4_H_HALF_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)


def _load_weights_only(path: str) -> object:
    payload, _, _ = load_torch_payload(
        path,
        map_location="cpu",
        label="SigLIP2 projection checkpoint",
    )
    return payload


def _load_fixed_radio_checkpoint(
    path: str,
    *,
    expected_sha256: str,
) -> object:
    payload, _, _ = load_fixed_radio_checkpoint_payload(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="official C-RADIOv4-H checkpoint",
    )
    return payload


class SigLIP2FeatureProjection(nn.Module):
    """Project RADIO 1280d features into SigLIP2 visual embedding space.

    Uses the spatial feature projection from RADIO: 2 attention blocks + MLP.
    Maps to SigLIP2's *spatial* vision space (1536d).
    NOTE: For text grounding, use SigLIP2SummaryHead instead — it maps to
    the text-aligned summary space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1280, num_heads=16, init_values=1e-5)
            for _ in range(2)
        ])
        self.mlp_fc1 = nn.Linear(1280, 1520)
        self.mlp_final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.blocks(x)
        x = self.mlp_fc1(x)
        x = self.mlp_final(x)
        return x

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2FeatureProjection":
        """Load from already-extracted projection state dict (e.g. siglip2_feat_projection.pth)."""
        sd = _load_weights_only(ckpt_path)
        proj = cls()
        proj.load_state_dict(sd, strict=True)
        return proj

    @classmethod
    def from_radio_checkpoint(
        cls,
        ckpt_path: str,
        *,
        expected_sha256: str = OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    ) -> "SigLIP2FeatureProjection":
        chk = _load_fixed_radio_checkpoint(
            ckpt_path,
            expected_sha256=expected_sha256,
        )
        sd = chk["state_dict"]
        proj = cls()
        proj_sd = {}
        prefix = "_feature_projections.siglip2-g."
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix):]
            if new_k.startswith("mlp.fc1"):
                new_k = new_k.replace("mlp.fc1", "mlp_fc1")
            elif new_k.startswith("mlp.final"):
                new_k = new_k.replace("mlp.final", "mlp_final")
            proj_sd[new_k] = v.float()
        proj.load_state_dict(proj_sd, strict=True)
        return proj


class SigLIP2SummaryHead(nn.Module):
    """RADIO's SigLIP2 summary head — maps 1280d tokens to the text-aligned 1536d space.

    Architecture: Linear(1280→1520) + 2 residual blocks(LN+GELU+Linear) + final(LN+GELU+Linear→1536).
    This is ``_heads.siglip2-g`` from the RADIO checkpoint. Unlike the spatial feature projection,
    the summary head produces embeddings in the same space as SigLIP2 text embeddings, making it
    suitable for text grounding / open-vocabulary tasks when its inputs are
    genuine or explicitly predicted summary tokens.  Applying this head to
    every spatial pixel or Gaussian is not an official SigLIP2 projection;
    use :class:`SigLIP2FeatureProjection` for full-grid spatial preservation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1280, 1520)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
        ])
        self.final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.fc1(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.final(x)

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2SummaryHead":
        """Load from extracted state dict (e.g. siglip2_summary_head.pth)."""
        sd = _load_weights_only(ckpt_path)
        head = cls()
        head.load_state_dict(sd, strict=True)
        return head

    @classmethod
    def from_radio_checkpoint(
        cls,
        ckpt_path: str,
        *,
        expected_sha256: str = OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    ) -> "SigLIP2SummaryHead":
        """Extract ``_heads.siglip2-g`` from a full RADIO checkpoint."""
        chk = _load_fixed_radio_checkpoint(
            ckpt_path,
            expected_sha256=expected_sha256,
        )
        sd = chk["state_dict"]
        head = cls()
        head_sd = {}
        prefix = "_heads.siglip2-g."
        for k, v in sd.items():
            if k.startswith(prefix):
                head_sd[k[len(prefix):]] = v.float()
        head.load_state_dict(head_sd, strict=True)
        return head
