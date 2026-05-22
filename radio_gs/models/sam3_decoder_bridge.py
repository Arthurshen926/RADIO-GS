"""Bridge RADIO feature maps into the official SAM3 image-decoder interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Sam3BackboneSpec:
    """Shape contract exposed by the official SAM3 image processor."""

    channels: int = 256
    fpn_sizes: tuple[int, int, int] = (288, 144, 72)


class Sam3BackboneBridge(nn.Module):
    """Project a RADIO/CTF-GS feature map to SAM3 ``backbone_out`` tensors.

    The module deliberately targets the public ``Sam3Processor`` decoder contract:
    it predicts the three FPN maps consumed by the official segmentation head and
    the final ``vision_features`` map used by the grounding encoder.
    """

    def __init__(
        self,
        *,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        spec: Sam3BackboneSpec | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.spec = spec or Sam3BackboneSpec()
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fpn_heads = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, self.spec.channels, kernel_size=1)
            for _ in self.spec.fpn_sizes
        )

    def forward(self, radio_feature: torch.Tensor) -> dict[str, Any]:
        if radio_feature.ndim == 3:
            radio_feature = radio_feature.unsqueeze(0)
        if radio_feature.ndim != 4:
            raise ValueError(f"Expected RADIO feature [B,C,H,W], got {tuple(radio_feature.shape)}")
        if radio_feature.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input channels, got {radio_feature.shape[1]}"
            )
        base = self.stem(radio_feature.float())
        fpn: list[torch.Tensor] = []
        for size, head in zip(self.spec.fpn_sizes, self.fpn_heads):
            resized = F.interpolate(
                base,
                size=(int(size), int(size)),
                mode="bilinear",
                align_corners=False,
            )
            fpn.append(head(resized))
        return {
            "vision_features": fpn[-1],
            "backbone_fpn": fpn,
        }


def sam3_backbone_bridge_loss(
    prediction: dict[str, Any],
    target: dict[str, Any],
    *,
    cosine_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruct official SAM3 FPN tensors with MSE plus token cosine loss."""

    pred_fpn = prediction["backbone_fpn"]
    tgt_fpn = target["backbone_fpn"]
    if len(pred_fpn) != len(tgt_fpn):
        raise ValueError(f"FPN level mismatch: {len(pred_fpn)} vs {len(tgt_fpn)}")

    total = torch.zeros((), device=pred_fpn[0].device)
    stats: dict[str, float] = {}
    for level, (pred, tgt) in enumerate(zip(pred_fpn, tgt_fpn)):
        tgt = tgt.to(device=pred.device, dtype=pred.dtype)
        mse = F.mse_loss(pred, tgt)
        pred_tokens = F.normalize(pred.flatten(2).transpose(1, 2).float(), dim=-1)
        tgt_tokens = F.normalize(tgt.flatten(2).transpose(1, 2).float(), dim=-1)
        cosine = 1.0 - (pred_tokens * tgt_tokens).sum(dim=-1).mean()
        level_loss = mse + float(cosine_weight) * cosine.to(mse.dtype)
        total = total + level_loss
        stats[f"fpn{level}_mse"] = float(mse.detach().cpu())
        stats[f"fpn{level}_cosine_loss"] = float(cosine.detach().cpu())

    pred_vision = prediction["vision_features"]
    tgt_vision = target["vision_features"].to(device=pred_vision.device, dtype=pred_vision.dtype)
    vision_mse = F.mse_loss(pred_vision, tgt_vision)
    total = total + vision_mse
    stats["vision_mse"] = float(vision_mse.detach().cpu())
    stats["total"] = float(total.detach().cpu())
    return total, stats

