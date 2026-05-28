"""Prompt-conditioned feature-only mask refinement heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptConditionedMaskHead(nn.Module):
    """Predict one mask logit per text prompt from rendered foundation features.

    The head is intentionally small: a shared visual stem processes the rendered
    feature map, a prompt projection supplies per-query FiLM-style conditioning,
    and the coarse rendered mask is injected as a spatial prompt.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 1280,
        prompt_dim: int = 1536,
        hidden_dim: int = 128,
        predict_quality: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if prompt_dim <= 0:
            raise ValueError("prompt_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.prompt_dim = int(prompt_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_quality = bool(predict_quality)
        self.visual = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, self.hidden_dim), num_channels=self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.prompt = nn.Sequential(
            nn.Linear(self.prompt_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        if self.predict_quality:
            self.quality = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.quality = None

    def _fused_query_features(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        coarse_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got {tuple(features.shape)}")
        if prompts.ndim == 2:
            prompts = prompts.unsqueeze(0)
        if prompts.ndim != 3:
            raise ValueError(f"prompts must be [B,Q,D] or [Q,D], got {tuple(prompts.shape)}")
        if coarse_masks.ndim != 4:
            raise ValueError(
                f"coarse_masks must be [B,Q,H,W], got {tuple(coarse_masks.shape)}"
            )
        batch, channels, height, width = features.shape
        if channels != self.feature_dim:
            raise ValueError(f"expected {self.feature_dim} feature channels, got {channels}")
        if prompts.shape[0] == 1 and batch > 1:
            prompts = prompts.expand(batch, -1, -1)
        if prompts.shape[0] != batch:
            raise ValueError("prompt batch does not match feature batch")
        if prompts.shape[2] != self.prompt_dim:
            raise ValueError(f"expected prompt dim {self.prompt_dim}, got {prompts.shape[2]}")
        if coarse_masks.shape[:2] != prompts.shape[:2]:
            raise ValueError("coarse mask batch/query dimensions must match prompts")
        if coarse_masks.shape[-2:] != (height, width):
            coarse_masks = F.interpolate(
                coarse_masks.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        query_count = prompts.shape[1]
        visual = self.visual(features.float())
        visual = visual[:, None].expand(batch, query_count, self.hidden_dim, height, width)
        visual = visual.reshape(batch * query_count, self.hidden_dim, height, width)

        prompt_params = self.prompt(prompts.float()).reshape(batch * query_count, self.hidden_dim * 2)
        scale, bias = prompt_params.chunk(2, dim=-1)
        scale = torch.tanh(scale).view(batch * query_count, self.hidden_dim, 1, 1)
        bias = bias.view(batch * query_count, self.hidden_dim, 1, 1)

        coarse = coarse_masks.reshape(batch * query_count, 1, *coarse_masks.shape[-2:]).float()
        coarse_feat = self.coarse(coarse)
        fused = visual * (1.0 + scale) + bias + coarse_feat
        return fused, batch, query_count, height, width

    def forward(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        coarse_masks: torch.Tensor,
    ) -> torch.Tensor:
        fused, batch, query_count, height, width = self._fused_query_features(
            features,
            prompts,
            coarse_masks,
        )
        logits = self.out(fused)
        return logits.reshape(batch, query_count, height, width)

    def forward_with_quality(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        coarse_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fused, batch, query_count, height, width = self._fused_query_features(
            features,
            prompts,
            coarse_masks,
        )
        logits = self.out(fused).reshape(batch, query_count, height, width)
        if self.quality is None:
            return logits, None
        quality_logits = self.quality(fused).reshape(batch, query_count)
        return logits, quality_logits
