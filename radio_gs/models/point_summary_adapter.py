"""Compact point-feature adapter for text-aligned ScanNet OVP evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactToSummaryAdapter(nn.Module):
    """Map compact RADIO-GS point features directly to SigLIP text space.

    This adapter is intentionally small and scene-trainable. It gives direct
    point-cloud evaluation a path that does not rely on reconstructing full
    1280d RADIO tokens through the HCD decoder before text classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1536,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, compact: torch.Tensor) -> torch.Tensor:
        if compact.dim() != 2:
            raise ValueError(f"Expected compact features [N,D], got {tuple(compact.shape)}")
        return self.net(compact.float())
