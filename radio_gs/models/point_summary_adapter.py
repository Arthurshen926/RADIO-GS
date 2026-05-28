"""Compact point-feature adapter for text-aligned ScanNet OVP evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_CONTEXT_DIMS = {
    "opacity": 1,
    "scale_log_mean": 1,
    "scale_log_max": 1,
    "view_count": 1,
}


def parse_point_summary_context_features(spec: str) -> tuple[str, ...]:
    """Parse optional scalar context names for the primitive summary adapter."""
    tokens = tuple(
        token.strip().lower()
        for token in str(spec or "").replace(",", " ").replace("+", " ").split()
        if token.strip()
    )
    unknown = sorted(set(tokens) - set(_CONTEXT_DIMS))
    if unknown:
        raise ValueError(
            "Unsupported point summary adapter context feature(s): "
            + ", ".join(unknown)
        )
    return tokens


def point_summary_context_dim(spec: str) -> int:
    return sum(_CONTEXT_DIMS[token] for token in parse_point_summary_context_features(spec))


def _reshape_scalar_context(name: str, value: torch.Tensor, rows: int) -> torch.Tensor:
    tensor = value.float()
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 1 or tensor.shape[0] != rows:
        raise ValueError(f"{name} must have shape [{rows}] or [{rows},1], got {tuple(value.shape)}")
    return tensor


def append_point_summary_context(
    compact: torch.Tensor,
    *,
    context_features: str = "",
    opacity: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    view_counts: torch.Tensor | None = None,
    view_count_max: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Append bounded primitive reliability/context scalars to compact features.

    The optional context is intended for the direct 3D primitive summary head:
    compact appearance remains the primary signal, while opacity, scale and
    multiview registration counts provide GT-free reliability cues.
    """
    if compact.ndim != 2:
        raise ValueError(f"Expected compact features [N,D], got {tuple(compact.shape)}")
    tokens = parse_point_summary_context_features(context_features)
    if not tokens:
        return compact.float()

    rows = int(compact.shape[0])
    parts = [compact.float()]
    for token in tokens:
        if token == "opacity":
            if opacity is None:
                raise ValueError("opacity is required by point summary adapter context")
            op = _reshape_scalar_context("opacity", opacity, rows).clamp(0.0, 1.0)
            parts.append(op.unsqueeze(1))
        elif token in {"scale_log_mean", "scale_log_max"}:
            if scales is None:
                raise ValueError("scales is required by point summary adapter context")
            scale_tensor = scales.float()
            if scale_tensor.ndim != 2 or scale_tensor.shape[0] != rows:
                raise ValueError(
                    f"scales must have shape [{rows},S], got {tuple(scales.shape)}"
                )
            log_scales = scale_tensor.clamp_min(1e-8).log()
            value = log_scales.mean(dim=1) if token == "scale_log_mean" else log_scales.max(dim=1).values
            parts.append(torch.tanh(value / 5.0).unsqueeze(1))
        elif token == "view_count":
            if view_counts is None:
                raise ValueError("view_counts is required by point summary adapter context")
            counts = _reshape_scalar_context("view_counts", view_counts, rows).clamp_min(0.0)
            if view_count_max is None:
                max_count = counts.max().detach()
            elif isinstance(view_count_max, torch.Tensor):
                max_count = view_count_max.to(device=counts.device, dtype=counts.dtype).detach()
            else:
                max_count = torch.tensor(float(view_count_max), device=counts.device, dtype=counts.dtype)
            denom = torch.log1p(max_count.clamp_min(counts.max().detach()).clamp_min(1.0))
            parts.append((torch.log1p(counts) / denom.clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(1))

    return torch.cat([part.to(device=compact.device, dtype=compact.dtype) for part in parts], dim=1)


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
