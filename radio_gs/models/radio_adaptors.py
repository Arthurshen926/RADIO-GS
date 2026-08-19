"""Utilities for frozen RADIO teacher adaptors.

RADIO C-v4 exposes task-oriented adaptors such as ``siglip2-g``,
``dino_v3``/``dino_v3_7b``, and ``sam3``.  This module provides the generic MLP
adaptor used by DINOv3 and SAM3 heads/projections so RADIO-GS can supervise
rendered 1280d features in those frozen teacher spaces without changing the
main scene representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from radio_gs.utils.immutable_artifacts import (
    load_fixed_radio_checkpoint_payload,
    sha256_file,
)


RADIO_ADAPTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "dino_v3": ("dino_v3", "dino_v3_7b"),
    "dino_v3_7b": ("dino_v3_7b", "dino_v3"),
    "siglip2-g": ("siglip2-g",),
    "sam3": ("sam3",),
}


class RadioMLPAdaptor(nn.Module):
    """Generic RADIO MLP adaptor for non-attention heads/projections.

    The state dict layout matches RADIO keys such as ``_heads.sam3.*`` and
    ``_feature_projections.dino_v3_7b.*``:
    ``fc1 -> residual LN/GELU/Linear blocks -> final LN/GELU/Linear``.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 1520,
        output_dim: int = 1024,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``[B, N, input_dim]`` tokens to adaptor space."""
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        return self.final(x)


def _checkpoint_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("RADIO checkpoint must be a state dict or contain 'state_dict'")
    return state


def _kind_prefix(kind: str) -> str:
    if kind in {"feature", "feature_projection", "projection"}:
        return "_feature_projections"
    if kind in {"head", "summary"}:
        return "_heads"
    raise ValueError("kind must be one of: feature_projection, feature, projection, head, summary")


def _candidate_names(name: str) -> tuple[str, ...]:
    return RADIO_ADAPTOR_ALIASES.get(name, (name,))


def _extract_prefixed_state(
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    kind: str,
) -> dict[str, torch.Tensor]:
    root = _kind_prefix(kind)
    for candidate in _candidate_names(name):
        prefix = f"{root}.{candidate}."
        extracted = {
            key[len(prefix) :]: value.float()
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    raise KeyError(f"No RADIO adaptor state found for name={name!r}, kind={kind!r}")


def _infer_mlp_shape(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int]:
    fc1 = state["fc1.weight"]
    final = state["final.2.weight"]
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.endswith(".2.weight")
    }
    return int(fc1.shape[1]), int(fc1.shape[0]), int(final.shape[0]), len(block_indices)


def load_radio_adaptor_from_checkpoint(
    checkpoint_path: str | Path,
    name: str,
    *,
    kind: str = "feature_projection",
    expected_sha256: str | None = None,
) -> RadioMLPAdaptor:
    """Load a frozen RADIO MLP adaptor through the restricted RADIO loader.

    Formal callers provide an externally trusted digest.  Legacy callers are
    still protected by the purpose-specific pickle allowlist and stable-file
    checks, but their digest is self-observed and therefore is not an external
    authority statement.
    """
    path = Path(checkpoint_path)
    trusted_sha256 = str(expected_sha256 or sha256_file(path))
    checkpoint, _observed_sha256, _source = (
        load_fixed_radio_checkpoint_payload(
            path,
            expected_sha256=trusted_sha256,
            map_location="cpu",
            label="frozen RADIO adaptor checkpoint",
        )
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("RADIO checkpoint must contain a mapping")
    adaptor_state = _extract_prefixed_state(_checkpoint_state_dict(checkpoint), name, kind)
    input_dim, hidden_dim, output_dim, num_blocks = _infer_mlp_shape(adaptor_state)
    adaptor = RadioMLPAdaptor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_blocks=num_blocks,
    )
    adaptor.load_state_dict(adaptor_state, strict=True)
    return adaptor


def project_feature_map_with_adaptor(
    features: torch.Tensor,
    adaptor: nn.Module,
    *,
    normalize: bool = True,
    amp: bool = False,
    checkpoint_adaptor: bool = False,
) -> torch.Tensor:
    """Project ``[B, C, H, W]`` RADIO features through a frozen adaptor.

    ``amp=True`` matches the official feature extractor's CUDA projection
    runtime.  It is opt-in so existing float32 callers retain their numerical
    contract, while large attention-based spatial adaptors can use the
    memory-efficient half-precision SDPA kernel without changing token scope.
    """
    if features.ndim != 4:
        raise ValueError(f"Expected features [B,C,H,W], got {tuple(features.shape)}")
    batch, channels, height, width = features.shape
    tokens = features.reshape(batch, channels, height * width).permute(0, 2, 1)
    try:
        first_param = next(adaptor.parameters())
    except StopIteration:
        first_param = None
    if first_param is not None:
        tokens = tokens.to(dtype=first_param.dtype, device=first_param.device)
    with torch.cuda.amp.autocast(enabled=bool(amp and tokens.is_cuda)):
        if bool(checkpoint_adaptor and torch.is_grad_enabled() and tokens.requires_grad):
            # The adaptor is frozen; recomputing its exact forward during
            # backward avoids retaining two full-grid transformer activation
            # stacks.  Only the gradient with respect to RADIO tokens is
            # needed, and no stochastic layer is active in eval mode.
            projected = checkpoint(adaptor, tokens, use_reentrant=False)
        else:
            projected = adaptor(tokens)
    projected = projected.permute(0, 2, 1).reshape(batch, -1, height, width)
    if normalize:
        projected = F.normalize(projected.float(), dim=1)
    return projected
