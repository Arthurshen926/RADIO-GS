"""Model-independent authorized query input for SUGM-v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class QueryPacket:
    modality: Literal["text", "image", "prompt"]
    token: torch.Tensor | None = None
    seed_probability: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.modality not in ("text", "image", "prompt"):
            raise ValueError("unsupported SUGM-v3 query modality")
        if self.modality in ("text", "image"):
            if self.seed_probability is not None or self.token is None:
                raise ValueError("text/image query requires only a frozen-encoder token")
            value = torch.as_tensor(self.token)
            if value.shape not in ((1536,), (1, 1536)) or not bool(torch.isfinite(value).all()):
                raise ValueError("text/image query token must be one finite D1536 vector")
        else:
            if self.token is not None or self.seed_probability is None:
                raise ValueError("prompt query requires only Gaussian-domain seed probability")
            seed = torch.as_tensor(self.seed_probability).reshape(-1)
            finite = torch.isfinite(seed)
            if not bool(finite.any()) or bool(((seed[finite] < 0) | (seed[finite] > 1)).any()):
                raise ValueError("prompt seed probability must contain finite values in [0,1]")


__all__ = ["QueryPacket"]
