"""Model-independent query packets for Query-Native Gaussian Memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import torch


QueryModality = Literal["text", "image", "prompt"]


@dataclass(frozen=True)
class QueryPacket:
    """Validated output of a replaceable, pretrained query encoder adapter.

    ``tokens`` are already aligned to the decoder query space. A registered
    prompt may additionally provide a Gaussian-domain seed probability; NaN
    denotes unknown rather than negative evidence.
    """

    tokens: torch.Tensor
    modality: QueryModality
    confidence: torch.Tensor | None = None
    seed_probability: torch.Tensor | None = None
    spatial_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.modality not in ("text", "image", "prompt"):
            raise ValueError("unsupported query modality")
        if self.tokens.ndim != 2 or self.tokens.shape[0] < 1:
            raise ValueError("query tokens must be nonempty [M,D]")
        if not bool(torch.isfinite(self.tokens).all()):
            raise ValueError("query tokens contain NaN or infinity")
        if self.confidence is not None and (
            self.confidence.shape != (self.tokens.shape[0],)
            or not bool(torch.isfinite(self.confidence).all())
            or bool((self.confidence < 0).any())
        ):
            raise ValueError("query-token confidence differs")
        if self.seed_probability is not None:
            finite = torch.isfinite(self.seed_probability)
            if self.seed_probability.ndim != 1 or bool(
                ((self.seed_probability[finite] < 0) | (self.seed_probability[finite] > 1)).any()
            ):
                raise ValueError("prompt seed probability differs")

    def to(self, device: torch.device | str) -> "QueryPacket":
        """Move tensor payloads without exposing encoder-specific state."""

        return replace(
            self,
            tokens=self.tokens.to(device),
            confidence=(None if self.confidence is None else self.confidence.to(device)),
            seed_probability=(
                None if self.seed_probability is None else self.seed_probability.to(device)
            ),
        )


__all__ = ["QueryModality", "QueryPacket"]
