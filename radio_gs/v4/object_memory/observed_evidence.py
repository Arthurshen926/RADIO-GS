"""Factual visible object evidence, separate from association and completion."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservedObjectEvidence:
    positive: torch.Tensor
    negative: torch.Tensor
    unknown: torch.Tensor
    view_ids: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        positive = torch.as_tensor(self.positive, dtype=torch.float32)
        negative = torch.as_tensor(self.negative, dtype=torch.float32)
        unknown = torch.as_tensor(self.unknown, dtype=torch.float32)
        if positive.ndim != 2 or negative.shape != positive.shape or unknown.shape != positive.shape:
            raise ValueError("positive, negative, and unknown evidence must have shape [M, E]")
        view_ids = torch.as_tensor(self.view_ids, dtype=torch.long)
        quality = torch.as_tensor(self.quality, dtype=torch.float32)
        if view_ids.shape != (positive.shape[0],) or quality.shape != (positive.shape[0],):
            raise ValueError("view ids and quality must have one value per mask")
        if not all(torch.isfinite(value).all() for value in (positive, negative, unknown, quality)):
            raise ValueError("observed object evidence must be finite")
        if bool((positive < 0).any()) or bool((negative < 0).any()) or bool((unknown < 0).any()):
            raise ValueError("observed object evidence must be non-negative")
        if bool((positive + negative + unknown > 1.0 + 1e-5).any()):
            raise ValueError("positive, negative, and unknown evidence exceeds unit mass")
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "view_ids", view_ids)
        object.__setattr__(self, "quality", quality)

    @classmethod
    def from_positive_visibility(
        cls,
        positive: torch.Tensor,
        visible: torch.Tensor,
        *,
        view_ids: torch.Tensor,
        quality: torch.Tensor,
    ) -> "ObservedObjectEvidence":
        """Keep ordinary visible mask exterior unknown, never implicit negative."""

        positive = torch.as_tensor(positive, dtype=torch.float32).clamp(0, 1)
        visible = torch.as_tensor(visible, dtype=torch.float32).clamp(0, 1)
        if visible.shape != positive.shape:
            raise ValueError("positive evidence and visibility must align")
        positive = torch.minimum(positive, visible)
        negative = torch.zeros_like(positive)
        unknown = 1.0 - positive
        return cls(positive, negative, unknown, view_ids, quality)

    @property
    def known(self) -> torch.Tensor:
        return self.positive + self.negative

    def with_explicit_negative(self, negative: torch.Tensor) -> "ObservedObjectEvidence":
        negative = torch.as_tensor(negative, dtype=torch.float32).clamp(0, 1)
        if negative.shape != self.positive.shape:
            raise ValueError("explicit negative evidence must align")
        negative = torch.minimum(negative, 1.0 - self.positive)
        return ObservedObjectEvidence(
            self.positive,
            negative,
            (1.0 - self.positive - negative).clamp_min(0),
            self.view_ids,
            self.quality,
        )
