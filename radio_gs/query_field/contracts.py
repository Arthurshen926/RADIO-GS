"""Typed, transient query evidence; estimated identity is never a hard prompt."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Support:
    indices: torch.Tensor
    weights: torch.Tensor
    # Optional ray identity. Contributors within a ray are alternatives;
    # different negative rays are separate constraints. Never stored in a field.
    components: torch.Tensor | None = None

    def validate(self, count: int) -> None:
        if self.indices.ndim != 1 or self.weights.shape != self.indices.shape:
            raise ValueError("support must have aligned one-dimensional indices and weights")
        if self.indices.dtype != torch.long or not len(self.indices):
            raise ValueError("support requires nonempty integer indices")
        if bool(((self.indices < 0) | (self.indices >= count)).any()):
            raise ValueError("support index outside Gaussian carrier")
        if not bool(torch.isfinite(self.weights).all()) or bool((self.weights < 0).any()):
            raise ValueError("support weights must be finite and nonnegative")
        if not bool(self.weights.sum() > 0):
            raise ValueError("support must have positive mass")
        if self.components is not None:
            if self.components.shape != self.indices.shape or self.components.dtype != torch.long:
                raise ValueError("support components must be aligned integer ray identifiers")
            if bool((self.components < 0).any()):
                raise ValueError("support components must be nonnegative")


@dataclass(frozen=True)
class QueryPacket:
    modality: Literal["text", "image", "point_2d", "point_3d"]
    task: Literal["instance", "union"] = "instance"
    semantic: torch.Tensor | None = None
    local: torch.Tensor | None = None
    positive: Support | None = None
    negative: Support | None = None
    scale: float | None = None
    # Only a supplied positive 3D element can be an exact membership constraint.
    # A camera ray and an estimated text/image anchor are uncertain associations.
    exact_positive: bool = False

    def validate(self, count: int, semantic_dim: int, local_dim: int) -> None:
        if self.modality not in {"text", "image", "point_2d", "point_3d"}:
            raise ValueError("unsupported query modality")
        if self.task not in {"instance", "union"}:
            raise ValueError("category partitions require the explicit category readout")
        if self.modality in {"text", "image"} and self.semantic is None:
            raise ValueError("text/image queries require matched native semantic evidence")
        if self.modality in {"point_2d", "point_3d"} and self.positive is None:
            raise ValueError("point queries require registered positive support")
        if self.exact_positive and (self.modality != "point_3d" or self.positive is None):
            raise ValueError("estimated identities and pixel rays are not exact positive elements")
        for vector, dim in [(self.semantic, semantic_dim), (self.local, local_dim)]:
            if vector is not None:
                if vector.ndim not in (1, 2) or vector.shape[-1] != dim:
                    raise ValueError("query and native feature dimensions differ")
                if not bool(torch.isfinite(vector).all()) or bool((vector.norm(dim=-1) <= 0).any()):
                    raise ValueError("query features must be finite and nonzero")
        if self.semantic is not None and self.semantic.numel() != semantic_dim:
            raise ValueError("one query packet contains one semantic query")
        for support in [self.positive, self.negative]:
            if support is not None:
                support.validate(count)
        if self.modality != "point_2d" and self.positive is not None and self.negative is not None:
            positive = self.positive.indices[self.positive.weights > 0].cpu()
            negative = self.negative.indices[self.negative.weights > 0].cpu()
            if bool(torch.isin(positive, negative).any()):
                raise ValueError("contradictory positive/negative element support")
        if self.scale is not None and (not torch.isfinite(torch.tensor(self.scale)) or self.scale <= 0):
            raise ValueError("scale must be finite and positive")


@dataclass(frozen=True)
class QueryResult:
    semantic_scores: torch.Tensor | None
    membership_scores: torch.Tensor
    selected: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_weights: torch.Tensor
    scale: float
    diagnostics: dict
