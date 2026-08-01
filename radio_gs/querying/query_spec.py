"""Typed evidence specification for query-consistent canonical fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field.field_signature import FeatureSpaceSignature


class QueryModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    REGISTERED_2D = "registered_2d"
    WORLD_3D = "world_3d"


class QueryIntent(str, Enum):
    CATEGORY = "category"
    INSTANCE = "instance"
    PART = "part"
    REGION = "region"


class RegistrationMode(str, Enum):
    NONE = "none"
    CAMERA = "camera"
    WORLD = "world"


class SelectionMode(str, Enum):
    ALL_COMPONENTS = "all_components"
    TOP_COMPONENT = "top_component"
    SEEDED_COMPONENT = "seeded_component"
    # Optional interaction readout: choose the smallest set of clean active
    # components that covers every positive interaction group.  It is never
    # selected implicitly, so existing benchmark contracts are unchanged.
    MIN_SEED_COVER = "min_seed_cover"
    TOP_K = "top_k"


@dataclass(frozen=True)
class PrototypeSet:
    features: torch.Tensor
    signature: FeatureSpaceSignature
    weights: torch.Tensor | None = None
    negatives: torch.Tensor | None = None

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.features).float()
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or min(values.shape) <= 0:
            raise ValueError("prototype features must be [K,D]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("prototype features contain NaN or infinity")
        values = F.normalize(values, dim=-1, eps=1e-8)
        object.__setattr__(self, "features", values)
        if self.weights is None:
            weights = torch.ones(values.shape[0], dtype=torch.float32) / values.shape[0]
        else:
            weights = torch.as_tensor(self.weights).float().reshape(-1)
            if weights.shape != (values.shape[0],) or bool((weights < 0).any()):
                raise ValueError("prototype weights must be non-negative [K]")
            weights = weights / weights.sum().clamp_min(1e-8)
        object.__setattr__(self, "weights", weights)
        if self.negatives is not None:
            negatives = torch.as_tensor(self.negatives).float()
            if negatives.ndim == 1:
                negatives = negatives[None]
            if negatives.ndim != 2 or negatives.shape[1] != values.shape[1]:
                raise ValueError("negative prototypes must be [M,D]")
            object.__setattr__(
                self, "negatives", F.normalize(negatives, dim=-1, eps=1e-8)
            )


@dataclass(frozen=True)
class SoftSeedSet:
    weights: torch.Tensor
    source: str
    normalization: str = "independent_max"

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.weights).float().reshape(-1)
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("seed weights must be a finite non-empty vector")
        if bool((values < 0).any()):
            raise ValueError("seed weights cannot be negative")
        normalization = str(self.normalization)
        if normalization not in {"independent_max", "none"}:
            raise ValueError(
                "seed normalization must be independent_max or none"
            )
        if normalization == "independent_max":
            maximum = values.max()
            if maximum > 0:
                values = values / maximum
        elif bool((values > 1).any()):
            raise ValueError("unnormalized seed weights must be in [0,1]")
        object.__setattr__(self, "weights", values)
        object.__setattr__(self, "normalization", normalization)


@dataclass(frozen=True)
class SoftSeedGroups:
    """Keep one primitive-responsibility column for every interaction event."""

    weights: torch.Tensor
    source: str
    normalization: str = "independent_max"

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.weights).float()
        if (
            values.ndim != 2
            or min(values.shape) <= 0
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("seed groups must be a finite non-empty [N,K] matrix")
        if bool((values < 0).any()) or not bool((values > 0).any(dim=0).all()):
            raise ValueError("every seed group must have non-negative support")
        normalization = str(self.normalization)
        if normalization not in {"independent_max", "none"}:
            raise ValueError(
                "seed-group normalization must be independent_max or none"
            )
        if normalization == "independent_max":
            values = values / values.amax(dim=0, keepdim=True).clamp_min(1e-30)
        elif bool((values > 1).any()):
            raise ValueError("unnormalized seed-group weights must be in [0,1]")
        object.__setattr__(self, "weights", values)
        object.__setattr__(self, "normalization", normalization)


@dataclass(frozen=True)
class PrimitiveUnaryEvidence:
    """Bounded signed evidence already registered to primitive rows.

    ``values`` stores signed purity multiplied by observation confidence.
    ``confidence`` keeps that joint observation mass explicit so registered
    evidence can be fused in probability space without treating an
    unobserved row (zero) as an ambiguous 50/50 observation.  It may be
    omitted only for legacy additive use; probability fusion fails closed
    without it.
    """

    values: torch.Tensor
    source: str
    confidence: torch.Tensor | None = None

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values).float().reshape(-1)
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("primitive unary evidence must be a finite vector")
        if bool((values < -1).any()) or bool((values > 1).any()):
            raise ValueError("primitive unary evidence must be in [-1,1]")
        object.__setattr__(self, "values", values)
        if self.confidence is None:
            return
        confidence = torch.as_tensor(self.confidence).float().reshape(-1)
        if (
            confidence.shape != values.shape
            or not bool(torch.isfinite(confidence).all())
            or bool((confidence < 0).any())
            or bool((confidence > 1).any())
        ):
            raise ValueError(
                "primitive unary confidence must align with values in [0,1]"
            )
        if bool((values.abs() > confidence + 1e-6).any()):
            raise ValueError(
                "signed primitive unary magnitude cannot exceed its confidence"
            )
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class QuerySpec:
    modality: QueryModality
    intent: QueryIntent
    registration: RegistrationMode
    semantic_evidence: PrototypeSet | None = None
    appearance_evidence: PrototypeSet | None = None
    boundary_evidence: PrototypeSet | None = None
    positive_seeds: SoftSeedSet | None = None
    negative_seeds: SoftSeedSet | None = None
    positive_seed_groups: SoftSeedGroups | None = None
    negative_seed_groups: SoftSeedGroups | None = None
    primitive_unary_evidence: PrimitiveUnaryEvidence | None = None
    granularity_m: tuple[float, ...] = ()
    selection_mode: SelectionMode = SelectionMode.ALL_COMPONENTS
    field_signature: FeatureSpaceSignature | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", QueryModality(self.modality))
        object.__setattr__(self, "intent", QueryIntent(self.intent))
        object.__setattr__(self, "registration", RegistrationMode(self.registration))
        object.__setattr__(self, "selection_mode", SelectionMode(self.selection_mode))
        granularities = tuple(float(v) for v in self.granularity_m)
        if any(v <= 0 for v in granularities):
            raise ValueError("granularity_m must contain positive scales")
        object.__setattr__(self, "granularity_m", granularities)
        if not any(
            value is not None
            for value in (
                self.semantic_evidence,
                self.appearance_evidence,
                self.boundary_evidence,
                self.positive_seeds,
                self.primitive_unary_evidence,
            )
        ):
            raise ValueError("query contains no evidence")
        if self.modality is QueryModality.TEXT and self.registration is not RegistrationMode.NONE:
            raise ValueError("text queries are unregistered")
        if self.modality is QueryModality.REGISTERED_2D and self.registration is not RegistrationMode.CAMERA:
            raise ValueError("registered 2D queries require camera registration")
        if (
            self.primitive_unary_evidence is not None
            and self.modality is not QueryModality.REGISTERED_2D
        ):
            raise ValueError(
                "primitive unary evidence is currently restricted to registered 2D queries"
            )
        if self.modality is QueryModality.WORLD_3D and self.registration is not RegistrationMode.WORLD:
            raise ValueError("world 3D queries require world registration")
        if (
            self.primitive_unary_evidence is not None
            and self.positive_seeds is not None
            and self.primitive_unary_evidence.values.shape
            != self.positive_seeds.weights.shape
        ):
            raise ValueError(
                "primitive unary evidence and registered seeds must align"
            )
        for name, groups, aggregate in (
            ("positive", self.positive_seed_groups, self.positive_seeds),
            ("negative", self.negative_seed_groups, self.negative_seeds),
        ):
            if groups is None:
                continue
            if aggregate is None:
                raise ValueError(f"{name} seed groups require aggregate seeds")
            if groups.weights.shape[0] != aggregate.weights.numel():
                raise ValueError(f"{name} seed groups and aggregate seeds must align")

    def assert_field_compatible(self, signature: FeatureSpaceSignature) -> None:
        if self.field_signature is not None:
            self.field_signature.assert_compatible(signature)
