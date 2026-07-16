"""Frozen official RADIO capability interfaces."""

from .capability_cache import (
    CanonicalCapabilityBank,
    CanonicalPrimitiveReliability,
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)

from .frozen_radio_views import (
    FrozenRadioViews,
    OfficialCropSummaryRuntime,
    OfficialRadioRuntime,
)
from .semantic_alignment import (
    GlobalSemanticBridgeManifest,
    GlobalRegionSummaryBridge,
    project_dense_region_semantics,
    SemanticAlignmentDecision,
    SemanticAlignmentPolicy,
    SemanticAlignmentStage,
    SemanticOracleResult,
)
from .surface_region_summary import (
    SURFACE_GEOMETRY_DIM,
    SurfaceRegionSummaryReadout,
    surface_region_geometry,
)

__all__ = [
    "CanonicalCapabilityBank",
    "CanonicalPrimitiveReliability",
    "FrozenRadioViews",
    "GlobalSemanticBridgeManifest",
    "GlobalRegionSummaryBridge",
    "OfficialCropSummaryRuntime",
    "OfficialRadioRuntime",
    "project_dense_region_semantics",
    "SemanticAlignmentDecision",
    "SemanticAlignmentPolicy",
    "SemanticAlignmentStage",
    "SemanticOracleResult",
    "SURFACE_GEOMETRY_DIM",
    "SurfaceRegionSummaryReadout",
    "load_canonical_capability_bank",
    "load_canonical_primitive_reliability",
    "load_canonical_support_graph",
    "surface_region_geometry",
]
