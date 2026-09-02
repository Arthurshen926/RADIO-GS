"""Learned unknown-surface completion for the v4 object memory."""

from .oracle import (
    OracleIdentityCompletionMLP,
    PartialObjectMembership,
    TokenContext,
    build_feature_cosine_similarity,
    build_pair_features,
    build_token_context,
    complete_unknown_only,
    completion_metrics,
)
from .message_passing import (
    EDGE_FEATURE_DIMENSION,
    EDGE_FEATURE_LAYOUT,
    EXTENT_GATE_INITIAL_LOGIT,
    F71_FEATURE_DIMENSION,
    EdgeCompatibilityMLP,
    SurfaceMessagePassing,
    SurfaceMessagePassingOutput,
    build_query_free_edge_features,
    validate_surface_voxel_adjacency,
)
from .structured_extent import (
    STRUCTURED_EXTENT_ITERATION_COUNT,
    STRUCTURED_EXTENT_MODES,
    StructuredExtentMode,
    StructuredExtentOutput,
    TokenConditionedStructuredExtent,
)

__all__ = [
    "OracleIdentityCompletionMLP",
    "PartialObjectMembership",
    "TokenContext",
    "build_feature_cosine_similarity",
    "build_pair_features",
    "build_token_context",
    "complete_unknown_only",
    "completion_metrics",
    "EDGE_FEATURE_DIMENSION",
    "EDGE_FEATURE_LAYOUT",
    "EXTENT_GATE_INITIAL_LOGIT",
    "F71_FEATURE_DIMENSION",
    "EdgeCompatibilityMLP",
    "SurfaceMessagePassing",
    "SurfaceMessagePassingOutput",
    "build_query_free_edge_features",
    "validate_surface_voxel_adjacency",
    "STRUCTURED_EXTENT_ITERATION_COUNT",
    "STRUCTURED_EXTENT_MODES",
    "StructuredExtentMode",
    "StructuredExtentOutput",
    "TokenConditionedStructuredExtent",
]
