"""Shared query interface for the canonical RADIO-GS feature field."""

from .unified_query import (
    QueryKind,
    QuerySpace,
    QuerySpec,
    SupportGraph,
    SupportPropagationConfig,
    binary_mask,
    build_support_graph,
    cosine_bank_torch,
    cosine_margin_torch,
    cosine_relevancy_torch,
    margin_to_relevancy_torch,
    propagate_support,
    seed_connected_component,
    score_feature_map,
    score_features,
)

__all__ = [
    "QueryKind",
    "QuerySpace",
    "QuerySpec",
    "SupportGraph",
    "SupportPropagationConfig",
    "binary_mask",
    "build_support_graph",
    "cosine_bank_torch",
    "cosine_margin_torch",
    "cosine_relevancy_torch",
    "margin_to_relevancy_torch",
    "propagate_support",
    "seed_connected_component",
    "score_feature_map",
    "score_features",
]
