"""Shared query interface for the canonical RADIO-GS feature field.

The query package contains both lightweight benchmark adapters and optional
3-D graph utilities backed by SciPy.  Keep the public API lazy so importing a
lightweight submodule (for example the official-SAM3 transient adapter) does
not initialize unrelated graph dependencies or their binary extensions.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "QueryKind": (".unified_query", "QueryKind"),
    "QuerySpace": (".unified_query", "QuerySpace"),
    "QuerySpec": (".unified_query", "QuerySpec"),
    "SupportGraph": (".unified_query", "SupportGraph"),
    "SupportPropagationConfig": (".unified_query", "SupportPropagationConfig"),
    "binary_mask": (".unified_query", "binary_mask"),
    "build_support_graph": (".unified_query", "build_support_graph"),
    "cosine_bank_torch": (".unified_query", "cosine_bank_torch"),
    "cosine_margin_torch": (".unified_query", "cosine_margin_torch"),
    "cosine_relevancy_torch": (".unified_query", "cosine_relevancy_torch"),
    "margin_to_relevancy_torch": (".unified_query", "margin_to_relevancy_torch"),
    "propagate_support": (".unified_query", "propagate_support"),
    "seed_connected_component": (".unified_query", "seed_connected_component"),
    "score_feature_map": (".unified_query", "score_feature_map"),
    "score_features": (".unified_query", "score_features"),
    "CanonicalQueryEngine": (".query_engine", "CanonicalQueryEngine"),
    "QueryResult": (".query_engine", "QueryResult"),
    "SceneSpaceCalibration": (".score_calibration", "SceneSpaceCalibration"),
    "deterministic_sample_rows": (".score_calibration", "deterministic_sample_rows"),
    "fit_scene_space_calibration": (".score_calibration", "fit_scene_space_calibration"),
    "robust_tanh_score_calibration": (
        ".score_calibration",
        "robust_tanh_score_calibration",
    ),
    "fuse_registered_observation_unary": (
        ".evidence_scorer",
        "fuse_registered_observation_unary",
    ),
    "registered_observation_anchor_only_confidence": (
        ".evidence_scorer",
        "registered_observation_anchor_only_confidence",
    ),
    "registered_observation_anchor_mask": (
        ".evidence_scorer",
        "registered_observation_anchor_mask",
    ),
    "registered_observation_effective_confidence": (
        ".evidence_scorer",
        "registered_observation_effective_confidence",
    ),
    "registered_seed_observation": (
        ".evidence_scorer",
        "registered_seed_observation",
    ),
    "registered_seed_unary": (".evidence_scorer", "registered_seed_unary"),
    "registered_raster_adjoint_observation": (
        ".evidence_scorer",
        "registered_raster_adjoint_observation",
    ),
    "shrink_unary_by_reliability": (
        ".evidence_scorer",
        "shrink_unary_by_reliability",
    ),
    "compile_registered_primitive_seeds": (
        ".query_compilers",
        "compile_registered_primitive_seeds",
    ),
    "geometric_consensus_unary": (
        ".reliability_fusion",
        "geometric_consensus_unary",
    ),
    "symmetric_bernoulli_product_of_experts": (
        ".reliability_fusion",
        "symmetric_bernoulli_product_of_experts",
    ),
    "PrimitiveUnaryEvidence": (".query_spec", "PrimitiveUnaryEvidence"),
    "PrototypeSet": (".query_spec", "PrototypeSet"),
    "QueryIntent": (".query_spec", "QueryIntent"),
    "QueryModality": (".query_spec", "QueryModality"),
    "TypedQuerySpec": (".query_spec", "QuerySpec"),
    "RegistrationMode": (".query_spec", "RegistrationMode"),
    "SelectionMode": (".query_spec", "SelectionMode"),
    "SoftSeedGroups": (".query_spec", "SoftSeedGroups"),
    "SoftSeedSet": (".query_spec", "SoftSeedSet"),
    "MonotoneLikelihoodRatioHead": (
        ".query_likelihood_head",
        "MonotoneLikelihoodRatioHead",
    ),
    "MonotoneChannelDensityRatioHead": (
        ".query_likelihood_head",
        "MonotoneChannelDensityRatioHead",
    ),
    "MonotoneOneSidedDensityRatioHead": (
        ".query_likelihood_head",
        "MonotoneOneSidedDensityRatioHead",
    ),
    "MonotoneSignedLikelihoodRatioHead": (
        ".query_likelihood_head",
        "MonotoneSignedLikelihoodRatioHead",
    ),
    "MonotoneQueryLikelihoodHead": (
        ".query_likelihood_head",
        "MonotoneQueryLikelihoodHead",
    ),
    "QueryLikelihoodInputs": (".query_likelihood_head", "QueryLikelihoodInputs"),
    "PrimitiveSupportGraph": (".support_solver", "PrimitiveSupportGraph"),
    "SupportGraphConfig": (".support_solver", "SupportGraphConfig"),
    "SupportSolverConfig": (".support_solver", "SupportSolverConfig"),
    "build_primitive_support_graph": (
        ".support_solver",
        "build_primitive_support_graph",
    ),
    "graph_for_query_intent": (".support_solver", "graph_for_query_intent"),
    "mix_support_graph_channels": (
        ".support_solver",
        "mix_support_graph_channels",
    ),
    "select_support_components": (".support_solver", "select_support_components"),
    "solve_primitive_support": (".support_solver", "solve_primitive_support"),
    "DIFFERENT_RELATION": (".latent_proposal_posterior", "DIFFERENT_RELATION"),
    "LatentProposalPosterior": (
        ".latent_proposal_posterior",
        "LatentProposalPosterior",
    ),
    "SAME_RELATION": (".latent_proposal_posterior", "SAME_RELATION"),
    "UNKNOWN_RELATION": (".latent_proposal_posterior", "UNKNOWN_RELATION"),
    "latent_proposal_null_posterior": (
        ".latent_proposal_posterior",
        "latent_proposal_null_posterior",
    ),
    "ternary_comembership_authority": (
        ".latent_proposal_posterior",
        "ternary_comembership_authority",
    ),
    "QueryAbstention": (
        ".synchronous_multiview_candidate_marginal",
        "QueryAbstention",
    ),
    "SynchronousMultiviewCandidateMarginal": (
        ".synchronous_multiview_candidate_marginal",
        "SynchronousMultiviewCandidateMarginal",
    ),
    "deterministic_visible_signed_points": (
        ".synchronous_multiview_candidate_marginal",
        "deterministic_visible_signed_points",
    ),
    "marginalize_synchronous_multiview_candidates": (
        ".synchronous_multiview_candidate_marginal",
        "marginalize_synchronous_multiview_candidates",
    ),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    """Resolve the historical flat API only when an attribute is requested."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
