"""Fail-closed checks for the frozen five-benchmark Method-v1 field."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


METHOD_ID = "radio-gs-method-v1"
FIELD_SCHEMA_VERSION = 2
FEATURE_DIM = 1280
COEFFICIENT_DIM = 512
LOCAL_DIM = 512
STAGE_WEIGHT = 0.05
GENERIC_COMPONENTS = ["profile", "listwise", "sibling", "synonym"]


class MethodV1ValidationError(ValueError):
    """Raised when an artifact cannot belong to Method-v1."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MethodV1ValidationError(message)


def validate_method_authority(authority: Mapping[str, Any]) -> None:
    """Validate the immutable choices that identify Method-v1."""

    _require(authority.get("schema_version") == 1, "authority schema differs")
    _require(authority.get("method_id") == METHOD_ID, "method identity differs")
    _require(
        authority.get("status") == "frozen_joint_development_baseline",
        "method is not frozen",
    )
    state = authority.get("persistent_scene_state")
    _require(isinstance(state, Mapping), "persistent scene state is missing")
    expected_state = {
        "count": 1,
        "field_schema_version": FIELD_SCHEMA_VERSION,
        "canonical_feature_dim": FEATURE_DIM,
        "coefficient_dim": COEFFICIENT_DIM,
        "local_dim": LOCAL_DIM,
        "trainable_reliability_state": False,
        "query_dependent_state": False,
        "stored_dino_field": False,
        "stored_sam_field": False,
        "stored_text_field": False,
    }
    for key, expected in expected_state.items():
        _require(state.get(key) == expected, f"persistent scene state {key} differs")

    construction = authority.get("construction")
    _require(isinstance(construction, Mapping), "construction authority is missing")
    _require(
        construction.get("registration_observation_contract")
        == "canonical-exact-marginal-mpr-v1",
        "registration contract differs",
    )
    _require(
        construction.get("field_observation_contract")
        == "canonical-factorized-radio-v1",
        "field contract differs",
    )
    _require(
        construction.get("capability_gradients") == "tangent_only",
        "capability gauge rule differs",
    )
    stages = construction.get("stage_order")
    _require(isinstance(stages, list), "construction stages are missing")
    _require(
        [stage.get("stage") for stage in stages if isinstance(stage, Mapping)]
        == [
            "factorized_d512_l512",
            "official_siglip2_full_grid",
            "genuine_source_crop_region_summary",
            "target_blind_generic_text_response",
        ],
        "construction stage order differs",
    )


def validate_complete_field_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject base-only, legacy, or partially fine-tuned field checkpoints."""

    _require(
        payload.get("schema_version") == FIELD_SCHEMA_VERSION, "field schema differs"
    )
    _require(
        payload.get("checkpoint_contract")
        == "canonical-factorized-radio-checkpoint-v1",
        "factorized checkpoint contract differs",
    )
    architecture = payload.get("architecture")
    _require(isinstance(architecture, Mapping), "field architecture is missing")
    expected_architecture = {
        "feature_dim": FEATURE_DIM,
        "coefficient_dim": COEFFICIENT_DIM,
        "local_dim": LOCAL_DIM,
        "fusion_reliability": False,
    }
    for key, expected in expected_architecture.items():
        _require(architecture.get(key) == expected, f"field architecture {key} differs")

    reliability = payload.get("reliability")
    _require(
        getattr(reliability, "ndim", -1) == 2 and int(reliability.shape[1]) == 0,
        "field persists target reliability",
    )
    render = payload.get("render_optimization")
    _require(
        isinstance(render, Mapping), "field is base-only; Method-v1 stages are absent"
    )
    _require(render.get("train_basis") is False, "persistent basis was not frozen")
    _require(render.get("train_fusion") is False, "persistent fusion was not frozen")
    _require(
        render.get("benchmark_masks_opened") is False, "benchmark masks were opened"
    )
    _require(
        render.get("benchmark_labels_opened") is False, "benchmark labels were opened"
    )
    _require(render.get("text_queries_opened") is False, "benchmark text was opened")

    official = render.get("official_render_capability")
    _require(
        isinstance(official, Mapping) and official.get("enabled") is True,
        "official spatial stage is absent",
    )
    _require(
        official.get("adaptor_weights") == {"siglip2-g": STAGE_WEIGHT},
        "official spatial weight differs",
    )
    _require(
        official.get("projection_order")
        == "complete_rendered_2d_grid_vs_resample(official_runtime_adaptor_output)",
        "official SigLIP2 projection is not full-grid",
    )
    _require(
        official.get("custom_adaptor_head") is False, "custom spatial head is present"
    )

    semantic = render.get("semantic_capability")
    _require(
        isinstance(semantic, Mapping) and semantic.get("enabled") is True,
        "genuine crop-summary stage is absent",
    )
    _require(semantic.get("weight") == STAGE_WEIGHT, "crop-summary weight differs")
    _require(
        semantic.get("uses_benchmark_masks") is False,
        "crop-summary used benchmark masks",
    )
    _require(
        semantic.get("uses_text_queries") is False, "crop-summary used benchmark text"
    )

    generic = render.get("generic_text_response")
    _require(
        isinstance(generic, Mapping) and generic.get("enabled") is True,
        "generic response stage is absent",
    )
    _require(generic.get("weight") == STAGE_WEIGHT, "generic response weight differs")
    _require(
        generic.get("components") == GENERIC_COMPONENTS,
        "generic response components differ",
    )
    _require(
        generic.get("benchmark_text_queries_opened") is False,
        "generic stage used benchmark text",
    )
    _require(
        generic.get("uses_benchmark_masks") is False,
        "generic stage used benchmark masks",
    )
    _require(
        generic.get("uses_target_metrics_for_selection") is False,
        "generic stage used target metrics",
    )

    return {
        "method_id": METHOD_ID,
        "field_schema_version": FIELD_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "coefficient_dim": COEFFICIENT_DIM,
        "local_dim": LOCAL_DIM,
        "construction_stages_complete": True,
    }


__all__ = [
    "MethodV1ValidationError",
    "validate_complete_field_payload",
    "validate_method_authority",
]
