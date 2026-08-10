"""Fail-closed interface and trainer hook for the isolated V2.1B candidate.

The interface deliberately contains no CLI, optimizer, loss, evaluator,
renderer, artifact writer, or file reader.  It makes the future trainer change
explicit while leaving the running V2.1 baseline byte-untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
)
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
)
from radio_gs.models.surface_region_v21b_reliability_conditioned_residual import (
    ReliabilityConditionedResidualV21BOutput,
    SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    V21B_RELIABILITY_COMPONENT_NAMES,
    V21B_RELIABILITY_COMPONENT_NAMES_SHA256,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SURFACE_REGION_V21B_INTERFACE_SCHEMA = (
    "radio_gs.surface_region_v21b_reliability_conditioned_rank256_interface.v1"
)
SURFACE_REGION_V21B_INTERFACE_SCHEMA_VERSION = 1
SURFACE_REGION_V21B_PREREGISTRATION_SHA256 = (
    "5397e29ece42b03c8af0b77865a9c8327920c59f0be050dbd99eb8ca91ece60a"
)


def source_access() -> dict[str, bool]:
    """Declare every disallowed supervision channel for this interface."""

    return {
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
        "runtime_query_strings_consumed": False,
        "per_scene_hyperparameters": False,
        "scene_identifiers_consumed_by_model": False,
    }


def interface_contract() -> dict[str, Any]:
    """Return the immutable method and integration boundary."""

    return {
        "schema": SURFACE_REGION_V21B_INTERFACE_SCHEMA,
        "schema_version": SURFACE_REGION_V21B_INTERFACE_SCHEMA_VERSION,
        "preregistration": {
            "path": (
                "paper/artifacts/"
                "surface_region_v21b_reliability_conditioned_rank256_"
                "preregistration_20260807.json"
            ),
            "sha256": SURFACE_REGION_V21B_PREREGISTRATION_SHA256,
        },
        "model_class": (
            "SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B"
        ),
        "immutable_external_base": "accepted_v2_e0",
        "input_keys": [
            "accepted_v2_e0",
            "pooled_context_radio_direction",
            "raw_full_scalar_summary",
            "typed_context_statistics",
        ],
        "routing_inputs": ["declared_active_mask", "effective_ood_mask"],
        "loss_input": "forward_output.semantic_descriptor",
        "source_normalization_inputs": ["median", "robust_scale"],
        "capacity": {
            "hidden_rank": 256,
            "descriptor_dim": 1536,
            "context_dim": 1280,
            "combined_scalar_dim": 30,
        },
        "reliability": {
            "component_names": list(V21B_RELIABILITY_COMPONENT_NAMES),
            "component_names_sha256": V21B_RELIABILITY_COMPONENT_NAMES_SHA256,
            "aggregation": "fixed_unweighted_mean_clamped_0_1",
            "full_scalar_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "typed_context_statistic_names_sha256": (
                TYPED_CONTEXT_STATISTIC_NAMES_SHA256
            ),
            "angle_budget_radians": "0.15_plus_0.60_times_reliability",
            "learned": False,
            "scene_parameterized": False,
        },
        "geometry": {
            "residual_gauge": "accepted_v2_unit_sphere_tangent_plane",
            "candidate_map": "unit_sphere_exponential_map",
            "low_reliability_angle_radians": 0.15,
            "high_reliability_angle_radians": 0.75,
            "maximum_hidden_to_tangent_gain": 0.25,
        },
        "fallback": (
            "inactive_or_ood_bitwise_accepted_v2_without_residual_gradient"
        ),
        "initialization": (
            "zero_final_projection_bitwise_identity_with_first_step_gradient"
        ),
        "future_trainer_integration": {
            "constructor": "build_model_from_source_normalization",
            "complete_scene_forward": "forward_complete_scene",
            "routing": "reuse_frozen_v21_declared_and_effective_ood_masks",
            "losses": "reuse_frozen_v21_base_response_and_typed_relation_losses",
            "optimizer": "construct_only_after_separate_execution_authority",
            "checkpoint_and_source_gate": (
                "require_new_v21b_schema_and_source_only_promotion_contract"
            ),
        },
        "source_access": source_access(),
    }


SURFACE_REGION_V21B_INTERFACE_CONTRACT_SHA256 = canonical_json_sha256(
    interface_contract()
)


def build_model_from_source_normalization(
    normalization: Mapping[str, Any],
) -> SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B:
    """Construct V2.1B solely from the frozen source-fit normalization."""

    if not isinstance(normalization, Mapping):
        raise ValueError("V2.1B normalization must be a mapping")
    if "median" not in normalization or "robust_scale" not in normalization:
        raise ValueError("V2.1B normalization buffers are missing")
    return SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B(
        scalar_median=torch.as_tensor(normalization["median"]),
        scalar_robust_scale=torch.as_tensor(normalization["robust_scale"]),
    )


def forward_complete_scene(
    model: SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    scene: Mapping[str, Any],
    *,
    declared_active_mask: torch.Tensor,
    effective_ood_mask: torch.Tensor,
    device: torch.device,
) -> ReliabilityConditionedResidualV21BOutput:
    """The exact drop-in forward hook for a separately authorized trainer."""

    if not isinstance(
        model,
        SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
    ):
        raise TypeError("V2.1B forward requires the registered model class")
    if not isinstance(scene, Mapping):
        raise ValueError("V2.1B scene must be a mapping")
    required = {
        "accepted_v2_e0",
        "pooled_context_radio_direction",
        "raw_full_scalar_summary",
        "typed_context_statistics",
    }
    if not required.issubset(scene):
        raise ValueError("V2.1B scene inputs are incomplete")
    base = torch.as_tensor(scene["accepted_v2_e0"])
    if base.ndim != 2:
        raise ValueError("V2.1B complete-scene base must be rank two")
    row_count = int(base.shape[0])
    declared = torch.as_tensor(declared_active_mask).detach().cpu()
    ood = torch.as_tensor(effective_ood_mask).detach().cpu()
    if (
        declared.dtype != torch.bool
        or ood.dtype != torch.bool
        or declared.shape != (row_count,)
        or ood.shape != (row_count,)
    ):
        raise ValueError("V2.1B routing masks differ from complete scene rows")
    return model.forward_with_diagnostics(
        base.to(device),
        torch.as_tensor(scene["pooled_context_radio_direction"]).to(device),
        torch.as_tensor(scene["raw_full_scalar_summary"]).to(device),
        torch.as_tensor(scene["typed_context_statistics"]).to(device),
        active_mask=declared.to(device),
        ood_mask=ood.to(device),
    )


__all__ = [
    "SURFACE_REGION_V21B_INTERFACE_CONTRACT_SHA256",
    "SURFACE_REGION_V21B_INTERFACE_SCHEMA",
    "SURFACE_REGION_V21B_INTERFACE_SCHEMA_VERSION",
    "SURFACE_REGION_V21B_PREREGISTRATION_SHA256",
    "build_model_from_source_normalization",
    "forward_complete_scene",
    "interface_contract",
    "source_access",
]
