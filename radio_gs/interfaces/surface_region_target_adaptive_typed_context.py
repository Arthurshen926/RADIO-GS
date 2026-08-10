"""Target-scene adaptive typed-context authority for target AcceptedV2 rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.interfaces.surface_region_typed_context import typed_context_source_access
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
    ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
    ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
    adaptive_typed_context_overlay_contract,
    validate_adaptive_typed_context_authority,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA = (
    "radio_gs.target_adaptive_typed_context_authority.v1"
)
TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION = 1


def target_adaptive_access_audit() -> dict[str, bool]:
    return {
        "query_independent": True,
        "target_geometry_authorities_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "target_metrics_computed": False,
    }


def target_adaptive_typed_context_contract() -> dict[str, Any]:
    return {
        "schema": TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA,
        "schema_version": TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION,
        "selection_and_carrier_contract_sha256": (
            ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256
        ),
        "selection_and_carrier_contract": adaptive_typed_context_overlay_contract(),
        "identity": "explicit_target_accepted_v2_physical_space_authority",
        "accepted_v2_descriptor_copied": False,
        "query_relevance_computed": False,
        "access_audit": target_adaptive_access_audit(),
    }


TARGET_ADAPTIVE_TYPED_CONTEXT_CONTRACT_SHA256 = canonical_json_sha256(
    target_adaptive_typed_context_contract()
)


def validate_target_adaptive_typed_context_authority(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("target adaptive typed-context authority must be a mapping")
    payload = dict(value)
    source_keys = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "scale_indices",
        "anchor_local_rows",
        "anchor_global_rows",
        "pooled_context_radio_direction",
        "typed_context_statistics",
        "context_present",
        "selection_complete",
        "typed_context_valid",
        "candidate_termination",
        "final_probe_width",
        "settled_candidate_count",
        "adaptive_round_count",
        "context_token_count",
        "context_token_row_offsets",
        "context_token_local_rows",
        "context_token_global_rows",
        "memory_audit",
        "channel_sha256",
    }
    required = source_keys | {"physical_space_authority", "access_audit"}
    contract = target_adaptive_typed_context_contract()
    if (
        set(payload) != required
        or payload.get("schema") != TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA
        or payload.get("schema_version") != TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != TARGET_ADAPTIVE_TYPED_CONTEXT_CONTRACT_SHA256
        or payload.get("access_audit") != target_adaptive_access_audit()
        or "accepted_v2_e0" in payload
    ):
        raise ValueError("target adaptive typed-context contract differs")
    physical = payload.get("physical_space_authority")
    if not isinstance(physical, Mapping):
        raise ValueError("target adaptive physical-space authority differs")
    expected_physical = target_physical_space_authority(
        dataset_id=physical.get("dataset_id"),
        scene_id=physical.get("scene_id"),
        geometry_checkpoint_sha256=physical.get("geometry_checkpoint_sha256"),
    )
    if (
        dict(physical) != expected_physical
        or payload.get("scene_id") != expected_physical["scene_id"]
        or payload.get("physical_space_id") != expected_physical["physical_space_id"]
    ):
        raise ValueError("target adaptive physical-space binding differs")

    # The tensor/CSR/memory mathematics are intentionally identical to the
    # frozen adaptive-v2 carrier.  Validate a transient source-schema view to
    # reuse those strict checks without publishing or accepting a fake ScanNet
    # identity.  No target file is opened here.
    proxy = {name: payload[name] for name in source_keys}
    proxy.update(
        {
            "schema": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
            "schema_version": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
            "contract": adaptive_typed_context_overlay_contract(),
            "contract_sha256": ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
            "scene_id": "scene0000_00",
            "physical_space_id": "scene0000",
            "source_access": typed_context_source_access(),
        }
    )
    validated = validate_adaptive_typed_context_authority(proxy)
    result = dict(payload)
    for name in (
        "canonical_region_indices",
        "pooled_context_radio_direction",
        "typed_context_statistics",
        "context_token_row_offsets",
        "context_token_local_rows",
        "context_token_global_rows",
    ):
        result[name] = validated[name]
    result["physical_space_authority"] = expected_physical
    return result


__all__ = [
    "TARGET_ADAPTIVE_TYPED_CONTEXT_CONTRACT_SHA256",
    "TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA",
    "TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION",
    "target_adaptive_access_audit",
    "target_adaptive_typed_context_contract",
    "validate_target_adaptive_typed_context_authority",
]
