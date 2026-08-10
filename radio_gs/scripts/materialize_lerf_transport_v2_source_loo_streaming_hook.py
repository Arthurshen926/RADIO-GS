#!/usr/bin/env python3
"""Low-memory source-only LOO hook for residual-shrinkage transport v2.

The hook is deliberately not a standalone scene reader.  A future source
teacher materializer calls it while its canonical retained-view tensors are
already resident, before those tensors are released.  Only scalar per-scene
statistics leave the hook; source view descriptors are never made durable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from radio_gs.interfaces.lerf_scale_residual_shrinkage_transport import (
    RESIDUAL_SHRINKAGE_CONTRACT_SHA256,
    SOURCE_LOO_SCHEMA,
    source_only_leave_one_view_out_residual_shrinkage_audit,
    validate_source_only_residual_shrinkage_audit,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


HOOK_SCHEMA = "radio_gs.lerf_transport_v2_source_loo_streaming_hook.v1"


def hook_contract() -> dict[str, Any]:
    return {
        "schema": HOOK_SCHEMA,
        "schema_version": 1,
        "residual_shrinkage_contract_sha256": (RESIDUAL_SHRINKAGE_CONTRACT_SHA256),
        "hook_point": (
            "after_canonical_top4_source_views_and_o0_three_scale_frame_are"
            "_available_before_top4_release"
        ),
        "inputs": {
            "top_descriptors": "source_only_rows_by_four_by_descriptor_dim",
            "top_frame_ids": "source_only_rows_by_four_negative_one_is_padding",
            "o0_descriptor_by_scale": "same_source_rows_by_three_by_descriptor_dim",
            "scene_id": "provenance_only_not_consumed_by_math",
        },
        "streaming": {
            "row_chunked_descriptor_compute": True,
            "compute_device": "cpu",
            "candidate_by_observation_scalar_matrix_transient": True,
            "source_view_descriptors_written": False,
            "target_data_written": False,
        },
        "output": "validated_per_scene_scalar_loo_audit_plus_provenance",
        "scene_or_query_specific_parameters": False,
        "query_embeddings_or_text_consumed": False,
        "target_images_labels_masks_metrics_consumed": False,
        "executes_target_metric": False,
        "authorizes_target_candidate": False,
    }


HOOK_CONTRACT_SHA256 = canonical_json_sha256(hook_contract())


@torch.inference_mode()
def capture_source_only_transport_v2_loo(
    *,
    scene_id: str,
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    row_chunk: int = 2048,
) -> dict[str, Any]:
    """Capture one scalar-only scene audit from already-resident tensors."""

    if not isinstance(scene_id, str) or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("transport-v2 source LOO scene id differs")
    audit = source_only_leave_one_view_out_residual_shrinkage_audit(
        top_descriptors,
        top_frame_ids,
        o0_descriptor_by_scale,
        row_chunk=row_chunk,
    )
    result = {
        "schema": HOOK_SCHEMA,
        "schema_version": 1,
        "hook_contract_sha256": HOOK_CONTRACT_SHA256,
        "scene_id": scene_id,
        "source_only_loo_audit": audit,
        "source_only_loo_audit_sha256": canonical_json_sha256(audit),
        "access_audit": {
            "source_top_descriptors_opened_from_caller_memory": True,
            "source_o0_descriptor_frame_opened_from_caller_memory": True,
            "source_view_descriptors_written": False,
            "query_embeddings_or_text_opened": False,
            "target_images_labels_masks_metrics_opened": False,
            "target_metric_executed": False,
        },
        "target_candidate_authorized": False,
    }
    validate_streaming_hook_capture(result)
    return result


def validate_streaming_hook_capture(value: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "hook_contract_sha256",
        "scene_id",
        "source_only_loo_audit",
        "source_only_loo_audit_sha256",
        "access_audit",
        "target_candidate_authorized",
    }
    audit = value.get("source_only_loo_audit")
    scene_id = value.get("scene_id")
    if (
        set(value) != required
        or value.get("schema") != HOOK_SCHEMA
        or value.get("schema_version") != 1
        or value.get("hook_contract_sha256") != HOOK_CONTRACT_SHA256
        or not isinstance(scene_id, str)
        or not scene_id
        or scene_id.strip() != scene_id
        or not isinstance(audit, Mapping)
        or audit.get("schema") != SOURCE_LOO_SCHEMA
        or value.get("source_only_loo_audit_sha256") != canonical_json_sha256(audit)
        or value.get("access_audit")
        != {
            "source_top_descriptors_opened_from_caller_memory": True,
            "source_o0_descriptor_frame_opened_from_caller_memory": True,
            "source_view_descriptors_written": False,
            "query_embeddings_or_text_opened": False,
            "target_images_labels_masks_metrics_opened": False,
            "target_metric_executed": False,
        }
        or value.get("target_candidate_authorized") is not False
    ):
        raise ValueError("transport-v2 source LOO streaming capture differs")
    validate_source_only_residual_shrinkage_audit(audit)


__all__ = [
    "HOOK_CONTRACT_SHA256",
    "HOOK_SCHEMA",
    "capture_source_only_transport_v2_loo",
    "hook_contract",
    "validate_streaming_hook_capture",
]
