"""Frozen bridge from a health-gated exact readout to the LERF evaluator.

This module is deliberately separate from the existing LERF pipeline.  It
restores the query axis only after an exact-relevance authority and its
health-gated execution authority have been validated, then binds that axis to
the query-opaque native-V3 readout without reordering either dimension.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import posixpath
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import (
    factorized_native_contrast_v21_lerf_exact as contrast_relevance,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


EXTERNAL_CACHE_SCHEMA = (
    "radio_gs.lerf_v21_native_v3_external_query_score_cache.v1"
)
METRIC_AUTHORITY_SCHEMA = (
    "radio_gs.lerf_v21_native_v3_frozen_metric_execution.v1"
)
SCHEMA_VERSION = 1
SUPPORTED_RELEVANCE_SCHEMAS = (
    contrast_relevance.QUERY_RELEVANCE_SCHEMA,
)
FROZEN_EVALUATOR = {
    "path": "/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py",
    "sha256": "8cb39acf08c4f90f6339002ef32022437f67e7cdaae80fc16206b49abbb917d5",
}
FROZEN_SUMMARY_HEAD = {
    "path": "/root/RADIO-GS/checkpoints/siglip2_summary_head.pth",
    "sha256": "41ccc47b2da9b1aed3ee1e80397dc721ec625e083054175c27698e8840b6263c",
}
METRIC_PROTOCOL = {
    "protocol_preset": "vala_paper_3d",
    "selection_mode": "score_threshold",
    "score_threshold": 0.6,
    "score_postprocess": "none",
    "projection_mode": "selected_only_alpha",
    "silhouette_threshold": 10.0 / 255.0,
    "alpha_binarization": "png_uint8_gt10",
    "mask_refinement": "none",
    "official_frames_only": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def cache_access_audit() -> dict[str, bool]:
    return {
        "health_gated_exact_relevance_validated_before_query_axis_restore": True,
        "query_ids_restored_from_validated_relevance_only": True,
        "opaque_readout_query_axis_reordered": False,
        "renderer_geometry_opened": True,
        "factorized_primitive_state_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "label_root_opened": False,
        "target_metrics_computed": False,
        "threshold_scan": False,
        "scene_specific_parameters": False,
    }


def metric_build_access_audit() -> dict[str, bool]:
    return {
        "external_query_score_cache_validated": True,
        "frozen_protocol_files_validated": True,
        "config_opened_for_sha256_only": True,
        "label_root_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "subprocess_started": False,
        "threshold_scan": False,
        "scene_specific_parameters": False,
    }


def external_cache_contract() -> dict[str, Any]:
    return {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "input": (
            "health_gated_exact_absolute_relevance_plus_query_opaque_"
            "native_v3_absolute_readout"
        ),
        "evaluator_surface": {
            "query_scores": "float32_N_by_Q_primitive_membership",
            "valid": "bool_N",
            "xyz": "float32_N_by_3_exact_renderer_row_axis",
            "query_names": "exact_relevance_query_ids_in_opaque_axis_order",
        },
        "score_transform": "none",
        "query_axis_reorder": False,
        "legacy_pipeline_modified": False,
        "access_audit": cache_access_audit(),
    }


EXTERNAL_CACHE_CONTRACT_SHA256 = canonical_json_sha256(
    external_cache_contract()
)


def query_axis_sha256(query_ids: object) -> str:
    return canonical_json_sha256(query_ids)


def external_cache_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "query_scores": tensor_sha256(value["query_scores"]),
        "valid": tensor_sha256(value["valid"]),
        "xyz": tensor_sha256(value["xyz"]),
        "query_names": query_axis_sha256(value["metadata"]["query_names"]),
    }


def _validated_query_axis(
    *,
    relevance: Mapping[str, Any],
    verified_query_execution: Mapping[str, Any],
    relevance_record: Mapping[str, str],
    exact_query_manifest_record: Mapping[str, str],
    all_query_cache_record: Mapping[str, str],
    canonical_negative_cache_record: Mapping[str, str],
) -> tuple[str, ...]:
    """Restore query IDs from an already validated health-gated authority."""

    if relevance.get("schema") not in SUPPORTED_RELEVANCE_SCHEMAS:
        raise ValueError("unsupported health-gated exact relevance schema")
    if relevance.get("schema") != contrast_relevance.QUERY_RELEVANCE_SCHEMA:
        raise ValueError("health-gated relevance dispatch differs")
    execution_record = record(
        verified_query_execution.get("verified_record"),
        label="verified query execution",
    )
    manifest_record = record(
        exact_query_manifest_record, label="exact query manifest"
    )
    all_record = record(all_query_cache_record, label="all-query cache")
    negative_record = record(
        canonical_negative_cache_record, label="canonical-negative cache"
    )
    input_authority = relevance.get("input_authority")
    manifest = verified_query_execution.get("verified_manifest")
    if not isinstance(input_authority, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("validated exact relevance lineage differs")
    expected_inputs = {
        "source_result": verified_query_execution.get("source_result"),
        "target_descriptor": verified_query_execution.get("target_descriptor"),
        "health_v4_audit": verified_query_execution.get("health_v4_audit"),
        "health_v4_preregistration": verified_query_execution.get(
            "health_v4_preregistration"
        ),
        "query_preregistration": verified_query_execution.get(
            "query_preregistration"
        ),
        "exact_query_manifest": manifest_record,
        "positive_text_cache": verified_query_execution.get(
            "positive_text_cache"
        ),
        "all_query_text_cache": all_record,
        "canonical_negative_bank": negative_record,
    }
    # Query identifiers are first consumed below, after both validators have
    # returned and every immutable lineage record has been matched.
    query_ids = relevance.get("query_ids")
    manifest_query_ids = manifest.get("query_ids")
    if (
        relevance.get("query_execution_authority") != execution_record
        or dict(input_authority) != expected_inputs
        or verified_query_execution.get("exact_query_manifest") != manifest_record
        or verified_query_execution.get("all_query_text_cache") != all_record
        or verified_query_execution.get("canonical_negative_bank") != negative_record
        or not isinstance(query_ids, list)
        or not isinstance(manifest_query_ids, list)
        or query_ids != manifest_query_ids
        or not query_ids
        or len(set(query_ids)) != len(query_ids)
        or any(not isinstance(item, str) or not item.strip() for item in query_ids)
        or query_axis_sha256(query_ids) != manifest.get("query_ids_sha256")
    ):
        raise ValueError("validated exact relevance query axis differs")
    return tuple(query_ids)


def build_external_query_score_cache(
    *,
    validated_relevance: Mapping[str, Any],
    verified_query_execution: Mapping[str, Any],
    validated_readout: Mapping[str, Any],
    relevance_record: Mapping[str, str],
    readout_record: Mapping[str, str],
    renderer_geometry_record: Mapping[str, str],
    exact_query_manifest_record: Mapping[str, str],
    all_query_cache_record: Mapping[str, str],
    canonical_negative_cache_record: Mapping[str, str],
    factorized_state_record: Mapping[str, str],
    state_xyz: torch.Tensor,
    state_valid: torch.Tensor,
    renderer_xyz: torch.Tensor,
) -> dict[str, Any]:
    """Build the exact evaluator cache without opening labels or metrics."""

    relevance_record = record(relevance_record, label="exact relevance")
    readout_record = record(readout_record, label="native-V3 readout")
    renderer_record = record(
        renderer_geometry_record, label="renderer geometry checkpoint"
    )
    state_record = record(factorized_state_record, label="factorized state")
    manifest_record = record(
        exact_query_manifest_record, label="exact query manifest"
    )
    all_record = record(all_query_cache_record, label="all-query cache")
    negative_record = record(
        canonical_negative_cache_record, label="canonical-negative cache"
    )
    query_ids = _validated_query_axis(
        relevance=validated_relevance,
        verified_query_execution=verified_query_execution,
        relevance_record=relevance_record,
        exact_query_manifest_record=manifest_record,
        all_query_cache_record=all_record,
        canonical_negative_cache_record=negative_record,
    )
    readout_inputs = validated_readout.get("input_authority")
    expected_readout_inputs = {
        "absolute_relevance",
        "native_v3_feature",
        "native_v3_inference",
        "factorized_primitive_state",
    }
    if not isinstance(readout_inputs, Mapping) or set(readout_inputs) != expected_readout_inputs:
        raise ValueError("native-V3 readout input authority differs")
    normalized_readout_inputs = {
        name: record(readout_inputs[name], label=f"readout input {name}")
        for name in sorted(expected_readout_inputs)
    }
    relevance_values = torch.as_tensor(
        validated_relevance.get("region_absolute_relevance")
    )
    relevance_canonical = torch.as_tensor(
        validated_relevance.get("canonical_region_indices")
    )
    readout_unary = torch.as_tensor(validated_readout.get("absolute_relevance"))
    readout_canonical = torch.as_tensor(
        validated_readout.get("canonical_region_indices")
    )
    membership = torch.as_tensor(validated_readout.get("primitive_membership"))
    readout_valid = torch.as_tensor(validated_readout.get("primitive_valid"))
    state_xyz = torch.as_tensor(state_xyz)
    state_valid = torch.as_tensor(state_valid)
    renderer_xyz = torch.as_tensor(renderer_xyz)
    descriptor = verified_query_execution.get("verified_prequery_gate", {}).get(
        "descriptor"
    )
    descriptor_state = (
        descriptor.get("input_authority", {}).get("factorized_primitive_state")
        if isinstance(descriptor, Mapping)
        else None
    )
    if (
        normalized_readout_inputs["absolute_relevance"] != relevance_record
        or normalized_readout_inputs["factorized_primitive_state"] != state_record
        or descriptor_state != state_record
        or validated_readout.get("scene_id") != validated_relevance.get("scene_id")
        or validated_readout.get("physical_space_id")
        != validated_relevance.get("physical_space_id")
        or validated_readout.get("region_fingerprints_sha256")
        != canonical_json_sha256(validated_relevance.get("region_fingerprints"))
        or not torch.equal(readout_canonical, relevance_canonical)
        or not torch.equal(readout_unary, relevance_values)
        or validated_readout.get("query_axis_count") != len(query_ids)
        or membership.dtype != torch.float32
        or membership.ndim != 2
        or membership.shape != (state_valid.numel(), len(query_ids))
        or state_valid.dtype != torch.bool
        or state_valid.ndim != 1
        or readout_valid.dtype != torch.bool
        or not torch.equal(readout_valid, state_valid)
        or state_xyz.dtype != torch.float32
        or renderer_xyz.dtype != torch.float32
        or state_xyz.shape != (state_valid.numel(), 3)
        or renderer_xyz.shape != state_xyz.shape
        or not torch.equal(renderer_xyz, state_xyz)
        or bool(membership[~state_valid].count_nonzero())
    ):
        raise ValueError("exact relevance/readout/renderer axis binding differs")
    payload = {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": external_cache_contract(),
        "contract_sha256": EXTERNAL_CACHE_CONTRACT_SHA256,
        "query_scores": membership.detach().cpu().contiguous(),
        "valid": state_valid.detach().cpu().contiguous(),
        "xyz": renderer_xyz.detach().cpu().contiguous(),
        "metadata": {
            "scene_id": str(validated_relevance["scene_id"]),
            "physical_space_id": str(validated_relevance["physical_space_id"]),
            "query_names": list(query_ids),
            "query_axis_sha256": query_axis_sha256(list(query_ids)),
            "score_semantics": (
                "binary_native_v3_absolute_relevance_greedy_novelty_union_"
                "primitive_membership"
            ),
            "score_transform": "none",
            "relevance_schema": validated_relevance["schema"],
            "relevance_authority": relevance_record,
            "readout_authority": readout_record,
            "renderer_geometry_checkpoint": renderer_record,
            "factorized_primitive_state": state_record,
            "exact_query_manifest": manifest_record,
            "all_query_text_cache": all_record,
            "canonical_negative_text_cache": negative_record,
        },
        "channel_sha256": {},
        "access_audit": cache_access_audit(),
    }
    payload["channel_sha256"] = external_cache_channel_sha256(payload)
    return validate_external_query_score_cache(payload)


def validate_external_query_score_cache(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "query_scores",
        "valid",
        "xyz",
        "metadata",
        "channel_sha256",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen external query-score cache fields differ")
    payload = dict(value)
    metadata = payload.get("metadata")
    metadata_keys = {
        "scene_id",
        "physical_space_id",
        "query_names",
        "query_axis_sha256",
        "score_semantics",
        "score_transform",
        "relevance_schema",
        "relevance_authority",
        "readout_authority",
        "renderer_geometry_checkpoint",
        "factorized_primitive_state",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != metadata_keys:
        raise ValueError("frozen external query-score cache metadata differs")
    query_ids = metadata.get("query_names")
    scores = torch.as_tensor(payload.get("query_scores"))
    valid = torch.as_tensor(payload.get("valid"))
    xyz = torch.as_tensor(payload.get("xyz"))
    if (
        payload.get("schema") != EXTERNAL_CACHE_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != external_cache_contract()
        or payload.get("contract_sha256") != EXTERNAL_CACHE_CONTRACT_SHA256
        or payload.get("access_audit") != cache_access_audit()
        or not isinstance(metadata.get("scene_id"), str)
        or not metadata.get("scene_id")
        or not isinstance(metadata.get("physical_space_id"), str)
        or not metadata.get("physical_space_id")
        or not isinstance(query_ids, list)
        or not query_ids
        or len(set(query_ids)) != len(query_ids)
        or any(not isinstance(item, str) or not item.strip() for item in query_ids)
        or metadata.get("query_axis_sha256") != query_axis_sha256(query_ids)
        or metadata.get("score_semantics")
        != "binary_native_v3_absolute_relevance_greedy_novelty_union_primitive_membership"
        or metadata.get("score_transform") != "none"
        or metadata.get("relevance_schema") not in SUPPORTED_RELEVANCE_SCHEMAS
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or scores.shape[1] != len(query_ids)
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(xyz).all())
        or bool(((scores != 0.0) & (scores != 1.0)).any())
        or bool(scores[~valid].count_nonzero())
        or payload.get("channel_sha256") != external_cache_channel_sha256(payload)
    ):
        raise ValueError("frozen external query-score cache tensor differs")
    payload["metadata"] = dict(metadata)
    for name in (
        "relevance_authority",
        "readout_authority",
        "renderer_geometry_checkpoint",
        "factorized_primitive_state",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        payload["metadata"][name] = record(
            metadata[name], label=f"external cache metadata {name}"
        )
    return payload


def metric_authority_contract() -> dict[str, Any]:
    return {
        "schema": METRIC_AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluator": FROZEN_EVALUATOR,
        "summary_head": FROZEN_SUMMARY_HEAD,
        "protocol": METRIC_PROTOCOL,
        "candidate_count": 1,
        "threshold_scan": False,
        "scene_specific_parameters": False,
        "scene_identity_derived_from_cache": True,
        "label_root_unopened_at_build": True,
    }


METRIC_AUTHORITY_CONTRACT_SHA256 = canonical_json_sha256(
    metric_authority_contract()
)


def validate_metric_authority_payload(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "launcher",
        "frozen_evaluator",
        "frozen_summary_head",
        "external_query_score_cache",
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
        "label_root",
        "output_dir",
        "protocol",
        "single_candidate_no_sweep",
        "scene_specific_parameters",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen native-V3 metric authority fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != METRIC_AUTHORITY_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != metric_authority_contract()
        or payload.get("contract_sha256") != METRIC_AUTHORITY_CONTRACT_SHA256
        or payload.get("status")
        != "authorized_single_frozen_native_v3_lerf_metric"
        or not isinstance(payload.get("scene_id"), str)
        or not payload.get("scene_id")
        or not isinstance(payload.get("physical_space_id"), str)
        or not payload.get("physical_space_id")
        or payload.get("frozen_evaluator") != FROZEN_EVALUATOR
        or payload.get("frozen_summary_head") != FROZEN_SUMMARY_HEAD
        or payload.get("protocol") != METRIC_PROTOCOL
        or payload.get("single_candidate_no_sweep") is not True
        or payload.get("scene_specific_parameters") is not False
        or payload.get("metric_execution_authorized") is not True
        or payload.get("access_audit") != metric_build_access_audit()
    ):
        raise ValueError("frozen native-V3 metric authority header differs")
    for name in (
        "implementation",
        "launcher",
        "frozen_evaluator",
        "frozen_summary_head",
        "external_query_score_cache",
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    ):
        payload[name] = record(payload[name], label=f"metric authority {name}")
    for name in ("label_root", "output_dir"):
        raw = payload.get(name)
        if (
            not isinstance(raw, str)
            or not raw.startswith("/")
            or posixpath.normpath(raw) != raw
        ):
            raise ValueError(f"metric authority {name} must be canonical absolute")
    return payload


__all__ = [
    "EXTERNAL_CACHE_CONTRACT_SHA256",
    "EXTERNAL_CACHE_SCHEMA",
    "FROZEN_EVALUATOR",
    "FROZEN_SUMMARY_HEAD",
    "METRIC_AUTHORITY_CONTRACT_SHA256",
    "METRIC_AUTHORITY_SCHEMA",
    "METRIC_PROTOCOL",
    "SCHEMA_VERSION",
    "SUPPORTED_RELEVANCE_SCHEMAS",
    "build_external_query_score_cache",
    "cache_access_audit",
    "external_cache_channel_sha256",
    "external_cache_contract",
    "metric_authority_contract",
    "metric_build_access_audit",
    "query_axis_sha256",
    "record",
    "validate_external_query_score_cache",
    "validate_metric_authority_payload",
]
