"""Explicit frozen-metric adapter for the frozen-relative primitive schema.

This is a parallel schema, not a widening of the native-V3 metric bridge.
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
    factorized_native_contrast_v21_frozen_relative_primitive_readout
    as primitive_formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as relative_formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_lerf_exact as contrast_relevance,
)
from radio_gs.interfaces import lerf_v21_native_v3_frozen_metric_bridge as frozen
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


EXTERNAL_CACHE_SCHEMA = (
    "radio_gs.lerf_v21_frozen_relative_external_query_score_cache.v1"
)
METRIC_AUTHORITY_SCHEMA = (
    "radio_gs.lerf_v21_frozen_relative_frozen_metric_execution.v1"
)
SCHEMA_VERSION = 1
FROZEN_EVALUATOR = dict(frozen.FROZEN_EVALUATOR)
FROZEN_SUMMARY_HEAD = dict(frozen.FROZEN_SUMMARY_HEAD)
METRIC_PROTOCOL = dict(frozen.METRIC_PROTOCOL)
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
        "frozen_relative_readout_validated": True,
        "primitive_union_explicit_schema_validated": True,
        "query_ids_restored_from_validated_relevance_only": True,
        "opaque_query_axis_reordered": False,
        "renderer_geometry_opened": True,
        "factorized_primitive_state_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan": False,
        "scene_specific_parameters": False,
    }


def metric_build_access_audit() -> dict[str, bool]:
    return {
        "external_query_score_cache_validated": True,
        "full_relative_primitive_chain_rebuilt": True,
        "frozen_protocol_files_validated": True,
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
        "relevance_schema": contrast_relevance.QUERY_RELEVANCE_SCHEMA,
        "relative_schema": relative_formal.READOUT_SCHEMA,
        "primitive_readout_schema": primitive_formal.READOUT_SCHEMA,
        "evaluator_surface": {
            "query_scores": "binary_float32_N_by_Q_primitive_membership",
            "valid": "factorized_primitive_state_bool_N",
            "xyz": "exact_renderer_float32_N_by_3",
            "query_names": "health_gated_exact_relevance_query_axis",
        },
        "score_transform": "none",
        "query_axis_reorder": False,
        "legacy_native_v3_bridge_modified": False,
        "access_audit": cache_access_audit(),
    }


EXTERNAL_CACHE_CONTRACT_SHA256 = canonical_json_sha256(external_cache_contract())


def query_axis_sha256(query_ids: object) -> str:
    return canonical_json_sha256(query_ids)


def channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "query_scores": tensor_sha256(value["query_scores"]),
        "valid": tensor_sha256(value["valid"]),
        "xyz": tensor_sha256(value["xyz"]),
        "query_names": query_axis_sha256(value["metadata"]["query_names"]),
    }


def _query_axis(
    *,
    relevance: Mapping[str, Any],
    execution: Mapping[str, Any],
    relevance_record: Mapping[str, str],
    manifest_record: Mapping[str, str],
    all_query_record: Mapping[str, str],
    negative_record: Mapping[str, str],
) -> tuple[str, ...]:
    return frozen._validated_query_axis(
        relevance=relevance,
        verified_query_execution=execution,
        relevance_record=relevance_record,
        exact_query_manifest_record=manifest_record,
        all_query_cache_record=all_query_record,
        canonical_negative_cache_record=negative_record,
    )


def build_external_query_score_cache(
    *,
    validated_relevance: Mapping[str, Any],
    verified_query_execution: Mapping[str, Any],
    validated_relative: Mapping[str, Any],
    validated_primitive: Mapping[str, Any],
    relevance_record: Mapping[str, str],
    relative_record: Mapping[str, str],
    primitive_record: Mapping[str, str],
    renderer_record: Mapping[str, str],
    manifest_record: Mapping[str, str],
    all_query_record: Mapping[str, str],
    negative_record: Mapping[str, str],
    state_record: Mapping[str, str],
    state_xyz: torch.Tensor,
    state_valid: torch.Tensor,
    renderer_xyz: torch.Tensor,
) -> dict[str, Any]:
    records = {
        "relevance_authority": record(relevance_record, label="exact relevance"),
        "frozen_relative_readout_authority": record(
            relative_record, label="frozen-relative readout"
        ),
        "primitive_readout_authority": record(
            primitive_record, label="primitive readout"
        ),
        "renderer_geometry_checkpoint": record(renderer_record, label="renderer"),
        "factorized_primitive_state": record(state_record, label="primitive state"),
        "exact_query_manifest": record(manifest_record, label="query manifest"),
        "all_query_text_cache": record(all_query_record, label="all-query cache"),
        "canonical_negative_text_cache": record(
            negative_record, label="negative cache"
        ),
    }
    query_ids = _query_axis(
        relevance=validated_relevance,
        execution=verified_query_execution,
        relevance_record=records["relevance_authority"],
        manifest_record=records["exact_query_manifest"],
        all_query_record=records["all_query_text_cache"],
        negative_record=records["canonical_negative_text_cache"],
    )
    relevance_values = torch.as_tensor(
        validated_relevance["region_absolute_relevance"]
    )
    relative_raw = torch.as_tensor(validated_relative["raw_relevance"])
    relative_values = torch.as_tensor(validated_relative["relative_relevance"])
    primitive_relative = torch.as_tensor(validated_primitive["relative_relevance"])
    membership = torch.as_tensor(validated_primitive["primitive_membership"])
    primitive_valid = torch.as_tensor(validated_primitive["primitive_valid"])
    state_xyz = torch.as_tensor(state_xyz)
    state_valid = torch.as_tensor(state_valid)
    renderer_xyz = torch.as_tensor(renderer_xyz)
    primitive_inputs = validated_primitive.get("input_authority")
    relative_inputs = validated_relative.get("input_authority")
    if (
        not isinstance(primitive_inputs, Mapping)
        or not isinstance(relative_inputs, Mapping)
        or record(
            primitive_inputs.get("frozen_relative_readout"),
            label="primitive relative input",
        )
        != records["frozen_relative_readout_authority"]
        or record(
            relative_inputs.get("exact_relevance"), label="relative exact input"
        )
        != records["relevance_authority"]
        or record(
            primitive_inputs.get("factorized_primitive_state"),
            label="primitive state input",
        )
        != records["factorized_primitive_state"]
        or validated_relative["scene_id"] != validated_relevance["scene_id"]
        or validated_primitive["scene_id"] != validated_relevance["scene_id"]
        or validated_relative["physical_space_id"]
        != validated_relevance["physical_space_id"]
        or validated_primitive["physical_space_id"]
        != validated_relevance["physical_space_id"]
        or not torch.equal(relative_raw, relevance_values)
        or not torch.equal(relative_values, primitive_relative)
        or validated_relative["query_axis_count"] != len(query_ids)
        or validated_primitive["query_axis_count"] != len(query_ids)
        or membership.dtype != torch.float32
        or membership.shape != (state_valid.numel(), len(query_ids))
        or primitive_valid.dtype != torch.bool
        or not torch.equal(primitive_valid, state_valid)
        or state_xyz.dtype != torch.float32
        or renderer_xyz.dtype != torch.float32
        or state_xyz.shape != (state_valid.numel(), 3)
        or not torch.equal(state_xyz, renderer_xyz)
        or bool(membership[~state_valid].count_nonzero())
    ):
        raise ValueError("frozen-relative metric adapter axis binding differs")
    payload = {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": external_cache_contract(),
        "contract_sha256": EXTERNAL_CACHE_CONTRACT_SHA256,
        "query_scores": membership.detach().cpu().contiguous(),
        "valid": state_valid.detach().cpu().contiguous(),
        "xyz": renderer_xyz.detach().cpu().contiguous(),
        "metadata": {
            "scene_id": validated_relevance["scene_id"],
            "physical_space_id": validated_relevance["physical_space_id"],
            "query_names": list(query_ids),
            "query_axis_sha256": query_axis_sha256(list(query_ids)),
            "score_semantics": (
                "binary_frozen_relative_selected_scale_greedy_novelty_union_"
                "primitive_membership"
            ),
            "score_transform": "none",
            **records,
        },
        "channel_sha256": {},
        "access_audit": cache_access_audit(),
    }
    payload["channel_sha256"] = channel_sha256(payload)
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
        raise ValueError("frozen-relative external cache fields differ")
    payload = dict(value)
    metadata = payload.get("metadata")
    record_names = {
        "relevance_authority",
        "frozen_relative_readout_authority",
        "primitive_readout_authority",
        "renderer_geometry_checkpoint",
        "factorized_primitive_state",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    }
    metadata_names = {
        "scene_id",
        "physical_space_id",
        "query_names",
        "query_axis_sha256",
        "score_semantics",
        "score_transform",
        *record_names,
    }
    scores = torch.as_tensor(payload.get("query_scores"))
    valid = torch.as_tensor(payload.get("valid"))
    xyz = torch.as_tensor(payload.get("xyz"))
    query_ids = metadata.get("query_names") if isinstance(metadata, Mapping) else None
    if (
        payload.get("schema") != EXTERNAL_CACHE_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != external_cache_contract()
        or payload.get("contract_sha256") != EXTERNAL_CACHE_CONTRACT_SHA256
        or payload.get("access_audit") != cache_access_audit()
        or not isinstance(metadata, Mapping)
        or set(metadata) != metadata_names
        or not isinstance(metadata.get("scene_id"), str)
        or not metadata.get("scene_id")
        or not isinstance(metadata.get("physical_space_id"), str)
        or not metadata.get("physical_space_id")
        or not isinstance(query_ids, list)
        or not query_ids
        or len(set(query_ids)) != len(query_ids)
        or metadata.get("query_axis_sha256") != query_axis_sha256(query_ids)
        or metadata.get("score_semantics")
        != (
            "binary_frozen_relative_selected_scale_greedy_novelty_union_"
            "primitive_membership"
        )
        or metadata.get("score_transform") != "none"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or scores.shape[1] != len(query_ids)
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or bool(((scores != 0.0) & (scores != 1.0)).any())
        or bool(scores[~valid].count_nonzero())
        or payload.get("channel_sha256") != channel_sha256(payload)
    ):
        raise ValueError("frozen-relative external cache differs")
    payload["metadata"] = dict(metadata)
    for name in record_names:
        payload["metadata"][name] = record(metadata[name], label=f"cache {name}")
    return payload


def metric_authority_contract() -> dict[str, Any]:
    return {
        "schema": METRIC_AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "external_cache_schema": EXTERNAL_CACHE_SCHEMA,
        "evaluator": FROZEN_EVALUATOR,
        "summary_head": FROZEN_SUMMARY_HEAD,
        "protocol": METRIC_PROTOCOL,
        "candidate_count": 1,
        "threshold_scan": False,
        "scene_specific_parameters": False,
        "label_root_unopened_at_build": True,
        "legacy_native_v3_metric_authority_modified": False,
    }


METRIC_AUTHORITY_CONTRACT_SHA256 = canonical_json_sha256(
    metric_authority_contract()
)


def validate_metric_authority(value: object) -> dict[str, Any]:
    record_names = {
        "implementation",
        "launcher",
        "frozen_evaluator",
        "frozen_summary_head",
        "external_query_score_cache",
        "relevance_authority",
        "frozen_relative_readout_authority",
        "primitive_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    }
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "physical_space_id",
        *record_names,
        "label_root",
        "output_dir",
        "protocol",
        "single_candidate_no_sweep",
        "scene_specific_parameters",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen-relative metric authority fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != METRIC_AUTHORITY_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != metric_authority_contract()
        or payload.get("contract_sha256") != METRIC_AUTHORITY_CONTRACT_SHA256
        or payload.get("status")
        != "authorized_single_frozen_relative_lerf_metric"
        or payload.get("protocol") != METRIC_PROTOCOL
        or payload.get("single_candidate_no_sweep") is not True
        or payload.get("scene_specific_parameters") is not False
        or payload.get("metric_execution_authorized") is not True
        or payload.get("access_audit") != metric_build_access_audit()
    ):
        raise ValueError("frozen-relative metric authority differs")
    for name in record_names:
        payload[name] = record(payload[name], label=f"metric {name}")
    for name in ("label_root", "output_dir"):
        raw = payload.get(name)
        if (
            not isinstance(raw, str)
            or not raw.startswith("/")
            or posixpath.normpath(raw) != raw
        ):
            raise ValueError(f"metric {name} must be canonical absolute")
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
    "build_external_query_score_cache",
    "cache_access_audit",
    "channel_sha256",
    "external_cache_contract",
    "metric_authority_contract",
    "metric_build_access_audit",
    "query_axis_sha256",
    "record",
    "validate_external_query_score_cache",
    "validate_metric_authority",
]
