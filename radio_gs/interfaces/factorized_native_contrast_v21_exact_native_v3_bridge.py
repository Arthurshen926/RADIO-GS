"""Explicit contrast-V2.1 exact-relevance to query-opaque native-V3 bridge.

This bridge does not widen the frozen surface-region readout schema.  It first
validates the complete contrast exact relevance and execution lineage, then
forwards only scene/physical identity, canonical region identity, the region
fingerprint SHA, and the opaque ``[R,Q]`` value tensor to the existing native
V3 absolute readout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as contrast
from radio_gs.interfaces import (
    surface_region_v21_native_v3_absolute_readout as native_readout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


BRIDGE_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_exact_native_v3_bridge.v1"
)
BRIDGE_SCHEMA_VERSION = 1
CONTRAST_EXACT_DISPATCH_NAME = "contrast_v21_lerf_exact_relevance_v1"


@dataclass(frozen=True)
class ValidatedRelevanceDispatch:
    dispatch_name: str
    schema: str
    payload: dict[str, Any]


def bridge_access_audit() -> dict[str, bool]:
    return {
        "strict_relevance_schema_validator_called": True,
        "query_execution_authority_validated": True,
        "health_v4_pass_lineage_validated": True,
        "source_result_and_checkpoint_lineage_validated": True,
        "target_descriptor_lineage_validated": True,
        "query_identifiers_validated_only_at_relevance_authority_boundary": True,
        "query_identifiers_forwarded_to_readout": False,
        "query_strings_forwarded_to_readout": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "diagnostic_only": True,
        "final_candidate": False,
    }


def bridge_contract() -> dict[str, Any]:
    return {
        "schema": BRIDGE_SCHEMA,
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "input_schema_dispatch": {
            contrast.QUERY_RELEVANCE_SCHEMA: {
                "dispatch_name": CONTRAST_EXACT_DISPATCH_NAME,
                "validator": (
                    "factorized_native_contrast_v21_lerf_exact."
                    "validate_query_relevance"
                ),
            }
        },
        "field_guessing": False,
        "future_calibrator_policy": (
            "requires_new_explicit_schema_and_validator_dispatch_entry"
        ),
        "lineage": {
            "query_execution_authority": "strict_full_validator",
            "health_gate": "source_student_envelope_health_v4_PASS",
            "source": "contrast_v21_result_and_checkpoint",
            "target": "contrast_v21_exact_query_descriptor_view",
        },
        "forwarded_to_readout": [
            "scene_id",
            "physical_space_id",
            "canonical_region_indices",
            "region_fingerprints_sha256",
            "region_absolute_relevance_R_by_Q",
        ],
        "query_axis": "opaque_after_authority_validation",
        "readout_function": (
            "surface_region_v21_native_v3_absolute_readout."
            "apply_native_v3_absolute_readout"
        ),
        "output_schema": native_readout.READOUT_SCHEMA,
        "old_surface_region_adapter_modified": False,
        "diagnostic_only": True,
        "metric_access": False,
    }


BRIDGE_CONTRACT_SHA256 = canonical_json_sha256(bridge_contract())


def dispatch_relevance_schema(value: object) -> ValidatedRelevanceDispatch:
    """Dispatch solely by an explicitly supported schema, never by fields."""

    if not isinstance(value, Mapping):
        raise ValueError("exact relevance bridge input must be a mapping")
    schema = value.get("schema")
    if schema == contrast.QUERY_RELEVANCE_SCHEMA:
        payload = contrast.validate_query_relevance(value)
        return ValidatedRelevanceDispatch(
            dispatch_name=CONTRAST_EXACT_DISPATCH_NAME,
            schema=contrast.QUERY_RELEVANCE_SCHEMA,
            payload=payload,
        )
    raise ValueError(f"unsupported explicit relevance schema dispatch: {schema!r}")


def _record(value: object, *, label: str) -> dict[str, str]:
    return contrast.record(value, label=label)


def validate_contrast_exact_lineage(
    *,
    dispatched: ValidatedRelevanceDispatch,
    relevance_record: object,
    query_execution: object,
) -> dict[str, Any]:
    """Bind the strict relevance payload to a fully validated query execution."""

    if (
        not isinstance(dispatched, ValidatedRelevanceDispatch)
        or dispatched.dispatch_name != CONTRAST_EXACT_DISPATCH_NAME
        or dispatched.schema != contrast.QUERY_RELEVANCE_SCHEMA
    ):
        raise ValueError("contrast exact relevance dispatch differs")
    payload = contrast.validate_query_relevance(dispatched.payload)
    record = _record(relevance_record, label="contrast exact relevance")
    if not isinstance(query_execution, Mapping):
        raise ValueError("contrast exact query execution must be validated")
    execution = dict(query_execution)
    required_verified = {
        "verified_record",
        "verified_prequery_gate",
        "verified_manifest",
        "verified_positive",
    }
    if not required_verified.issubset(execution):
        raise ValueError("contrast exact query execution validation differs")
    execution_record = _record(
        execution["verified_record"], label="contrast exact query execution"
    )
    producer = _record(payload["producer"], label="contrast exact producer")
    authority = _record(
        payload["query_execution_authority"],
        label="contrast exact query authority",
    )
    if authority != execution_record:
        raise ValueError("contrast exact relevance/query authority differs")
    if producer != _record(execution["implementation"], label="query implementation"):
        raise ValueError("contrast exact relevance producer differs")
    if execution.get("query_relevance_output") != record["path"]:
        raise ValueError("contrast exact relevance output binding differs")

    gate = execution["verified_prequery_gate"]
    descriptor = gate.get("descriptor_view") if isinstance(gate, Mapping) else None
    full_descriptor = gate.get("descriptor") if isinstance(gate, Mapping) else None
    health = gate.get("health_v4_audit") if isinstance(gate, Mapping) else None
    source = gate.get("source_gate") if isinstance(gate, Mapping) else None
    if (
        not isinstance(descriptor, Mapping)
        or not isinstance(full_descriptor, Mapping)
        or not isinstance(health, Mapping)
    ):
        raise ValueError("contrast exact prequery lineage differs")
    source_checkpoint = (
        source.get("result", {}).get("checkpoint")
        if isinstance(source, Mapping)
        else None
    )
    expected_inputs = {
        "source_result": execution["source_result"],
        "target_descriptor": execution["target_descriptor"],
        "health_v4_audit": execution["health_v4_audit"],
        "health_v4_preregistration": execution["health_v4_preregistration"],
        "query_preregistration": execution["query_preregistration"],
        "exact_query_manifest": execution["exact_query_manifest"],
        "positive_text_cache": execution["positive_text_cache"],
        "all_query_text_cache": execution["all_query_text_cache"],
        "canonical_negative_bank": execution["canonical_negative_bank"],
    }
    canonical = payload["canonical_region_indices"]
    descriptor_canonical = descriptor.get("canonical_region_indices")
    fingerprint_sha = payload["channel_sha256"]["region_fingerprints"]
    query_ids = payload["query_ids"]
    positive = execution["verified_positive"]
    if (
        payload["input_authority"] != expected_inputs
        or health.get("status") != "pass"
        or health.get("query_authority_eligible") is not True
        or gate.get("health_v4_audit_record") != execution["health_v4_audit"]
        or gate.get("source_result_record") != execution["source_result"]
        or gate.get("target_descriptor_record") != execution["target_descriptor"]
        or source_checkpoint
        != full_descriptor.get("input_authority", {}).get(
            "source_contrast_v21_checkpoint"
        )
        or payload["scene_id"] != descriptor.get("scene_id")
        or payload["physical_space_id"] != descriptor.get("physical_space_id")
        or payload["region_row_ids"] != descriptor.get("region_row_ids")
        or payload["region_fingerprints"] != descriptor.get("region_fingerprints")
        or fingerprint_sha
        != canonical_json_sha256(descriptor.get("region_fingerprints"))
        or not torch.equal(canonical, descriptor_canonical)
        or tensor_sha256(canonical)
        != payload["channel_sha256"]["canonical_region_indices"]
        or tuple(query_ids) != tuple(positive.query_ids)
        or list(query_ids) != list(execution["verified_manifest"]["query_ids"])
    ):
        raise ValueError("contrast exact relevance source/target/health lineage differs")
    return payload


def query_opaque_view(
    validated_payload: Mapping[str, Any],
) -> native_readout.QueryOpaqueAbsoluteRelevance:
    """Reduce a validated payload to the five allowed downstream channels."""

    canonical = validated_payload["canonical_region_indices"]
    values = validated_payload["region_absolute_relevance"]
    return native_readout.QueryOpaqueAbsoluteRelevance(
        scene_id=str(validated_payload["scene_id"]),
        physical_space_id=str(validated_payload["physical_space_id"]),
        canonical_region_indices=canonical.detach().clone().contiguous(),
        region_fingerprints_sha256=str(
            validated_payload["channel_sha256"]["region_fingerprints"]
        ),
        values=values.detach().clone().contiguous(),
    )


__all__ = [
    "BRIDGE_CONTRACT_SHA256",
    "BRIDGE_SCHEMA",
    "BRIDGE_SCHEMA_VERSION",
    "CONTRAST_EXACT_DISPATCH_NAME",
    "ValidatedRelevanceDispatch",
    "bridge_access_audit",
    "bridge_contract",
    "dispatch_relevance_schema",
    "query_opaque_view",
    "validate_contrast_exact_lineage",
]
