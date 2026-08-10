"""Independent exact-query contract for the promoted contrast V2.1 descriptor.

This module intentionally does not import or mutate the current
``factorized_native_lerf_exact`` pipeline.  Query/text artifacts are reachable
only after source, descriptor, source-student health-v4 PASS, and preregistration lineage have
all been validated.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from pathlib import Path
import re
from types import ModuleType
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as target
from radio_gs.interfaces import factorized_native_lerf_exact as frozen_protocol
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.querying import v21_absolute_relevance_adapter as relevance_adapter
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


QUERY_EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_lerf_exact_query_execution.v1"
)
QUERY_RELEVANCE_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_lerf_exact_relevance.v1"
)
SCHEMA_VERSION = 1
HEALTH_V4_MODULE = (
    "radio_gs.interfaces.factorized_native_contrast_v21_target_health_v4"
)
ROOT = Path(__file__).resolve().parents[2]
QUERY_PREREGISTRATION = {
    "path": str(
        ROOT
        / "paper/artifacts/factorized_native_contrast_v21_lerf_exact_relevance_preregistration_20260807.json"
    ),
    "sha256": "62b9e314bd677f37ab9daa29a90d8a6a18deb5cc5f7c7a1c4b96966e19a87d88",
}
HEALTH_V4_PREREGISTRATION = {
    "path": str(
        ROOT
        / "paper/artifacts/factorized_native_contrast_v21_source_student_envelope_health_v4_preregistration_20260807.json"
    ),
    "sha256": "a1b04ce679e45388c2d2addde2f9a1f4cc62a4a7b1def1b1d3c5d3e7f0d8fc0f",
}
FROZEN_ALL_QUERY_CACHE = dict(frozen_protocol.FROZEN_ALL_QUERY_CACHE)
FROZEN_CANONICAL_NEGATIVE_BANK = dict(
    frozen_protocol.FROZEN_CANONICAL_NEGATIVE_BANK
)
FROZEN_EVALUATOR = dict(frozen_protocol.FROZEN_EVALUATOR)
EXACT_QUERY_MANIFEST_SCHEMA = frozen_protocol.EXACT_QUERY_MANIFEST_SCHEMA
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HealthV4UnavailableError(RuntimeError):
    """Raised before query access when the frozen health-v4 formal is absent."""


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def canonical_output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be canonical absolute")
    return resolved


def prequery_access_audit() -> dict[str, bool]:
    return {
        "contrast_v21_source_promotion_validated": True,
        "contrast_v21_target_descriptor_validated": True,
        "health_v4_pass_validated": True,
        "query_and_health_preregistration_validated": True,
        "exact_query_manifest_opened": False,
        "positive_text_cache_opened": False,
        "all_query_text_cache_opened": False,
        "canonical_negative_bank_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def query_access_audit() -> dict[str, bool]:
    return {
        **prequery_access_audit(),
        "exact_query_manifest_opened": True,
        "positive_text_cache_opened": True,
        "all_query_text_cache_opened": True,
        "canonical_negative_bank_opened": True,
    }


def query_contract() -> dict[str, Any]:
    return {
        "schema": QUERY_RELEVANCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "descriptor_schema": target.TARGET_DESCRIPTOR_SCHEMA,
        "descriptor_view_schema": target.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA,
        "health_gate": "source_student_envelope_health_v4_formal_PASS",
        "positive_text": "frozen_exact_scene_subset_of_official_all_query_bank",
        "canonical_negative": "frozen_four_row_official_siglip2_bank",
        "formula": "binary_softmax_positive_vs_max_canonical_negative",
        "scorer": "existing_calibrated_v21_absolute_relevance",
        "logit_scale": relevance_loss.INFERENCE_LOGIT_SCALE,
        "absolute_equal_logit_boundary": 0.5,
        "assume_normalized": True,
        "output": "float32_region_absolute_relevance_R_by_Q",
        "postprocess": "none",
        "threshold_scan": False,
        "scene_specific_parameters": False,
        "query_smoothing": False,
        "scene_minmax_remap": False,
        "query_ranking_normalization": False,
        "metric_access": False,
    }


QUERY_CONTRACT_SHA256 = canonical_json_sha256(query_contract())


def resolve_health_v4_dispatch() -> dict[str, Any]:
    """Resolve the health formal only when it exists and exports the frozen API."""

    try:
        module = importlib.import_module(HEALTH_V4_MODULE)
    except ModuleNotFoundError as error:
        if error.name != HEALTH_V4_MODULE:
            raise
        raise HealthV4UnavailableError(
            "contrast V2.1 health-v4 formal is not available; query files stay closed"
        ) from error
    required = (
        "HEALTH_AUDIT_SCHEMA",
        "HEALTH_AUDIT_IMPLEMENTATION_PATH",
        "validate_health_audit",
        "validate_query_gate_binding",
    )
    if any(not hasattr(module, name) for name in required):
        raise HealthV4UnavailableError(
            "contrast V2.1 health-v4 formal lacks the query-gate dispatch API"
        )
    schema = getattr(module, "HEALTH_AUDIT_SCHEMA")
    implementation = Path(
        getattr(module, "HEALTH_AUDIT_IMPLEMENTATION_PATH")
    ).resolve()
    if not isinstance(schema, str) or not schema or not implementation.is_absolute():
        raise HealthV4UnavailableError("contrast V2.1 health-v4 dispatch differs")
    if not implementation.is_file() or implementation.is_symlink():
        raise HealthV4UnavailableError(
            "contrast V2.1 health-v4 implementation is unavailable; query files stay closed"
        )
    return {
        "module": module,
        "schema": schema,
        "formal_record": file_record(Path(module.__file__).resolve()),
        "implementation_record": file_record(implementation),
    }


def _validate_query_preregistration(
    *, source_result_record: Mapping[str, str]
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        QUERY_PREREGISTRATION["path"],
        expected_sha256=QUERY_PREREGISTRATION["sha256"],
        label="contrast V2.1 exact relevance preregistration",
    )
    if (
        {"path": str(source), "sha256": digest} != QUERY_PREREGISTRATION
        or raw.get("status")
        != "frozen_before_opening_any_target_query_manifest_or_text_cache"
        or raw.get("candidate", {}).get("source_result")
        != dict(source_result_record)
        or raw.get("candidate", {}).get("health_v4_preregistration")
        != HEALTH_V4_PREREGISTRATION
        or raw.get("candidate", {}).get("target_descriptor_schema")
        != target.TARGET_DESCRIPTOR_SCHEMA
        or raw.get("candidate", {}).get("exact_query_view_schema")
        != target.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA
        or raw.get("query_protocol", {}).get("all_query_text_cache")
        != FROZEN_ALL_QUERY_CACHE
        or raw.get("query_protocol", {}).get("canonical_negative_bank")
        != FROZEN_CANONICAL_NEGATIVE_BANK
        or raw.get("query_protocol", {}).get("threshold_scan") is not False
        or raw.get("query_protocol", {}).get("scene_specific_parameters") is not False
    ):
        raise ValueError("contrast V2.1 exact relevance preregistration differs")
    return dict(raw)


def _load_target_descriptor(
    descriptor_record: Mapping[str, str],
    *,
    source_result_record: Mapping[str, str],
    source_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw, digest, source = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="contrast V2.1 target descriptor",
    )
    if descriptor_record != {"path": str(source), "sha256": digest}:
        raise ValueError("contrast V2.1 descriptor record differs")
    descriptor = target.validate_target_descriptor_authority(raw)
    execution = target.validate_target_execution_authority(
        descriptor["target_execution_authority"]["path"],
        expected_sha256=descriptor["target_execution_authority"]["sha256"],
        expected_output=descriptor_record["path"],
    )
    view = target.exact_query_descriptor_view(descriptor)
    if (
        execution["source_contrast_v21_result"] != dict(source_result_record)
        or descriptor["target_execution_authority"] != execution["verified_record"]
        or descriptor["producer"] != execution["implementation"]
        or descriptor["input_authority"]["source_contrast_v21_result"]
        != dict(source_result_record)
        or descriptor["input_authority"]["source_contrast_v21_checkpoint"]
        != source_gate["result"]["checkpoint"]
        or descriptor["source_selected_step"] != source_gate["selected_step"]
        or view["source_descriptor_schema"] != target.TARGET_DESCRIPTOR_SCHEMA
        or view["source_descriptor_contract_sha256"]
        != target.TARGET_DESCRIPTOR_CONTRACT_SHA256
    ):
        raise ValueError("contrast V2.1 descriptor/source lineage differs")
    return descriptor, execution, view


def validate_prequery_gate(
    *,
    source_result_record: object,
    target_descriptor_record: object,
    health_v4_audit_record: object,
) -> dict[str, Any]:
    """Validate every non-query gate; no query/text file is opened here."""

    source_record = record(
        source_result_record, label="contrast V2.1 source result"
    )
    descriptor_record = record(
        target_descriptor_record, label="contrast V2.1 target descriptor"
    )
    health_record = record(
        health_v4_audit_record, label="contrast V2.1 health-v4 audit"
    )
    source_gate = target.validate_source_contrast_v21_result(source_record)
    descriptor, execution, descriptor_view = _load_target_descriptor(
        descriptor_record,
        source_result_record=source_record,
        source_gate=source_gate,
    )
    query_preregistration = _validate_query_preregistration(
        source_result_record=source_record
    )
    validate_file_record(
        HEALTH_V4_PREREGISTRATION,
        label="contrast V2.1 health-v4 preregistration",
    )
    dispatch = resolve_health_v4_dispatch()
    raw, digest, source = load_json_object(
        health_record["path"],
        expected_sha256=health_record["sha256"],
        label="contrast V2.1 health-v4 PASS audit",
    )
    if (
        health_record != {"path": str(source), "sha256": digest}
        or raw.get("schema") != dispatch["schema"]
    ):
        raise ValueError("contrast V2.1 health-v4 schema dispatch differs")
    module: ModuleType = dispatch["module"]
    health = module.validate_health_audit(raw, require_pass=True)
    binding = module.validate_query_gate_binding(
        health,
        descriptor_record=descriptor_record,
        descriptor_value=descriptor,
        source_result_record=source_record,
        source_checkpoint_record=source_gate["result"]["checkpoint"],
        health_preregistration_record=HEALTH_V4_PREREGISTRATION,
    )
    # The health formal validates PASS internally and may return either its
    # normalized audit mapping or the boolean singleton True.
    if binding is not True and not isinstance(binding, Mapping):
        raise ValueError("contrast V2.1 health-v4 query-gate lineage differs")
    return {
        "source_result_record": source_record,
        "source_gate": source_gate,
        "target_descriptor_record": descriptor_record,
        "descriptor": descriptor,
        "descriptor_execution": execution,
        "descriptor_view": descriptor_view,
        "health_v4_audit_record": health_record,
        "health_v4_audit": health,
        "health_v4_dispatch": {
            "schema": dispatch["schema"],
            "formal_record": dispatch["formal_record"],
            "implementation_record": dispatch["implementation_record"],
        },
        "query_preregistration": query_preregistration,
        "query_preregistration_record": dict(QUERY_PREREGISTRATION),
        "health_v4_preregistration_record": dict(HEALTH_V4_PREREGISTRATION),
        "access_audit": prequery_access_audit(),
    }


def validate_exact_query_manifest(value: object, *, scene_id: str) -> dict[str, Any]:
    return frozen_protocol.validate_exact_query_manifest(value, scene_id=scene_id)


def query_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "query_ids": canonical_json_sha256(value["query_ids"]),
        "region_absolute_relevance": tensor_sha256(
            value["region_absolute_relevance"]
        ),
    }


def validate_query_relevance(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "query_execution_authority",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "region_fingerprints",
        "query_ids",
        "region_absolute_relevance",
        "channel_sha256",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("contrast V2.1 exact relevance fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != QUERY_RELEVANCE_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != query_contract()
        or payload.get("contract_sha256") != QUERY_CONTRACT_SHA256
        or payload.get("access_audit") != query_access_audit()
    ):
        raise ValueError("contrast V2.1 exact relevance header differs")
    payload["producer"] = record(payload["producer"], label="query producer")
    payload["query_execution_authority"] = record(
        payload["query_execution_authority"], label="query execution authority"
    )
    expected_inputs = {
        "source_result",
        "target_descriptor",
        "health_v4_audit",
        "health_v4_preregistration",
        "query_preregistration",
        "exact_query_manifest",
        "positive_text_cache",
        "all_query_text_cache",
        "canonical_negative_bank",
    }
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("contrast V2.1 exact relevance inputs differ")
    payload["input_authority"] = {
        name: record(inputs[name], label=f"query input {name}")
        for name in sorted(expected_inputs)
    }
    rows = payload.get("region_row_ids")
    canonical = payload.get("canonical_region_indices")
    fingerprints = payload.get("region_fingerprints")
    queries = payload.get("query_ids")
    relevance = payload.get("region_absolute_relevance")
    regions = len(rows) if isinstance(rows, list) else -1
    query_count = len(queries) if isinstance(queries, list) else -1
    if (
        regions <= 0
        or query_count <= 0
        or len(set(rows)) != regions
        or any(not isinstance(item, str) or not item for item in rows)
        or len(set(queries)) != query_count
        or any(not isinstance(item, str) or not item for item in queries)
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.int64
        or canonical.device.type != "cpu"
        or canonical.shape != (regions,)
        or not torch.is_tensor(relevance)
        or relevance.dtype != torch.float32
        or relevance.device.type != "cpu"
        or relevance.shape != (regions, query_count)
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any())
        or bool((relevance > 1).any())
        or payload.get("channel_sha256") != query_channel_sha256(payload)
    ):
        raise ValueError("contrast V2.1 exact relevance tensor differs")
    return payload


__all__ = [
    "EXACT_QUERY_MANIFEST_SCHEMA",
    "FROZEN_ALL_QUERY_CACHE",
    "FROZEN_CANONICAL_NEGATIVE_BANK",
    "HEALTH_V4_MODULE",
    "HEALTH_V4_PREREGISTRATION",
    "HealthV4UnavailableError",
    "QUERY_CONTRACT_SHA256",
    "QUERY_EXECUTION_SCHEMA",
    "QUERY_PREREGISTRATION",
    "QUERY_RELEVANCE_SCHEMA",
    "SCHEMA_VERSION",
    "canonical_output",
    "prequery_access_audit",
    "query_access_audit",
    "query_channel_sha256",
    "query_contract",
    "record",
    "resolve_health_v4_dispatch",
    "validate_exact_query_manifest",
    "validate_prequery_gate",
    "validate_query_relevance",
]
