"""Fail-closed shared deployment contracts for rank-256 V2.1B/V2.1C.

This module is intentionally independent from the frozen V2.1 rank-64
deployment chain.  It dispatches to the selected source-only promotion gate
before opening any target or benchmark-query artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import surface_region_v21b_source_gate as v21b_gate
from radio_gs.interfaces import surface_region_v21c_source_gate as v21c_gate
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.querying.v21_absolute_relevance_adapter import (
    OFFICIAL_TEXT_CANONICALIZATION,
)
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as v21b_trainer,
)
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as v21c_trainer,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


SOURCE_VARIANTS = ("v21b", "v21c")
TARGET_EXECUTION_SCHEMA = (
    "radio_gs.surface_region_rank256_champion_target_execution_authority.v1"
)
TARGET_DESCRIPTOR_SCHEMA = (
    "radio_gs.surface_region_rank256_champion_target_descriptor.v1"
)
EXACT_QUERY_MANIFEST_SCHEMA = "radio_gs.lerf_exact_scene_query_manifest.v1"
EXACT_QUERY_RECEIPT_SCHEMA = (
    "radio_gs.surface_region_rank256_champion_exact_query_subset_receipt.v1"
)
QUERY_EXECUTION_SCHEMA = (
    "radio_gs.surface_region_rank256_champion_query_execution_authority.v1"
)
QUERY_RELEVANCE_SCHEMA = (
    "radio_gs.surface_region_rank256_champion_query_relevance.v1"
)
EXTERNAL_EXECUTION_SCHEMA = (
    "radio_gs.lerf_rank256_champion_external_cache_execution_authority.v1"
)
EXTERNAL_CACHE_SCHEMA = "radio_gs.lerf_rank256_champion_external_scores.v1"
METRIC_EXECUTION_SCHEMA = (
    "radio_gs.lerf_rank256_champion_one_shot_metric_execution_authority.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be absolute and canonical")
    return resolved


def source_gate_validator(variant: str):
    if variant == "v21b":
        return v21b_gate.validate_source_pilot_chain
    if variant == "v21c":
        return v21c_gate.validate_source_pilot_chain
    raise ValueError(f"unsupported rank-256 source variant: {variant}")


def validate_champion_source(
    variant: str,
    source_result: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate a real source PASS before any target/query access."""

    gate = source_gate_validator(str(variant))(
        source_result,
        expected_sha256=expected_sha256,
        require_promotion=True,
    )
    if (
        gate.get("source_promotion_authorized") is not True
        or gate.get("benchmark_opened") is not False
        or gate.get("checkpoint") is None
        or gate.get("normalization_authority") is None
    ):
        raise ValueError("rank-256 champion source promotion is not authorized")
    gate = dict(gate)
    gate["source_result"] = _record(
        gate["source_result"], label="rank-256 source result"
    )
    gate["checkpoint"] = _record(
        gate["checkpoint"], label="rank-256 checkpoint"
    )
    gate["normalization_authority"] = _record(
        gate["normalization_authority"], label="rank-256 normalization"
    )
    return gate


def source_canonical_negative(gate: Mapping[str, Any], variant: str) -> dict[str, str]:
    execution = _record(gate["execution_authority"], label="source execution")
    raw, _, _ = load_json_object(
        execution["path"],
        expected_sha256=execution["sha256"],
        label="rank-256 source execution authority",
    )
    if variant == "v21c":
        parent = _record(
            raw.get("parent_v21b_execution_authority"),
            label="V2.1C parent V2.1B authority",
        )
        raw, _, _ = load_json_object(
            parent["path"],
            expected_sha256=parent["sha256"],
            label="V2.1C parent V2.1B execution authority",
        )
    return _record(
        raw.get("canonical_negative_bank"),
        label="rank-256 canonical-negative bank",
    )


def load_champion_model(
    gate: Mapping[str, Any], variant: str
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    normalization_raw, _, _ = load_torch_mapping(
        gate["normalization_authority"]["path"],
        expected_sha256=gate["normalization_authority"]["sha256"],
        map_location="cpu",
        label="rank-256 normalization",
    )
    if variant == "v21b":
        normalization = v21b_gate.validate_normalization_authority(normalization_raw)
    elif variant == "v21c":
        normalization = v21c_gate.validate_normalization(normalization_raw)
    else:
        raise ValueError("rank-256 source variant differs")
    checkpoint_raw, _, _ = load_torch_mapping(
        gate["checkpoint"]["path"],
        expected_sha256=gate["checkpoint"]["sha256"],
        map_location="cpu",
        label="rank-256 checkpoint",
    )
    if variant == "v21b":
        checkpoint = v21b_gate.validate_checkpoint_payload(
            checkpoint_raw, normalization=normalization
        )
    else:
        checkpoint = dict(checkpoint_raw)
        if (
            checkpoint.get("schema") != v21c_trainer.STAGE_II_CHECKPOINT_SCHEMA
            or checkpoint.get("normalization_authority")
            != gate["normalization_authority"]
        ):
            raise ValueError("V2.1C promoted checkpoint differs")
        state, digest = v21b_gate._validate_model_state(
            checkpoint.get("model_state_dict"), normalization=normalization
        )
        if (
            checkpoint.get("model_state_dict_sha256") != digest
            or gate.get("model_state_dict_sha256") != digest
        ):
            raise ValueError("V2.1C promoted model-state digest differs")
        checkpoint["model_state_dict"] = state
    model = v21b_gate.v21b_interface.build_model_from_source_normalization(
        normalization
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.cpu().eval().requires_grad_(False), normalization, checkpoint


def target_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_files": True,
        "benchmark_queries_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def query_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_query_files": True,
        "benchmark_queries_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def target_contract() -> dict[str, Any]:
    return {
        "schema": TARGET_DESCRIPTOR_SCHEMA,
        "descriptor": "source_promoted_rank256_reliability_conditioned_siglip2",
        "model_family": "v21b_v21c_shared_rank256_architecture",
        "query_free": True,
        "fallback": "inactive_or_ood_bitwise_accepted_v2",
        "normalization": "source_fit_only",
    }


TARGET_CONTRACT_SHA256 = canonical_json_sha256(target_contract())


def validate_target_descriptor(value: object) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "source_variant", "scene_id", "physical_space_id",
        "physical_space_authority", "producer", "target_execution_authority",
        "input_authority", "region_row_ids", "canonical_region_indices",
        "region_fingerprints", "semantic_descriptor", "reliability_score",
        "angular_budget_radians", "full_scalar_eligible_mask",
        "typed_context_valid_mask", "normalization_ood_mask",
        "effective_ood_mask", "active_update_mask", "immutable_fallback_mask",
        "descriptor_changed_mask", "fallback_bitwise_equal", "routing_audit",
        "channel_sha256", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("rank-256 target descriptor fields differ")
    payload = dict(value)
    if (
        payload["schema"] != TARGET_DESCRIPTOR_SCHEMA
        or payload["schema_version"] != 1
        or payload["contract"] != target_contract()
        or payload["contract_sha256"] != TARGET_CONTRACT_SHA256
        or payload["source_variant"] not in SOURCE_VARIANTS
        or payload["access_audit"] != target_access_audit()
    ):
        raise ValueError("rank-256 target descriptor header differs")
    payload["producer"] = _record(payload["producer"], label="target producer")
    payload["target_execution_authority"] = _record(
        payload["target_execution_authority"], label="target execution"
    )
    inputs = payload["input_authority"]
    expected_inputs = {
        "target_accepted_v2", "target_adaptive_typed_context",
        "factorized_primitive_state", "champion_checkpoint",
        "champion_normalization",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("rank-256 target inputs differ")
    payload["input_authority"] = {
        name: _record(inputs[name], label=f"target input {name}")
        for name in sorted(expected_inputs)
    }
    rows = payload["region_row_ids"]
    canonical = payload["canonical_region_indices"]
    fingerprints = payload["region_fingerprints"]
    descriptor = payload["semantic_descriptor"]
    regions = len(rows) if isinstance(rows, list) else -1
    masks = (
        "full_scalar_eligible_mask", "typed_context_valid_mask",
        "normalization_ood_mask", "effective_ood_mask", "active_update_mask",
        "immutable_fallback_mask", "descriptor_changed_mask",
    )
    if (
        regions <= 0 or len(set(rows)) != regions
        or not isinstance(fingerprints, list) or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or not torch.is_tensor(canonical) or canonical.dtype != torch.long
        or canonical.device.type != "cpu" or canonical.shape != (regions,)
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or not torch.is_tensor(descriptor) or descriptor.dtype != torch.float32
        or descriptor.device.type != "cpu" or descriptor.shape != (regions, 1536)
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("rank-256 target tensor layout differs")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("rank-256 target descriptor is not unit L2")
    for name in masks:
        tensor = payload[name]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.bool or tensor.device.type != "cpu" or tensor.shape != (regions,):
            raise ValueError(f"rank-256 target mask differs: {name}")
    for name in ("reliability_score", "angular_budget_radians"):
        tensor = payload[name]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.shape != (regions,) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"rank-256 diagnostic differs: {name}")
    if (
        not torch.equal(
            payload["active_update_mask"],
            ~(payload["immutable_fallback_mask"]),
        )
        or payload["fallback_bitwise_equal"] is not True
    ):
        raise ValueError("rank-256 fallback routing differs")
    expected_sha = target_channel_sha256(payload)
    if payload["channel_sha256"] != expected_sha:
        raise ValueError("rank-256 target channel SHA-256 differs")
    return payload


def target_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    names = (
        "canonical_region_indices", "semantic_descriptor", "reliability_score",
        "angular_budget_radians", "full_scalar_eligible_mask",
        "typed_context_valid_mask", "normalization_ood_mask",
        "effective_ood_mask", "active_update_mask", "immutable_fallback_mask",
        "descriptor_changed_mask",
    )
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        **{name: tensor_sha256(value[name]) for name in names},
    }


def validate_exact_query_manifest(value: object, *, scene_id: str) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "scene_id", "query_ids",
        "query_ids_sha256", "frozen_all_query_cache", "frozen_evaluator",
        "frozen_before_champion_query_execution",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("exact scene query manifest fields differ")
    result = dict(value)
    queries = result["query_ids"]
    if (
        result["schema"] != EXACT_QUERY_MANIFEST_SCHEMA
        or result["schema_version"] != 1
        or result["scene_id"] != scene_id
        or not isinstance(queries, list) or not queries
        or len(set(queries)) != len(queries)
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or result["query_ids_sha256"] != canonical_json_sha256(queries)
        or result["frozen_before_champion_query_execution"] is not True
    ):
        raise ValueError("exact scene query manifest identity differs")
    result["frozen_all_query_cache"] = _record(
        result["frozen_all_query_cache"], label="manifest all-query cache"
    )
    result["frozen_evaluator"] = _record(
        result["frozen_evaluator"], label="manifest evaluator"
    )
    return result


def validate_exact_query_receipt(value: object) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "status", "source_variant",
        "source_result", "scene_id", "query_manifest", "all_query_cache",
        "output_cache", "query_ids", "query_ids_sha256", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("exact-query subset receipt fields differ")
    result = dict(value)
    if (
        result["schema"] != EXACT_QUERY_RECEIPT_SCHEMA
        or result["schema_version"] != 1
        or result["status"] != "source_gated_exact_query_subset_complete"
        or result["source_variant"] not in SOURCE_VARIANTS
        or result["query_ids_sha256"] != canonical_json_sha256(result["query_ids"])
        or result["access_audit"] != query_access_audit()
    ):
        raise ValueError("exact-query subset receipt identity differs")
    for name in ("source_result", "query_manifest", "all_query_cache", "output_cache"):
        result[name] = _record(result[name], label=f"exact-query receipt {name}")
    return result


def query_contract() -> dict[str, Any]:
    return {
        "schema": QUERY_RELEVANCE_SCHEMA,
        "descriptor": "source_promoted_rank256_champion_target_unit_l2_siglip2",
        "positive_text": "source_gated_exact_scene_query_subset",
        "canonical_negative": "source_training_exact_frozen_four_row_bank",
        "formula": "binary_softmax_positive_vs_max_canonical_negative",
        "logit_scale": 10.0,
        "postprocess": "none",
        "absolute_relevance_boundary": 0.5,
    }


QUERY_CONTRACT_SHA256 = canonical_json_sha256(query_contract())


def query_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "query_ids": canonical_json_sha256(value["query_ids"]),
        "region_absolute_relevance": tensor_sha256(value["region_absolute_relevance"]),
    }


def validate_query_relevance(value: object) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "source_variant", "scene_id", "physical_space_id", "producer",
        "query_execution_authority", "input_authority", "region_row_ids",
        "canonical_region_indices", "region_fingerprints", "query_ids",
        "region_absolute_relevance", "channel_sha256", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("rank-256 query relevance fields differ")
    payload = dict(value)
    if (
        payload["schema"] != QUERY_RELEVANCE_SCHEMA
        or payload["schema_version"] != 1
        or payload["contract"] != query_contract()
        or payload["contract_sha256"] != QUERY_CONTRACT_SHA256
        or payload["source_variant"] not in SOURCE_VARIANTS
        or payload["access_audit"] != query_access_audit()
    ):
        raise ValueError("rank-256 query relevance header differs")
    payload["producer"] = _record(payload["producer"], label="query producer")
    payload["query_execution_authority"] = _record(
        payload["query_execution_authority"], label="query execution"
    )
    inputs = payload["input_authority"]
    expected = {
        "target_descriptor", "positive_text_cache", "positive_text_receipt",
        "canonical_negative_bank",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected:
        raise ValueError("rank-256 query input authority differs")
    payload["input_authority"] = {
        name: _record(inputs[name], label=f"query input {name}") for name in sorted(expected)
    }
    regions = len(payload["region_row_ids"])
    queries = len(payload["query_ids"])
    relevance = payload["region_absolute_relevance"]
    if (
        regions <= 0 or queries <= 0
        or len(set(payload["query_ids"])) != queries
        or not torch.is_tensor(relevance) or relevance.dtype != torch.float32
        or relevance.device.type != "cpu" or relevance.shape != (regions, queries)
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any()) or bool((relevance > 1).any())
    ):
        raise ValueError("rank-256 query relevance tensor differs")
    if payload["channel_sha256"] != query_channel_sha256(payload):
        raise ValueError("rank-256 query relevance SHA-256 differs")
    return payload


def physical_authority(dataset_id: str, scene_id: str, geometry_sha256: str):
    return target_physical_space_authority(
        dataset_id=dataset_id,
        scene_id=scene_id,
        geometry_checkpoint_sha256=geometry_sha256,
    )


__all__ = [name for name in globals() if name.isupper() or name.startswith("validate_")] + [
    "_output", "_record", "load_champion_model", "physical_authority",
    "query_access_audit", "query_channel_sha256", "query_contract",
    "source_canonical_negative", "source_gate_validator", "target_access_audit",
    "target_channel_sha256", "target_contract",
]
