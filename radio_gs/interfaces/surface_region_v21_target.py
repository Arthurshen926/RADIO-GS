"""Formal target execution and output authorities for V2.1 descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.interfaces import surface_region_target_accepted_v2 as accepted_formal
from radio_gs.interfaces import (
    surface_region_target_adaptive_typed_context as adaptive_formal,
)
from radio_gs.interfaces import surface_region_v21_source_gate as source_formal
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    validate_file_record,
)


TARGET_EXECUTION_SCHEMA = "radio_gs.surface_region_v21_target_execution_authority.v1"
TARGET_DESCRIPTOR_SCHEMA = "radio_gs.surface_region_v21_target_descriptor_authority.v1"
TARGET_INPUT_NAMES = (
    "target_accepted_v2",
    "target_adaptive_typed_context",
    "factorized_primitive_state",
    "v21_checkpoint",
    "v21_normalization",
)
TARGET_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_surface_region_v21_target_descriptor.py"
)
TARGET_IMPLEMENTATION_DEPENDENCIES = {
    "target_descriptor_authority": Path(__file__).resolve(),
    "target_accepted_v2_authority": Path(accepted_formal.__file__).resolve(),
    "target_adaptive_context_authority": Path(adaptive_formal.__file__).resolve(),
    "source_promotion_gate": Path(source_formal.__file__).resolve(),
}
TARGET_PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def target_descriptor_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_validated_before_target_files": True,
        "target_geometry_authorities_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "query_relevance_computed": False,
        "target_metrics_computed": False,
    }


def target_descriptor_contract() -> dict[str, Any]:
    return {
        "schema": TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "base": "immutable_target_accepted_v2_e0",
        "context": "target_adaptive_typed_context_authority_v1",
        "full_scalar": "exact_factorized_state_overlap_summary_18d",
        "model": "source_promoted_v21_checkpoint",
        "normalization": "source_train4_frozen_normalization",
        "routing": "full_scalar_eligible_and_typed_valid_and_not_source_envelope_ood",
        "fallback": "bitwise_immutable_target_accepted_v2_e0",
        "descriptor": "float32_unit_l2_siglip2",
        "query_relevance_computed": False,
        "access_audit": target_descriptor_access_audit(),
    }


TARGET_DESCRIPTOR_CONTRACT_SHA256 = canonical_json_sha256(target_descriptor_contract())


def _record_shape(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("V2.1 target descriptor output must be absolute and canonical")
    return resolved


def validate_target_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_scene_id: str | None = None,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source promotion before opening any target input record."""

    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="V2.1 target descriptor execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "source_pilot_result",
        "implementation",
        "implementation_dependencies",
        "preregistration",
        "target_inputs",
        "target_descriptor_output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    scene = str(authority.get("scene_id", ""))
    physical = str(authority.get("physical_space_id", ""))
    if (
        set(authority) != required
        or authority.get("schema") != TARGET_EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_v21_source_promotion_for_query_free_descriptor_only"
        or not scene
        or not physical
        or authority.get("materialization_authorized") is not True
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != target_descriptor_access_audit()
        or (expected_scene_id is not None and scene != str(expected_scene_id))
    ):
        raise ValueError("V2.1 target descriptor execution header differs")
    source_result = _record_shape(
        authority["source_pilot_result"], label="V2.1 source pilot result"
    )

    # This call must remain before validate_file_record on every target input.
    source_gate = validate_source_pilot_chain(
        source_result["path"],
        expected_sha256=source_result["sha256"],
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("V2.1 source promotion is not authorized")

    implementation = validate_file_record(
        authority["implementation"], label="V2.1 target descriptor implementation"
    )
    if implementation != TARGET_IMPLEMENTATION_PATH:
        raise ValueError("V2.1 target descriptor implementation differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        TARGET_IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("V2.1 target descriptor implementation dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in TARGET_IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"V2.1 target descriptor dependency {name}"
        )
        if verified != expected:
            raise ValueError(f"V2.1 target descriptor dependency differs: {name}")
        verified_dependencies[name] = _record_shape(
            dependencies[name], label=f"V2.1 target descriptor dependency {name}"
        )
    preregistration = validate_file_record(
        authority["preregistration"], label="V2.1 target descriptor preregistration"
    )
    if preregistration != TARGET_PREREGISTRATION_PATH:
        raise ValueError("V2.1 target descriptor preregistration differs")

    inputs = authority.get("target_inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(TARGET_INPUT_NAMES):
        raise ValueError("V2.1 target descriptor input authority differs")
    verified_inputs: dict[str, dict[str, str]] = {}
    for name in TARGET_INPUT_NAMES:
        shaped = _record_shape(inputs[name], label=f"V2.1 target {name}")
        verified = validate_file_record(shaped, label=f"V2.1 target {name}")
        if str(verified) != shaped["path"]:
            raise ValueError(f"V2.1 target {name} path is not canonical")
        verified_inputs[name] = shaped
    if (
        verified_inputs["v21_checkpoint"] != source_gate["checkpoint"]
        or verified_inputs["v21_normalization"]
        != source_gate["normalization_authority"]
    ):
        raise ValueError("V2.1 target model inputs differ from promoted source gate")
    output = _canonical_output(authority["target_descriptor_output"])
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("V2.1 target descriptor output differs")
    authority["target_inputs"] = verified_inputs
    authority["implementation"] = _record_shape(
        authority["implementation"], label="V2.1 target descriptor implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority["preregistration"] = _record_shape(
        authority["preregistration"], label="V2.1 target descriptor preregistration"
    )
    authority["target_descriptor_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    return authority


def validate_target_descriptor_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1 target descriptor authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "physical_space_authority",
        "producer",
        "target_execution_authority",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "region_fingerprints",
        "semantic_descriptor",
        "full_scalar_eligible_mask",
        "typed_context_valid_mask",
        "normalization_ood_mask",
        "effective_ood_mask",
        "active_update_mask",
        "immutable_fallback_mask",
        "descriptor_changed_mask",
        "fallback_bitwise_equal",
        "routing_audit",
        "channel_sha256",
        "access_audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != TARGET_DESCRIPTOR_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("contract") != target_descriptor_contract()
        or payload.get("contract_sha256") != TARGET_DESCRIPTOR_CONTRACT_SHA256
        or payload.get("access_audit") != target_descriptor_access_audit()
        or payload.get("fallback_bitwise_equal") is not True
    ):
        raise ValueError("V2.1 target descriptor contract differs")
    physical = payload["physical_space_authority"]
    if not isinstance(physical, Mapping):
        raise ValueError("V2.1 target descriptor physical-space authority differs")
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
        raise ValueError("V2.1 target descriptor physical-space binding differs")
    payload["physical_space_authority"] = expected_physical
    for name in ("producer", "target_execution_authority"):
        payload[name] = _record_shape(payload[name], label=f"V2.1 descriptor {name}")
    inputs = payload["input_authority"]
    if not isinstance(inputs, Mapping) or set(inputs) != set(TARGET_INPUT_NAMES):
        raise ValueError("V2.1 target descriptor input records differ")
    payload["input_authority"] = {
        name: _record_shape(inputs[name], label=f"V2.1 descriptor {name}")
        for name in TARGET_INPUT_NAMES
    }
    rows = payload["region_row_ids"]
    canonical = payload["canonical_region_indices"]
    fingerprints = payload["region_fingerprints"]
    descriptor = payload["semantic_descriptor"]
    regions = len(rows) if isinstance(rows, list) else -1
    masks = {}
    for name in (
        "full_scalar_eligible_mask",
        "typed_context_valid_mask",
        "normalization_ood_mask",
        "effective_ood_mask",
        "active_update_mask",
        "immutable_fallback_mask",
        "descriptor_changed_mask",
    ):
        tensor = payload[name]
        if (
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.bool
            or tensor.shape != (regions,)
        ):
            raise ValueError(f"V2.1 target descriptor {name} differs")
        masks[name] = tensor
    if (
        regions <= 0
        or len(set(rows)) != regions
        or any(not isinstance(row, str) or not row for row in rows)
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.shape != (regions,)
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or not torch.is_tensor(descriptor)
        or descriptor.dtype != torch.float32
        or descriptor.shape != (regions, shard.trainer.DESCRIPTOR_DIM)
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("V2.1 target descriptor tensor layout differs")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("V2.1 target descriptor is not unit L2")
    eligible = masks["full_scalar_eligible_mask"]
    typed = masks["typed_context_valid_mask"]
    norm_ood = masks["normalization_ood_mask"]
    effective_ood = masks["effective_ood_mask"]
    active = masks["active_update_mask"]
    fallback = masks["immutable_fallback_mask"]
    changed = masks["descriptor_changed_mask"]
    expected_effective = norm_ood | ~eligible
    expected_active = typed & ~expected_effective
    if (
        not torch.equal(effective_ood, expected_effective)
        or not torch.equal(active, expected_active)
        or not torch.equal(fallback, ~active)
        or bool((changed & ~active).any())
    ):
        raise ValueError("V2.1 target descriptor routing masks differ")
    audit = payload["routing_audit"]
    expected_audit = {
        "regions": regions,
        "full_scalar_eligible": int(eligible.sum()),
        "typed_context_valid": int(typed.sum()),
        "normalization_ood": int(norm_ood.sum()),
        "effective_ood": int(effective_ood.sum()),
        "active_update": int(active.sum()),
        "immutable_fallback": int(fallback.sum()),
        "descriptor_changed": int(changed.sum()),
    }
    if audit != expected_audit:
        raise ValueError("V2.1 target descriptor routing audit differs")
    expected_channels = target_descriptor_channel_sha256(payload)
    if payload["channel_sha256"] != expected_channels:
        raise ValueError("V2.1 target descriptor channel SHA-256 differs")
    return payload


def target_descriptor_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    masks = {
        name: value[name]
        for name in (
            "full_scalar_eligible_mask",
            "typed_context_valid_mask",
            "normalization_ood_mask",
            "effective_ood_mask",
            "active_update_mask",
            "immutable_fallback_mask",
            "descriptor_changed_mask",
        )
    }
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": shard._tensor_sha(
            value["canonical_region_indices"]
        ),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "semantic_descriptor": shard._tensor_sha(value["semantic_descriptor"]),
        **{name: shard._tensor_sha(tensor) for name, tensor in masks.items()},
    }


__all__ = [
    "TARGET_DESCRIPTOR_CONTRACT_SHA256",
    "TARGET_DESCRIPTOR_SCHEMA",
    "TARGET_EXECUTION_SCHEMA",
    "TARGET_INPUT_NAMES",
    "TARGET_IMPLEMENTATION_DEPENDENCIES",
    "TARGET_IMPLEMENTATION_PATH",
    "TARGET_PREREGISTRATION_PATH",
    "target_descriptor_access_audit",
    "target_descriptor_contract",
    "target_descriptor_channel_sha256",
    "validate_target_descriptor_authority",
    "validate_target_execution_authority",
]
