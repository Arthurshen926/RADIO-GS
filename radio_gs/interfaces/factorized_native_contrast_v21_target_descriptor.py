"""Formal query-free target consumer for the promoted contrast V2.1 readout.

This is deliberately parallel to, rather than a mutation of, the frozen V1
three-arm target consumer.  Its source gate accepts exactly one source-only
contrast V2.1 promotion result and validates the complete nested authority,
normalization, contrast-reference, checkpoint, and official-head chain before
any target artifact may be opened.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import (
    materialize_factorized_native_target_descriptor as base_materializer,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as legacy_trainer,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as contrast_v2,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as contrast_v21,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


TARGET_EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_target_descriptor_"
    "execution_authority.v1"
)
TARGET_DESCRIPTOR_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_target_descriptor.v1"
)
EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA = "radio_gs.exact_query_descriptor_view.v1"
TARGET_INPUT_NAMES = ("target_accepted_v2", "factorized_primitive_state")
DESCRIPTOR_INPUT_NAMES = (
    "source_contrast_v21_result",
    "source_contrast_v21_checkpoint",
    "source_normalization",
    "source_contrast_reference",
    "official_radio_checkpoint",
    *TARGET_INPUT_NAMES,
)
TARGET_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_factorized_native_contrast_v21_target_descriptor.py"
)
TARGET_IMPLEMENTATION_DEPENDENCIES = {
    "target_descriptor_authority": Path(__file__).resolve(),
    "readout_interface": Path(readout.__file__).resolve(),
    "contrast_v21_source_trainer": Path(contrast_v21.__file__).resolve(),
    "base_canonical_forward": Path(base_materializer.__file__).resolve(),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def target_descriptor_access_audit() -> dict[str, bool]:
    return {
        "contrast_v21_source_promotion_validated_before_target_files": True,
        "source_selection_without_target_access": True,
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
        "source_selection": {
            "result_schema": contrast_v21.RESULT_SCHEMA,
            "checkpoint_schema": contrast_v21.CHECKPOINT_SCHEMA,
            "schema_version": 21,
            "required_status": (
                "source_only_contrast_v21_promotion_candidate_complete"
            ),
            "required_arm": DIRECTION_ONLY,
            "selected_step": "frozen_source_validation_gate",
        },
        "target_input": (
            "accepted_v2_canonical_region_rows_plus_exact_factorized_state_v2"
        ),
        "model_input": "unit_direction_only_with_validated_unused_auxiliary_channels",
        "raw_radio_vector_reconstruction": False,
        "model": "source_promoted_contrast_v21_direction_only_checkpoint",
        "projection": "frozen_official_siglip2_g_summary_head",
        "routing": "exact_factorized_state_required_at_region_anchor",
        "fallback": "bitwise_immutable_target_accepted_v2_e0",
        "descriptor": "float32_unit_l2_siglip2_1536",
        "exact_query_compatibility": {
            "view_schema": EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA,
            "channels": [
                "scene_id",
                "physical_space_id",
                "region_row_ids",
                "canonical_region_indices",
                "region_fingerprints",
                "semantic_descriptor",
            ],
            "formula_compatibility": (
                "direct_input_to_existing_calibrated_exact_cosine_relevance"
            ),
        },
        "legacy_v1_consumer_changed": False,
        "query_relevance_computed": False,
        "access_audit": target_descriptor_access_audit(),
    }


TARGET_DESCRIPTOR_CONTRACT_SHA256 = canonical_json_sha256(
    target_descriptor_contract()
)


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _canonical_output(value: object) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError("contrast V2.1 target output must be canonical absolute")
    return resolved


def _validate_contrast_reference(
    value: object, *, source_cohort_authority_sha256: str
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "fit_scenes",
        "heldout_validation_scenes",
        "equal_scene_weighting",
        "teacher_center",
        "teacher_center_norm",
        "teacher_center_squared_norm",
        "source_cohort_authority_sha256",
        "validation_contribution",
        "benchmark_opened",
        "source_access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("contrast V2.1 reference fields differ")
    reference = dict(value)
    center = reference.get("teacher_center")
    if (
        reference.get("schema") != contrast_v2.CONTRAST_REFERENCE_SCHEMA
        or reference.get("schema_version") != 1
        or reference.get("fit_scenes") != list(contrast_v21.TRAIN_SCENES)
        or reference.get("heldout_validation_scenes")
        != list(contrast_v21.VALIDATION_SCENES)
        or reference.get("equal_scene_weighting") is not True
        or reference.get("source_cohort_authority_sha256")
        != source_cohort_authority_sha256
        or reference.get("validation_contribution") is not False
        or reference.get("benchmark_opened") is not False
        or reference.get("source_access") != contrast_v21.source_access()
        or not torch.is_tensor(center)
        or center.dtype != torch.float32
        or center.device.type != "cpu"
        or center.shape != (shard.trainer.DESCRIPTOR_DIM,)
        or not center.is_contiguous()
        or not bool(torch.isfinite(center).all())
    ):
        raise ValueError("contrast V2.1 reference contract differs")
    norm = float(center.norm())
    squared_norm = float(center.square().sum())
    if (
        not math.isclose(float(reference["teacher_center_norm"]), norm, rel_tol=0.0, abs_tol=1e-7)
        or not math.isclose(
            float(reference["teacher_center_squared_norm"]),
            squared_norm,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not 0.0 < norm < 1.0
    ):
        raise ValueError("contrast V2.1 reference center differs")
    return reference


def _validate_result_header(
    raw: object, *, record: Mapping[str, str]
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "arm",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "normalization",
        "contrast_reference",
        "checkpoint",
        "selected_step",
        "history",
        "last_training_step",
        "benchmark_opened",
        "source_access",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("contrast V2.1 source result fields differ")
    result = dict(raw)
    contract = contrast_v21.training_contract()
    selected = result.get("selected_step")
    history = result.get("history")
    expected_steps = [
        0,
        *range(
            contrast_v21.EVALUATION_INTERVAL,
            contrast_v21.OPTIMIZER_STEPS + 1,
            contrast_v21.EVALUATION_INTERVAL,
        ),
    ]
    if (
        result.get("schema") != contrast_v21.RESULT_SCHEMA
        or result.get("schema_version") != 21
        or result.get("status")
        != "source_only_contrast_v21_promotion_candidate_complete"
        or result.get("arm") != DIRECTION_ONLY
        or result.get("training_contract") != contract
        or result.get("training_contract_sha256")
        != canonical_json_sha256(contract)
        or result.get("benchmark_opened") is not False
        or result.get("source_access") != contrast_v21.source_access()
        or not isinstance(selected, int)
        or not isinstance(history, list)
        or [entry.get("step") for entry in history] != expected_steps
        or contrast_v21.select_step(history) != selected
        or history[expected_steps.index(selected)]
        .get("validation", {})
        .get("selection", {})
        .get("eligible")
        is not True
        or not isinstance(result.get("last_training_step"), Mapping)
        or result["last_training_step"].get("step")
        != contrast_v21.OPTIMIZER_STEPS
    ):
        raise ValueError("contrast V2.1 source result contract differs")
    for name in (
        "execution_authority",
        "normalization",
        "contrast_reference",
        "checkpoint",
    ):
        result[name] = _record(result[name], label=f"contrast V2.1 {name}")
    result["verified_record"] = dict(record)
    return result


def _validate_checkpoint(
    result: Mapping[str, Any], normalization: Mapping[str, Any]
) -> dict[str, Any]:
    raw, digest, source = load_torch_mapping(
        result["checkpoint"]["path"],
        expected_sha256=result["checkpoint"]["sha256"],
        map_location="cpu",
        label="contrast V2.1 source checkpoint",
    )
    required = {
        "schema",
        "schema_version",
        "training_contract",
        "training_contract_sha256",
        "interface_contract_sha256",
        "model_architecture",
        "model_state_dict",
        "model_state_dict_sha256",
        "normalization",
        "contrast_reference",
        "execution_authority",
        "selected_step",
        "source_access",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("contrast V2.1 checkpoint fields differ")
    checkpoint = dict(raw)
    model = readout.build_model(DIRECTION_ONLY, normalization)
    state = checkpoint.get("model_state_dict")
    selected_entry = result["history"][
        [entry["step"] for entry in result["history"]].index(result["selected_step"])
    ]
    if (
        checkpoint.get("schema") != contrast_v21.CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != 21
        or checkpoint.get("training_contract") != result["training_contract"]
        or checkpoint.get("training_contract_sha256")
        != result["training_contract_sha256"]
        or checkpoint.get("interface_contract_sha256")
        != readout.INTERFACE_CONTRACT_SHA256
        or checkpoint.get("model_architecture")
        != model.architecture(readout.INTERFACE_CONTRACT_SHA256)
        or checkpoint.get("normalization") != result["normalization"]
        or checkpoint.get("contrast_reference") != result["contrast_reference"]
        or checkpoint.get("execution_authority")
        != result["execution_authority"]
        or checkpoint.get("selected_step") != result["selected_step"]
        or checkpoint.get("source_access") != contrast_v21.source_access()
        or not isinstance(state, Mapping)
        or checkpoint.get("model_state_dict_sha256")
        != contrast_v2._state_sha(state)
        or checkpoint.get("model_state_dict_sha256")
        != selected_entry.get("model_state_dict_sha256")
    ):
        raise ValueError("contrast V2.1 checkpoint contract differs")
    model.load_state_dict(state, strict=True)
    checkpoint["verified_record"] = {"path": str(source), "sha256": digest}
    return checkpoint


def validate_source_contrast_v21_result(record_value: object) -> dict[str, Any]:
    """Validate the complete source-only V2.1 promotion chain."""

    record = _record(record_value, label="contrast V2.1 source result")
    raw, digest, source = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="contrast V2.1 source result",
    )
    if record != {"path": str(source), "sha256": digest}:
        raise ValueError("contrast V2.1 source result record differs")
    result = _validate_result_header(raw, record=record)

    # This recursively verifies V2.1, V2, and base-V1 code/authority hashes.
    prepared = contrast_v21.prepare_inputs(
        result["execution_authority"]["path"],
        expected_sha256=result["execution_authority"]["sha256"],
    )
    if (
        result["execution_authority"]
        != {
            "path": prepared.authority["verified_path"],
            "sha256": prepared.authority["verified_sha256"],
        }
    ):
        raise ValueError("contrast V2.1 execution authority record differs")

    normalization_raw, _, _ = load_torch_mapping(
        result["normalization"]["path"],
        expected_sha256=result["normalization"]["sha256"],
        map_location="cpu",
        label="contrast V2.1 source normalization",
    )
    normalization = readout.validate_source_normalization(normalization_raw)
    cohort_sha = prepared.base_v2.source.registry["authority_sha256"]
    if normalization.get("source_state_cohort_authority_sha256") != cohort_sha:
        raise ValueError("contrast V2.1 normalization cohort differs")

    reference_raw, _, _ = load_torch_mapping(
        result["contrast_reference"]["path"],
        expected_sha256=result["contrast_reference"]["sha256"],
        map_location="cpu",
        label="contrast V2.1 source reference",
    )
    reference = _validate_contrast_reference(
        reference_raw, source_cohort_authority_sha256=cohort_sha
    )
    checkpoint = _validate_checkpoint(result, normalization)

    official = _record(
        prepared.base_v2.source.authority["official_radio_checkpoint"],
        label="official RADIO checkpoint",
    )
    verified_official = validate_file_record(
        official, label="official RADIO checkpoint"
    )
    if (
        str(verified_official) != official["path"]
        or official["sha256"] != shard.OFFICIAL_RADIO_CHECKPOINT_SHA256
    ):
        raise ValueError("official RADIO checkpoint singleton differs")
    return {
        "result": result,
        "checkpoint": checkpoint,
        "normalization": normalization,
        "contrast_reference": reference,
        "official_radio_checkpoint": official,
        "selected_step": result["selected_step"],
        "arm": DIRECTION_ONLY,
        "source_only_passed": True,
    }


def validate_target_execution_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="contrast V2.1 target execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "source_contrast_v21_result",
        "implementation",
        "implementation_dependencies",
        "target_inputs",
        "target_descriptor_output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("contrast V2.1 target execution fields differ")
    authority = dict(raw)
    if (
        authority.get("schema") != TARGET_EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_contrast_v21_source_promotion_for_query_free_target"
        or authority.get("materialization_authorized") is not True
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != target_descriptor_access_audit()
    ):
        raise ValueError("contrast V2.1 target execution header differs")

    # Ordering is part of the contract: target paths are not opened until this
    # complete source-only promotion gate returns successfully.
    source_gate = validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )

    implementation = validate_file_record(
        authority["implementation"], label="contrast V2.1 target implementation"
    )
    if implementation != TARGET_IMPLEMENTATION_PATH:
        raise ValueError("contrast V2.1 target implementation differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        TARGET_IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("contrast V2.1 target dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in TARGET_IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"contrast V2.1 dependency {name}"
        )
        if verified != expected:
            raise ValueError(f"contrast V2.1 dependency differs: {name}")
        verified_dependencies[name] = _record(
            dependencies[name], label=f"contrast V2.1 dependency {name}"
        )
    target_inputs = authority.get("target_inputs")
    if not isinstance(target_inputs, Mapping) or set(target_inputs) != set(
        TARGET_INPUT_NAMES
    ):
        raise ValueError("contrast V2.1 target inputs differ")
    verified_inputs: dict[str, dict[str, str]] = {}
    for name in TARGET_INPUT_NAMES:
        shaped = _record(target_inputs[name], label=f"contrast V2.1 target {name}")
        verified = validate_file_record(
            shaped, label=f"contrast V2.1 target {name}"
        )
        if str(verified) != shaped["path"]:
            raise ValueError(f"contrast V2.1 target {name} is not canonical")
        verified_inputs[name] = shaped
    output = _canonical_output(authority["target_descriptor_output"])
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("contrast V2.1 target output differs")
    authority["source_contrast_v21_result"] = _record(
        authority["source_contrast_v21_result"],
        label="contrast V2.1 source result",
    )
    authority["implementation"] = _record(
        authority["implementation"], label="contrast V2.1 target implementation"
    )
    authority["implementation_dependencies"] = verified_dependencies
    authority["target_inputs"] = verified_inputs
    authority["target_descriptor_output"] = output
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    authority["verified_source_gate"] = source_gate
    return authority


def target_descriptor_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "semantic_descriptor": tensor_sha256(value["semantic_descriptor"]),
        "exact_state_anchor_mask": tensor_sha256(value["exact_state_anchor_mask"]),
        "active_update_mask": tensor_sha256(value["active_update_mask"]),
        "immutable_fallback_mask": tensor_sha256(value["immutable_fallback_mask"]),
        "descriptor_changed_mask": tensor_sha256(value["descriptor_changed_mask"]),
    }


def validate_target_descriptor_authority(value: object) -> dict[str, Any]:
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
        "source_arm",
        "source_selected_step",
        "source_gate_audit",
        "region_row_ids",
        "canonical_region_indices",
        "region_fingerprints",
        "semantic_descriptor",
        "exact_state_anchor_mask",
        "active_update_mask",
        "immutable_fallback_mask",
        "descriptor_changed_mask",
        "fallback_bitwise_equal",
        "routing_audit",
        "channel_sha256",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("contrast V2.1 target descriptor fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != TARGET_DESCRIPTOR_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("contract") != target_descriptor_contract()
        or payload.get("contract_sha256") != TARGET_DESCRIPTOR_CONTRACT_SHA256
        or payload.get("access_audit") != target_descriptor_access_audit()
        or payload.get("source_arm") != DIRECTION_ONLY
        or not isinstance(payload.get("source_selected_step"), int)
        or payload.get("source_gate_audit")
        != {
            "result_schema": contrast_v21.RESULT_SCHEMA,
            "checkpoint_schema": contrast_v21.CHECKPOINT_SCHEMA,
            "schema_version": 21,
            "status": "source_only_contrast_v21_promotion_candidate_complete",
            "source_only_passed": True,
        }
        or payload.get("fallback_bitwise_equal") is not True
    ):
        raise ValueError("contrast V2.1 target descriptor contract differs")
    physical = payload.get("physical_space_authority")
    if not isinstance(physical, Mapping):
        raise ValueError("contrast V2.1 target physical-space authority differs")
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
        raise ValueError("contrast V2.1 target physical-space binding differs")
    payload["physical_space_authority"] = expected_physical
    payload["producer"] = _record(payload["producer"], label="descriptor producer")
    payload["target_execution_authority"] = _record(
        payload["target_execution_authority"], label="target execution authority"
    )
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != set(DESCRIPTOR_INPUT_NAMES):
        raise ValueError("contrast V2.1 descriptor input records differ")
    payload["input_authority"] = {
        name: _record(inputs[name], label=f"contrast V2.1 descriptor {name}")
        for name in DESCRIPTOR_INPUT_NAMES
    }
    rows = payload.get("region_row_ids")
    canonical = payload.get("canonical_region_indices")
    fingerprints = payload.get("region_fingerprints")
    descriptor = payload.get("semantic_descriptor")
    regions = len(rows) if isinstance(rows, list) else -1
    masks: dict[str, torch.Tensor] = {}
    for name in (
        "exact_state_anchor_mask",
        "active_update_mask",
        "immutable_fallback_mask",
        "descriptor_changed_mask",
    ):
        tensor = payload.get(name)
        if (
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.bool
            or tensor.device.type != "cpu"
            or tensor.shape != (regions,)
        ):
            raise ValueError(f"contrast V2.1 target {name} differs")
        masks[name] = tensor
    if (
        regions <= 0
        or len(set(rows)) != regions
        or any(not isinstance(row, str) or not row for row in rows)
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.device.type != "cpu"
        or canonical.shape != (regions,)
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or not torch.is_tensor(descriptor)
        or descriptor.dtype != torch.float32
        or descriptor.device.type != "cpu"
        or descriptor.shape != (regions, shard.trainer.DESCRIPTOR_DIM)
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("contrast V2.1 target tensor layout differs")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("contrast V2.1 target descriptor is not unit L2")
    exact = masks["exact_state_anchor_mask"]
    active = masks["active_update_mask"]
    fallback = masks["immutable_fallback_mask"]
    changed = masks["descriptor_changed_mask"]
    if (
        not torch.equal(active, exact)
        or not torch.equal(fallback, ~active)
        or bool((changed & fallback).any())
    ):
        raise ValueError("contrast V2.1 target routing masks differ")
    expected_audit = {
        "regions": regions,
        "exact_state_anchor": int(exact.sum()),
        "active_update": int(active.sum()),
        "immutable_fallback": int(fallback.sum()),
        "descriptor_changed": int(changed.sum()),
    }
    if payload.get("routing_audit") != expected_audit:
        raise ValueError("contrast V2.1 target routing audit differs")
    if payload.get("channel_sha256") != target_descriptor_channel_sha256(payload):
        raise ValueError("contrast V2.1 target channel SHA-256 differs")
    return payload


def exact_query_descriptor_view(value: object) -> dict[str, Any]:
    payload = validate_target_descriptor_authority(value)
    return {
        "schema": EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA,
        "scene_id": payload["scene_id"],
        "physical_space_id": payload["physical_space_id"],
        "region_row_ids": list(payload["region_row_ids"]),
        "canonical_region_indices": payload["canonical_region_indices"].clone(),
        "region_fingerprints": list(payload["region_fingerprints"]),
        "semantic_descriptor": payload["semantic_descriptor"].clone(),
        "source_descriptor_schema": TARGET_DESCRIPTOR_SCHEMA,
        "source_descriptor_contract_sha256": TARGET_DESCRIPTOR_CONTRACT_SHA256,
    }


__all__ = [
    "DESCRIPTOR_INPUT_NAMES",
    "EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA",
    "TARGET_DESCRIPTOR_CONTRACT_SHA256",
    "TARGET_DESCRIPTOR_SCHEMA",
    "TARGET_EXECUTION_SCHEMA",
    "TARGET_IMPLEMENTATION_DEPENDENCIES",
    "TARGET_IMPLEMENTATION_PATH",
    "TARGET_INPUT_NAMES",
    "exact_query_descriptor_view",
    "target_descriptor_access_audit",
    "target_descriptor_channel_sha256",
    "target_descriptor_contract",
    "validate_source_contrast_v21_result",
    "validate_target_descriptor_authority",
    "validate_target_execution_authority",
]
