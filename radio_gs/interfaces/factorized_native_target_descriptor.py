"""Formal query-free target descriptor for the factorized-native readout.

The source gate consumes all three frozen source-only arm results and chooses
the winner before any target artifact is opened.  The target payload exposes
the small descriptor view used by exact query scoring without impersonating
either legacy V2.1 or rank-256 descriptor schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.scripts import (
    materialize_full_scalar_clean_training_shard as shard,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as source_trainer,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


TARGET_EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_target_descriptor_execution_authority.v1"
)
TARGET_DESCRIPTOR_SCHEMA = "radio_gs.factorized_native_target_descriptor.v1"
EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA = "radio_gs.exact_query_descriptor_view.v1"
TARGET_INPUT_NAMES = ("target_accepted_v2", "factorized_primitive_state")
DESCRIPTOR_INPUT_NAMES = (
    "source_arm_results",
    "winner_checkpoint",
    "winner_normalization",
    "official_radio_checkpoint",
    *TARGET_INPUT_NAMES,
)
TARGET_IMPLEMENTATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_factorized_native_target_descriptor.py"
)
TARGET_IMPLEMENTATION_DEPENDENCIES = {
    "target_descriptor_authority": Path(__file__).resolve(),
    "readout_interface": Path(readout.__file__).resolve(),
    "source_trainer": Path(source_trainer.__file__).resolve(),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def target_descriptor_access_audit() -> dict[str, bool]:
    return {
        "all_source_arms_validated_before_target_files": True,
        "source_winner_selected_without_target_access": True,
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
            "cohort": "frozen_exact4train_2validation",
            "required_arms": list(FACTORIZED_NATIVE_READOUT_ARMS),
            "candidate_gate": "per_arm_frozen_non_regression_and_improvement",
            "ranking": [
                "maximum_selected_macro_mean_all_view_cosine",
                "maximum_selected_macro_p05_row_mean_all_view_cosine",
                "fixed_arm_order",
            ],
        },
        "target_input": (
            "accepted_v2_canonical_region_rows_plus_exact_factorized_state_v2"
        ),
        "model_input": "unit_direction_plus_separate_log_amplitude_plus_state",
        "raw_radio_vector_reconstruction": False,
        "model": "source_selected_factorized_native_checkpoint",
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
        "legacy_default_changed": False,
        "query_relevance_computed": False,
        "access_audit": target_descriptor_access_audit(),
    }


TARGET_DESCRIPTOR_CONTRACT_SHA256 = canonical_json_sha256(
    target_descriptor_contract()
)


def _record(value: object, *, label: str) -> dict[str, str]:
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
        raise ValueError("factorized-native target output must be canonical absolute")
    return resolved


def _validate_result_header(
    raw: object, *, arm: str, record: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"factorized-native {arm} source result must be a mapping")
    result = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "arm",
        "training_contract",
        "training_contract_sha256",
        "execution_authority",
        "normalization",
        "checkpoint",
        "selected_step",
        "history",
        "benchmark_opened",
        "source_access",
    }
    contract = source_trainer.training_contract(arm)
    status = result.get("status")
    selected = result.get("selected_step")
    checkpoint = result.get("checkpoint")
    if (
        set(result) != required
        or result.get("schema") != source_trainer.RESULT_SCHEMA
        or result.get("schema_version") != 1
        or result.get("arm") != arm
        or result.get("training_contract") != contract
        or result.get("training_contract_sha256")
        != canonical_json_sha256(contract)
        or result.get("benchmark_opened") is not False
        or result.get("source_access") != readout.source_access()
        or status
        not in {
            "source_only_promotion_candidate_complete",
            "source_only_complete_no_eligible_candidate",
        }
        or (status.endswith("no_eligible_candidate") and selected is not None)
        or (status.endswith("no_eligible_candidate") and checkpoint is not None)
        or (status.endswith("promotion_candidate_complete") and not isinstance(selected, int))
        or (status.endswith("promotion_candidate_complete") and checkpoint is None)
    ):
        raise ValueError(f"factorized-native {arm} source result differs")
    history = result["history"]
    if (
        not isinstance(history, list)
        or len(history) != source_trainer.OPTIMIZER_STEPS + 1
        or [entry.get("step") for entry in history] != list(range(len(history)))
    ):
        raise ValueError(f"factorized-native {arm} source history differs")
    result["execution_authority"] = _record(
        result["execution_authority"], label=f"{arm} source execution authority"
    )
    result["normalization"] = _record(
        result["normalization"], label=f"{arm} source normalization"
    )
    if checkpoint is not None:
        result["checkpoint"] = _record(
            checkpoint, label=f"{arm} source checkpoint"
        )
        entry = history[selected]
        selection = entry.get("validation", {}).get("selection", {})
        if selection.get("eligible") is not True:
            raise ValueError(f"factorized-native {arm} selected step is not eligible")
    result["verified_record"] = dict(record)
    return result


def _validate_candidate_checkpoint(
    result: Mapping[str, Any], normalization: Mapping[str, Any]
) -> dict[str, Any]:
    arm = str(result["arm"])
    raw, digest, source = load_torch_mapping(
        result["checkpoint"]["path"],
        expected_sha256=result["checkpoint"]["sha256"],
        map_location="cpu",
        label=f"factorized-native {arm} source checkpoint",
    )
    if not isinstance(raw, Mapping):
        raise ValueError("factorized-native source checkpoint must be a mapping")
    checkpoint = dict(raw)
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
        "execution_authority",
        "selected_step",
        "source_access",
    }
    model = readout.build_model(arm, normalization)
    state = checkpoint.get("model_state_dict")
    if (
        set(checkpoint) != required
        or checkpoint.get("schema") != source_trainer.CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("training_contract") != result["training_contract"]
        or checkpoint.get("training_contract_sha256")
        != result["training_contract_sha256"]
        or checkpoint.get("interface_contract_sha256")
        != readout.INTERFACE_CONTRACT_SHA256
        or checkpoint.get("model_architecture")
        != model.architecture(readout.INTERFACE_CONTRACT_SHA256)
        or checkpoint.get("normalization") != result["normalization"]
        or checkpoint.get("execution_authority") != result["execution_authority"]
        or checkpoint.get("selected_step") != result["selected_step"]
        or checkpoint.get("source_access") != readout.source_access()
        or not isinstance(state, Mapping)
        or checkpoint.get("model_state_dict_sha256")
        != source_trainer._state_sha(state)
        or checkpoint.get("model_state_dict_sha256")
        != result["history"][result["selected_step"]].get(
            "model_state_dict_sha256"
        )
    ):
        raise ValueError("factorized-native source checkpoint contract differs")
    model.load_state_dict(state, strict=True)
    checkpoint["verified_record"] = {"path": str(source), "sha256": digest}
    return checkpoint


def validate_source_arm_winner(
    records: object,
) -> dict[str, Any]:
    """Validate all arms and choose the frozen source-validation winner."""

    if not isinstance(records, Mapping) or tuple(records.keys()) != tuple(
        FACTORIZED_NATIVE_READOUT_ARMS
    ):
        raise ValueError("source arm results must follow the exact frozen arm order")
    results: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    common_execution: dict[str, str] | None = None
    official_radio: dict[str, str] | None = None
    for arm_index, arm in enumerate(FACTORIZED_NATIVE_READOUT_ARMS):
        record = _record(records[arm], label=f"factorized-native {arm} result")
        raw, digest, source = load_json_object(
            record["path"],
            expected_sha256=record["sha256"],
            label=f"factorized-native {arm} source result",
        )
        if record != {"path": str(source), "sha256": digest}:
            raise ValueError("factorized-native source result record differs")
        result = _validate_result_header(raw, arm=arm, record=record)

        normalization_raw, _, _ = load_torch_mapping(
            result["normalization"]["path"],
            expected_sha256=result["normalization"]["sha256"],
            map_location="cpu",
            label=f"factorized-native {arm} source normalization",
        )
        normalization = readout.validate_source_normalization(normalization_raw)
        execution_raw, _, execution_path = load_json_object(
            result["execution_authority"]["path"],
            expected_sha256=result["execution_authority"]["sha256"],
            label=f"factorized-native {arm} source execution authority",
        )
        execution = source_trainer.validate_execution_authority(execution_raw)
        if str(execution_path) != result["execution_authority"]["path"]:
            raise ValueError("factorized-native source execution path differs")
        for name, expected_path in source_trainer._expected_code_paths().items():
            verified_code = validate_file_record(
                execution[name], label=f"factorized-native source {name}"
            )
            if verified_code != expected_path.resolve():
                raise ValueError(f"factorized-native source code differs: {name}")
        if common_execution is None:
            common_execution = dict(result["execution_authority"])
            official_radio = _record(
                execution["official_radio_checkpoint"],
                label="official RADIO checkpoint",
            )
            validate_file_record(official_radio, label="official RADIO checkpoint")
            if official_radio["sha256"] != shard.OFFICIAL_RADIO_CHECKPOINT_SHA256:
                raise ValueError("official RADIO checkpoint singleton differs")
        elif result["execution_authority"] != common_execution:
            raise ValueError("factorized-native arms use different source authorities")

        result["verified_normalization"] = normalization
        if result["checkpoint"] is not None:
            checkpoint = _validate_candidate_checkpoint(result, normalization)
            selected = result["history"][result["selected_step"]]["validation"]
            candidates.append(
                {
                    "arm": arm,
                    "arm_index": arm_index,
                    "result": result,
                    "checkpoint": checkpoint,
                    "macro_mean": float(selected["macro_mean_all_view_cosine"]),
                    "macro_p05": float(
                        selected["macro_p05_row_mean_all_view_cosine"]
                    ),
                }
            )
        results[arm] = result
    if not candidates or official_radio is None:
        raise ValueError("no factorized-native source arm passed the frozen gate")
    winner = max(
        candidates,
        key=lambda item: (item["macro_mean"], item["macro_p05"], -item["arm_index"]),
    )
    return {
        "arm_results": results,
        "winner_arm": winner["arm"],
        "winner_result": winner["result"],
        "winner_checkpoint": winner["checkpoint"],
        "winner_normalization": winner["result"]["verified_normalization"],
        "official_radio_checkpoint": official_radio,
        "ranking": {
            "macro_mean_all_view_cosine": winner["macro_mean"],
            "macro_p05_row_mean_all_view_cosine": winner["macro_p05"],
            "arm_order_tie_break": winner["arm_index"],
        },
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
        label="factorized-native target execution authority",
    )
    authority = dict(raw)
    required = {
        "schema",
        "schema_version",
        "status",
        "source_arm_results",
        "implementation",
        "implementation_dependencies",
        "target_inputs",
        "target_descriptor_output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != TARGET_EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_three_arm_source_selection_for_query_free_target"
        or authority.get("materialization_authorized") is not True
        or authority.get("query_execution_authorized") is not False
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != target_descriptor_access_audit()
    ):
        raise ValueError("factorized-native target execution header differs")

    # Ordering is intentional: source selection completes before target records
    # or even the target materializer implementation are opened.
    source_gate = validate_source_arm_winner(authority["source_arm_results"])

    implementation = validate_file_record(
        authority["implementation"], label="factorized-native target implementation"
    )
    if implementation != TARGET_IMPLEMENTATION_PATH:
        raise ValueError("factorized-native target implementation differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        TARGET_IMPLEMENTATION_DEPENDENCIES
    ):
        raise ValueError("factorized-native target dependencies differ")
    verified_dependencies: dict[str, dict[str, str]] = {}
    for name, expected in TARGET_IMPLEMENTATION_DEPENDENCIES.items():
        verified = validate_file_record(
            dependencies[name], label=f"factorized-native dependency {name}"
        )
        if verified != expected:
            raise ValueError(f"factorized-native dependency differs: {name}")
        verified_dependencies[name] = _record(
            dependencies[name], label=f"factorized-native dependency {name}"
        )
    target_inputs = authority.get("target_inputs")
    if not isinstance(target_inputs, Mapping) or set(target_inputs) != set(
        TARGET_INPUT_NAMES
    ):
        raise ValueError("factorized-native target inputs differ")
    verified_inputs: dict[str, dict[str, str]] = {}
    for name in TARGET_INPUT_NAMES:
        shaped = _record(target_inputs[name], label=f"factorized-native target {name}")
        verified = validate_file_record(shaped, label=f"factorized-native target {name}")
        if str(verified) != shaped["path"]:
            raise ValueError(f"factorized-native target {name} is not canonical")
        verified_inputs[name] = shaped
    output = _canonical_output(authority["target_descriptor_output"])
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("factorized-native target output differs")
    authority["source_arm_results"] = {
        arm: _record(authority["source_arm_results"][arm], label=f"{arm} result")
        for arm in FACTORIZED_NATIVE_READOUT_ARMS
    }
    authority["implementation"] = _record(
        authority["implementation"], label="factorized-native target implementation"
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
    if not isinstance(value, Mapping):
        raise ValueError("factorized-native target descriptor must be a mapping")
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
        "winner_arm",
        "winner_selected_step",
        "winner_source_ranking",
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
    if (
        set(payload) != required
        or payload.get("schema") != TARGET_DESCRIPTOR_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("contract") != target_descriptor_contract()
        or payload.get("contract_sha256") != TARGET_DESCRIPTOR_CONTRACT_SHA256
        or payload.get("access_audit") != target_descriptor_access_audit()
        or payload.get("winner_arm") not in FACTORIZED_NATIVE_READOUT_ARMS
        or not isinstance(payload.get("winner_selected_step"), int)
        or payload.get("fallback_bitwise_equal") is not True
    ):
        raise ValueError("factorized-native target descriptor contract differs")
    physical = payload.get("physical_space_authority")
    if not isinstance(physical, Mapping):
        raise ValueError("factorized-native target physical-space authority differs")
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
        raise ValueError("factorized-native target physical-space binding differs")
    payload["physical_space_authority"] = expected_physical
    payload["producer"] = _record(payload["producer"], label="descriptor producer")
    payload["target_execution_authority"] = _record(
        payload["target_execution_authority"], label="target execution authority"
    )
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != set(DESCRIPTOR_INPUT_NAMES):
        raise ValueError("factorized-native descriptor input records differ")
    arm_results = inputs.get("source_arm_results")
    if not isinstance(arm_results, Mapping) or tuple(arm_results.keys()) != tuple(
        FACTORIZED_NATIVE_READOUT_ARMS
    ):
        raise ValueError("factorized-native descriptor arm records differ")
    payload["input_authority"] = {
        "source_arm_results": {
            arm: _record(arm_results[arm], label=f"descriptor {arm} result")
            for arm in FACTORIZED_NATIVE_READOUT_ARMS
        },
        **{
            name: _record(inputs[name], label=f"descriptor {name}")
            for name in DESCRIPTOR_INPUT_NAMES
            if name != "source_arm_results"
        },
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
            raise ValueError(f"factorized-native target {name} differs")
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
        raise ValueError("factorized-native target tensor layout differs")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("factorized-native target descriptor is not unit L2")
    exact = masks["exact_state_anchor_mask"]
    active = masks["active_update_mask"]
    fallback = masks["immutable_fallback_mask"]
    changed = masks["descriptor_changed_mask"]
    if (
        not torch.equal(active, exact)
        or not torch.equal(fallback, ~active)
        or bool((changed & fallback).any())
    ):
        raise ValueError("factorized-native target routing masks differ")
    expected_audit = {
        "regions": regions,
        "exact_state_anchor": int(exact.sum()),
        "active_update": int(active.sum()),
        "immutable_fallback": int(fallback.sum()),
        "descriptor_changed": int(changed.sum()),
    }
    if payload.get("routing_audit") != expected_audit:
        raise ValueError("factorized-native target routing audit differs")
    ranking = payload.get("winner_source_ranking")
    if not isinstance(ranking, Mapping) or set(ranking) != {
        "macro_mean_all_view_cosine",
        "macro_p05_row_mean_all_view_cosine",
        "arm_order_tie_break",
    }:
        raise ValueError("factorized-native source ranking audit differs")
    if payload.get("channel_sha256") != target_descriptor_channel_sha256(payload):
        raise ValueError("factorized-native target channel SHA-256 differs")
    return payload


def exact_query_descriptor_view(value: object) -> dict[str, Any]:
    """Return the schema-neutral channels consumed by exact query scoring."""

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
    "validate_source_arm_winner",
    "validate_target_descriptor_authority",
    "validate_target_execution_authority",
]
