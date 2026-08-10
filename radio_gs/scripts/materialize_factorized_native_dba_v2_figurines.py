#!/usr/bin/env python3
"""Materialize and structurally gate the DBA-v2 Figurines exact-query chain."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.querying.v21_absolute_relevance_adapter import (
    calibrated_v21_absolute_relevance,
)
from radio_gs.scripts import (
    audit_factorized_native_target_descriptor_health_v2 as health_stats,
)
from radio_gs.scripts import (
    build_lerf_o0_anchored_graph_residual_cache as exact_o0,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance as legacy_query,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_target_descriptor as legacy_target,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 as contrast_v2,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v1 as dba_v1,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v2 as dba_v2,
)
from radio_gs.scripts.materialize_factorized_native_target_descriptor import (
    apply_factorized_native_canonical_forward,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


DESCRIPTOR_AUTHORITY_SCHEMA = "radio_gs.factorized_native_dba_v2_target_execution.v1"
DESCRIPTOR_SCHEMA = "radio_gs.factorized_native_dba_v2_target_descriptor.v1"
QUERY_AUTHORITY_SCHEMA = "radio_gs.factorized_native_dba_v2_exact_query_execution.v1"
RELEVANCE_SCHEMA = "radio_gs.factorized_native_dba_v2_exact_relevance.v1"
AUDIT_SCHEMA = "radio_gs.factorized_native_dba_v2_premetric_structural_audit.v1"
SCHEMA_VERSION = 1
SCENE_ID = "figurines"
EXPECTED_SOURCE_SELECTED_STEP = 40
EXPECTED_O0_SCALE_IDS = ["0.25", "0.45", "0.7"]
EXPECTED_O0_SCALE_RADII_M = [0.25, 0.45, 0.7]
CANDIDATE_BOUNDARY = 0.5
O0_STRONG_BOUNDARY = 0.6
MINIMUM_REGIONS_PER_SUPPORTED_QUERY = 3
MINIMUM_O0_QUERY_RECALL = 0.8
MAXIMUM_POSITIVE_UNIT_RATE = 0.02
MINIMUM_COVERAGE_GAIN_OVER_V21 = 5
MAXIMUM_CENTROID_SQUARED_NORM_INCREASE = 0.02
MINIMUM_SPREAD_RATIO_TO_V21 = 0.75
MINIMUM_EFFECTIVE_RANK_RATIO_TO_V21 = 0.75
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def access_audit(*, query_opened: bool) -> dict[str, bool]:
    return {
        "dba_v2_source_promotion_validated": True,
        "target_geometry_authority_opened": True,
        "exact_query_protocol_opened": query_opened,
        "o0_query_scores_opened": query_opened,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_metrics_opened": False,
        "target_metrics_computed": False,
    }


def structural_contract() -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "candidate_boundary": CANDIDATE_BOUNDARY,
        "o0_strong_boundary": O0_STRONG_BOUNDARY,
        "o0_readout": (
            "exact_frozen_canonical_negative_probability_then_knn10_"
            "raw_peak_scale_then_independent_per_scale_minmax_clip_2x_minus_1"
        ),
        "query_supported_definition": (
            f"at_least_{MINIMUM_REGIONS_PER_SUPPORTED_QUERY}_units_strictly_above_boundary"
        ),
        "checks": {
            "candidate_supported_queries_at_least_o0_strong_supported_queries": True,
            "o0_strong_query_recall_at_least": MINIMUM_O0_QUERY_RECALL,
            "candidate_coverage_gain_over_contrast_v21_at_least": (
                MINIMUM_COVERAGE_GAIN_OVER_V21
            ),
            "candidate_positive_unit_rate_at_most": MAXIMUM_POSITIVE_UNIT_RATE,
            "descriptor_centroid_squared_norm_increase_at_most": (
                MAXIMUM_CENTROID_SQUARED_NORM_INCREASE
            ),
            "descriptor_spread_ratio_to_contrast_v21_at_least": (
                MINIMUM_SPREAD_RATIO_TO_V21
            ),
            "descriptor_effective_rank_ratio_to_contrast_v21_at_least": (
                MINIMUM_EFFECTIVE_RANK_RATIO_TO_V21
            ),
            "all_required": True,
        },
        "threshold_origin": (
            "method_level_frozen_before_dba_v2_target_descriptor_or_relevance_materialization"
        ),
        "metric_execution_authorized": False,
    }


STRUCTURAL_CONTRACT_SHA256 = canonical_json_sha256(structural_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def _argument_record(path: str, digest: str, *, label: str) -> dict[str, str]:
    record = {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": str(digest),
    }
    validate_file_record(record, label=label)
    return record


def _new(path: str, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def validate_source_result(record_value: object) -> dict[str, Any]:
    record = _record(record_value, label="DBA-v2 source result")
    raw, digest, source = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="DBA-v2 source result",
    )
    required = {
        "arm",
        "benchmark_opened",
        "checkpoint",
        "execution_authority",
        "history",
        "input_authority",
        "last_training_step",
        "schema",
        "schema_version",
        "selected_step",
        "source_access",
        "status",
        "target_query_or_metric_authorized",
        "training_contract",
        "training_contract_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("DBA-v2 source result fields differ")
    result = dict(raw)
    history = result.get("history")
    selected = result.get("selected_step")
    expected_steps = list(range(0, dba_v2.OPTIMIZER_STEPS + 1, dba_v2.EVALUATION_INTERVAL))
    _validate_selected_source_history(history, selected)
    if (
        result.get("schema") != dba_v2.RESULT_SCHEMA
        or result.get("schema_version") != dba_v2.SCHEMA_VERSION
        or result.get("status") != "source_only_dba_v2_promotion_candidate_complete"
        or result.get("arm") != DIRECTION_ONLY
        or result.get("training_contract") != dba_v2.training_contract()
        or result.get("training_contract_sha256") != dba_v2.TRAINING_CONTRACT_SHA256
        or result.get("benchmark_opened") is not False
        or result.get("target_query_or_metric_authorized") is not False
        or result.get("source_access") != dba_v2.source_access()
        or result.get("last_training_step", {}).get("step") != dba_v2.OPTIMIZER_STEPS
    ):
        raise ValueError("DBA-v2 source promotion contract differs")
    execution_record = _record(
        result["execution_authority"], label="DBA-v2 execution authority"
    )
    prepared = dba_v2.prepare_inputs(
        execution_record["path"], expected_sha256=execution_record["sha256"]
    )
    checkpoint_record = _record(result["checkpoint"], label="DBA-v2 checkpoint")
    checkpoint_raw, checkpoint_digest, checkpoint_path = load_torch_mapping(
        checkpoint_record["path"],
        expected_sha256=checkpoint_record["sha256"],
        map_location="cpu",
        label="DBA-v2 checkpoint",
    )
    checkpoint = dict(checkpoint_raw)
    state = checkpoint.get("model_state_dict")
    model = readout.build_model(
        DIRECTION_ONLY, prepared.base.source_gate["normalization"]
    )
    if (
        checkpoint.get("schema") != dba_v2.CHECKPOINT_SCHEMA
        or checkpoint.get("schema_version") != dba_v2.SCHEMA_VERSION
        or checkpoint.get("training_contract") != dba_v2.training_contract()
        or checkpoint.get("training_contract_sha256")
        != dba_v2.TRAINING_CONTRACT_SHA256
        or checkpoint.get("interface_contract_sha256")
        != readout.INTERFACE_CONTRACT_SHA256
        or checkpoint.get("model_architecture")
        != model.architecture(readout.INTERFACE_CONTRACT_SHA256)
        or checkpoint.get("selected_step") != selected
        or checkpoint.get("execution_authority") != execution_record
        or checkpoint.get("source_access") != dba_v2.source_access()
        or not isinstance(state, Mapping)
        or checkpoint.get("model_state_dict_sha256") != contrast_v2._state_sha(state)
        or checkpoint.get("model_state_dict_sha256")
        != history[expected_steps.index(selected)]["model_state_dict_sha256"]
    ):
        raise ValueError("DBA-v2 checkpoint contract differs")
    model.load_state_dict(state, strict=True)
    checkpoint["verified_record"] = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_digest,
    }
    result["verified_record"] = {"path": str(source), "sha256": digest}
    result["execution_authority"] = execution_record
    result["checkpoint"] = checkpoint_record
    return {
        "result": result,
        "checkpoint": checkpoint,
        "prepared": prepared,
        "source_gate": prepared.base.source_gate,
        "source_only_passed": True,
    }


def _validate_selected_source_history(history: object, selected: object) -> None:
    """Fail closed unless the promoted source is the frozen eligible step 40."""

    expected_steps = list(
        range(0, dba_v2.OPTIMIZER_STEPS + 1, dba_v2.EVALUATION_INTERVAL)
    )
    if (
        not isinstance(history, list)
        or [row.get("step") for row in history] != expected_steps
        or not isinstance(selected, int)
        or selected != EXPECTED_SOURCE_SELECTED_STEP
        or dba_v1.select_step(history) != selected
        or history[expected_steps.index(selected)]["validation"]["selection"]["eligible"]
        is not True
    ):
        raise ValueError("DBA-v2 selected source history differs")


def build_descriptor_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.authority_output, label="DBA-v2 descriptor authority")
    descriptor_output = _new(args.descriptor_output, label="DBA-v2 descriptor output")
    source_record = _argument_record(
        args.source_result, args.source_result_sha256, label="DBA-v2 source result"
    )
    source = validate_source_result(source_record)
    legacy_record = _argument_record(
        args.legacy_target_authority,
        args.legacy_target_authority_sha256,
        label="legacy query-free target authority",
    )
    legacy = legacy_target.formal.validate_target_execution_authority(
        legacy_record["path"], expected_sha256=legacy_record["sha256"]
    )
    if (
        legacy["verified_source_gate"]["result"]["verified_record"]
        != source["checkpoint"]["warm_start_source_contrast_v21_result"]
        or legacy["target_inputs"].keys()
        != {"target_accepted_v2", "factorized_primitive_state"}
    ):
        raise ValueError("DBA-v2 target lineage differs")
    authority = {
        "schema": DESCRIPTOR_AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_query_free_after_dba_v2_source_promotion",
        "implementation": file_record(Path(__file__).resolve()),
        "source_result": source_record,
        "legacy_query_free_target_authority": legacy_record,
        "descriptor_output": str(descriptor_output),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": access_audit(query_opened=False),
    }
    write_frozen_json(output, authority)
    return {"authority": file_record(output), "descriptor_output": str(descriptor_output)}


def validate_descriptor_authority(
    path: str | Path, *, expected_sha256: str, expected_output: str | Path | None = None
) -> dict[str, Any]:
    raw, digest, source_path = load_json_object(
        path, expected_sha256=expected_sha256, label="DBA-v2 descriptor authority"
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "source_result",
        "legacy_query_free_target_authority",
        "descriptor_output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("DBA-v2 descriptor authority fields differ")
    authority = dict(raw)
    if (
        authority["schema"] != DESCRIPTOR_AUTHORITY_SCHEMA
        or authority["schema_version"] != SCHEMA_VERSION
        or authority["status"] != "authorized_query_free_after_dba_v2_source_promotion"
        or authority["materialization_authorized"] is not True
        or authority["query_execution_authorized"] is not False
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != access_audit(query_opened=False)
        or validate_file_record(authority["implementation"], label="DBA-v2 target implementation")
        != Path(__file__).resolve()
    ):
        raise ValueError("DBA-v2 descriptor authority header differs")
    source = validate_source_result(authority["source_result"])
    legacy_record = _record(
        authority["legacy_query_free_target_authority"],
        label="legacy query-free target authority",
    )
    legacy = legacy_target.formal.validate_target_execution_authority(
        legacy_record["path"], expected_sha256=legacy_record["sha256"]
    )
    output = str(Path(authority["descriptor_output"]).resolve())
    if expected_output is not None and output != str(Path(expected_output).resolve()):
        raise ValueError("DBA-v2 descriptor output differs")
    authority.update(
        {
            "verified_record": {"path": str(source_path), "sha256": digest},
            "verified_source": source,
            "verified_legacy": legacy,
            "legacy_query_free_target_authority": legacy_record,
            "descriptor_output": output,
        }
    )
    return authority


def _descriptor_channel(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "schema": payload["schema"],
            "scene_id": payload["scene_id"],
            "physical_space_id": payload["physical_space_id"],
            "source_selected_step": payload["source_selected_step"],
            "region_row_ids": payload["region_row_ids"],
            "region_fingerprints": payload["region_fingerprints"],
            "canonical_region_indices_sha256": tensor_sha256(
                payload["canonical_region_indices"]
            ),
            "semantic_descriptor_sha256": tensor_sha256(
                payload["semantic_descriptor"]
            ),
        }
    )


def materialize_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="DBA-v2 target descriptor")
    execution = validate_descriptor_authority(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
        expected_output=output,
    )
    target = legacy_target._load_target_inputs(execution["verified_legacy"])
    accepted, state = target["accepted"], target["state"]
    source = execution["verified_source"]
    model = readout.build_model(DIRECTION_ONLY, source["source_gate"]["normalization"])
    model.load_state_dict(source["checkpoint"]["model_state_dict"], strict=True)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        source["source_gate"]["official_radio_checkpoint"]["path"],
        expected_sha256=source["source_gate"]["official_radio_checkpoint"]["sha256"],
    )
    forward = apply_factorized_native_canonical_forward(
        accepted_v2_e0=accepted["accepted_v2_e0"],
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        anchor_index=accepted["anchor_index"],
        state=state,
        model=model,
        head=head,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    region_ids = [
        shard.stable_region_id(accepted["scene_id"], fingerprint)
        for fingerprint in accepted["region_fingerprints"]
    ]
    payload = {
        "schema": DESCRIPTOR_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_result": dict(execution["verified_source"]["result"]["verified_record"]),
            "source_checkpoint": dict(execution["verified_source"]["checkpoint"]["verified_record"]),
            "legacy_query_free_target_authority": dict(
                execution["legacy_query_free_target_authority"]
            ),
            **dict(target["records"]),
        },
        "source_arm": DIRECTION_ONLY,
        "source_selected_step": execution["verified_source"]["result"]["selected_step"],
        "region_row_ids": region_ids,
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": forward["semantic_descriptor"],
        "exact_state_anchor_mask": forward["exact_state_anchor_mask"],
        "active_update_mask": forward["active_update_mask"],
        "immutable_fallback_mask": forward["immutable_fallback_mask"],
        "descriptor_changed_mask": forward["descriptor_changed_mask"],
        "fallback_bitwise_equal": forward["fallback_bitwise_equal"],
        "access_audit": access_audit(query_opened=False),
    }
    if payload["scene_id"] != SCENE_ID or payload["semantic_descriptor"].shape != (4096, 1536):
        raise ValueError("DBA-v2 Figurines target descriptor axes differ")
    payload["channel_sha256"] = _descriptor_channel(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "DBA-v2 query-free target descriptor complete",
        "shape": list(payload["semantic_descriptor"].shape),
        "source_selected_step": payload["source_selected_step"],
        "output": file_record(output),
    }


def _load_descriptor(record: Mapping[str, str]) -> dict[str, Any]:
    raw, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="DBA-v2 target descriptor",
    )
    required = {
        "schema", "schema_version", "scene_id", "physical_space_id", "producer",
        "target_execution_authority", "input_authority", "source_arm",
        "source_selected_step", "region_row_ids", "canonical_region_indices",
        "region_fingerprints", "semantic_descriptor", "exact_state_anchor_mask",
        "active_update_mask", "immutable_fallback_mask", "descriptor_changed_mask",
        "fallback_bitwise_equal", "access_audit", "channel_sha256",
    }
    payload = dict(raw)
    descriptor = payload.get("semantic_descriptor")
    if (
        set(payload) != required
        or payload.get("schema") != DESCRIPTOR_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("scene_id") != SCENE_ID
        or payload.get("source_arm") != DIRECTION_ONLY
        or not torch.is_tensor(descriptor)
        or descriptor.shape != (4096, 1536)
        or descriptor.dtype != torch.float32
        or payload.get("channel_sha256") != _descriptor_channel(payload)
        or payload.get("access_audit") != access_audit(query_opened=False)
    ):
        raise ValueError("DBA-v2 target descriptor contract differs")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0):
        raise ValueError("DBA-v2 target descriptors are not unit L2")
    payload["verified_record"] = {"path": str(source), "sha256": digest}
    return payload


def build_query_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.authority_output, label="DBA-v2 query authority")
    relevance_output = _new(args.relevance_output, label="DBA-v2 relevance output")
    audit_output = _new(args.audit_output, label="DBA-v2 structural audit output")
    source_record = _argument_record(
        args.source_result, args.source_result_sha256, label="DBA-v2 source result"
    )
    source = validate_source_result(source_record)
    descriptor_record = _argument_record(
        args.target_descriptor,
        args.target_descriptor_sha256,
        label="DBA-v2 target descriptor",
    )
    descriptor = _load_descriptor(descriptor_record)
    descriptor_execution_record = _record(
        descriptor["target_execution_authority"],
        label="DBA-v2 descriptor execution authority",
    )
    descriptor_execution = validate_descriptor_authority(
        descriptor_execution_record["path"],
        expected_sha256=descriptor_execution_record["sha256"],
        expected_output=descriptor_record["path"],
    )
    legacy_query_record = _argument_record(
        args.legacy_query_authority,
        args.legacy_query_authority_sha256,
        label="legacy exact query authority",
    )
    legacy = legacy_query.validate_authority(
        legacy_query_record["path"], expected_sha256=legacy_query_record["sha256"]
    )
    old_relevance_record = _argument_record(
        args.legacy_v21_relevance,
        args.legacy_v21_relevance_sha256,
        label="legacy contrast-v21 relevance",
    )
    o0_positive = _argument_record(
        args.o0_positive, args.o0_positive_sha256, label="O0 positive query scores"
    )
    o0_negative = _argument_record(
        args.o0_negative, args.o0_negative_sha256, label="O0 negative query scores"
    )
    old_raw, _, _ = load_torch_mapping(
        old_relevance_record["path"],
        expected_sha256=old_relevance_record["sha256"],
        map_location="cpu",
        label="legacy contrast-v21 relevance",
    )
    old = legacy_query.formal.validate_query_relevance(old_raw)
    o0_positive_raw, _, _ = load_torch_mapping(
        o0_positive["path"],
        expected_sha256=o0_positive["sha256"],
        map_location="cpu",
        label="O0 positive query scores",
    )
    o0_negative_raw, _, _ = load_torch_mapping(
        o0_negative["path"],
        expected_sha256=o0_negative["sha256"],
        map_location="cpu",
        label="O0 negative query scores",
    )
    _validate_o0_pair(
        o0_positive_raw,
        o0_negative_raw,
        query_ids=list(legacy["verified_positive"].query_ids),
        physical_space_id=descriptor["physical_space_id"],
    )
    source_result_record = source["result"]["verified_record"]
    _validate_descriptor_source_binding(descriptor, descriptor_execution, source)
    _validate_legacy_query_axes(
        old, descriptor, list(legacy["verified_positive"].query_ids)
    )
    if (
        descriptor["physical_space_id"] != legacy["physical_space_id"]
        or descriptor["scene_id"] != legacy["scene_id"]
        or descriptor_execution["verified_source"]["result"]["verified_record"]
        != source_result_record
    ):
        raise ValueError("DBA-v2 descriptor/query identity differs")
    authority = {
        "schema": QUERY_AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_exact_query_for_premetric_structural_audit_only",
        "implementation": file_record(Path(__file__).resolve()),
        "structural_contract": structural_contract(),
        "structural_contract_sha256": STRUCTURAL_CONTRACT_SHA256,
        "source_result": source_record,
        "target_descriptor": descriptor_record,
        "legacy_exact_query_authority": legacy_query_record,
        "legacy_contrast_v21_relevance": old_relevance_record,
        "o0_positive_query_scores": o0_positive,
        "o0_negative_query_scores": o0_negative,
        "relevance_output": str(relevance_output),
        "audit_output": str(audit_output),
        "query_execution_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": access_audit(query_opened=True),
    }
    write_frozen_json(output, authority)
    return {"authority": file_record(output)}


def validate_query_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256, label="DBA-v2 query authority"
    )
    required = {
        "schema", "schema_version", "status", "implementation",
        "structural_contract", "structural_contract_sha256", "source_result",
        "target_descriptor", "legacy_exact_query_authority",
        "legacy_contrast_v21_relevance", "o0_positive_query_scores",
        "o0_negative_query_scores", "relevance_output", "audit_output",
        "query_execution_authorized", "metric_execution_authorized", "access_audit",
    }
    authority = dict(raw)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or authority.get("schema") != QUERY_AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_exact_query_for_premetric_structural_audit_only"
        or authority.get("structural_contract") != structural_contract()
        or authority.get("structural_contract_sha256") != STRUCTURAL_CONTRACT_SHA256
        or authority.get("query_execution_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != access_audit(query_opened=True)
        or validate_file_record(authority["implementation"], label="DBA-v2 query implementation")
        != Path(__file__).resolve()
    ):
        raise ValueError("DBA-v2 query authority header differs")
    source_gate = validate_source_result(authority["source_result"])
    descriptor_record = _record(authority["target_descriptor"], label="DBA-v2 target descriptor")
    descriptor = _load_descriptor(descriptor_record)
    descriptor_execution_record = _record(
        descriptor["target_execution_authority"],
        label="DBA-v2 descriptor execution authority",
    )
    descriptor_execution = validate_descriptor_authority(
        descriptor_execution_record["path"],
        expected_sha256=descriptor_execution_record["sha256"],
        expected_output=descriptor_record["path"],
    )
    legacy_record = _record(
        authority["legacy_exact_query_authority"], label="legacy exact query authority"
    )
    legacy = legacy_query.validate_authority(
        legacy_record["path"], expected_sha256=legacy_record["sha256"]
    )
    records = {}
    for name in (
        "legacy_contrast_v21_relevance",
        "o0_positive_query_scores",
        "o0_negative_query_scores",
    ):
        records[name] = _record(authority[name], label=f"DBA-v2 {name}")
        validate_file_record(records[name], label=f"DBA-v2 {name}")
    old_raw, _, _ = load_torch_mapping(
        records["legacy_contrast_v21_relevance"]["path"],
        expected_sha256=records["legacy_contrast_v21_relevance"]["sha256"],
        map_location="cpu",
        label="legacy contrast-v21 relevance",
    )
    old = legacy_query.formal.validate_query_relevance(old_raw)
    o0_positive_raw, _, _ = load_torch_mapping(
        records["o0_positive_query_scores"]["path"],
        expected_sha256=records["o0_positive_query_scores"]["sha256"],
        map_location="cpu",
        label="O0 positive query scores",
    )
    o0_negative_raw, _, _ = load_torch_mapping(
        records["o0_negative_query_scores"]["path"],
        expected_sha256=records["o0_negative_query_scores"]["sha256"],
        map_location="cpu",
        label="O0 negative query scores",
    )
    _validate_o0_pair(
        o0_positive_raw,
        o0_negative_raw,
        query_ids=list(legacy["verified_positive"].query_ids),
        physical_space_id=descriptor["physical_space_id"],
    )
    _validate_descriptor_source_binding(descriptor, descriptor_execution, source_gate)
    _validate_legacy_query_axes(
        old, descriptor, list(legacy["verified_positive"].query_ids)
    )
    if (
        descriptor["physical_space_id"] != legacy["physical_space_id"]
        or descriptor["scene_id"] != legacy["scene_id"]
    ):
        raise ValueError("DBA-v2 exact query full-axis lineage differs")
    authority.update(
        {
            "verified_record": {"path": str(source), "sha256": digest},
            "verified_source": source_gate,
            "verified_descriptor": descriptor,
            "verified_legacy_query": legacy,
            **records,
        }
    )
    return authority


def _validate_o0_pair(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
    *,
    query_ids: list[str],
    physical_space_id: str,
) -> None:
    positive_scores = positive.get("query_scores")
    negative_scores = negative.get("query_scores")
    positive_xyz = positive.get("xyz")
    negative_xyz = negative.get("xyz")
    positive_valid = positive.get("valid")
    negative_valid = negative.get("valid")
    renderer = str(positive.get("renderer_geometry_checkpoint_sha256"))
    if (
        positive.get("version") != 4
        or negative.get("version") != 4
        or positive.get("contract")
        != "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
        or negative.get("contract") != positive.get("contract")
        or not torch.is_tensor(positive_scores)
        or not torch.is_tensor(negative_scores)
        or not torch.is_tensor(positive_xyz)
        or not torch.is_tensor(negative_xyz)
        or not torch.is_tensor(positive_valid)
        or not torch.is_tensor(negative_valid)
        or positive_scores.ndim != 3
        or negative_scores.shape != (*positive_scores.shape[:2], 4)
        or positive_scores.shape[-1] != len(query_ids)
        or positive_scores.dtype != torch.float32
        or negative_scores.dtype != torch.float32
        or not bool(torch.isfinite(positive_scores).all())
        or not bool(torch.isfinite(negative_scores).all())
        or list(positive.get("query_ids", [])) != query_ids
        or tuple(negative.get("query_ids", []))
        != tuple(exact_o0.v2.frozen.NEGATIVE_PROMPTS)
        or positive.get("scale_ids") != negative.get("scale_ids")
        or positive.get("scale_radii_m") != negative.get("scale_radii_m")
        or positive.get("scale_ids") != EXPECTED_O0_SCALE_IDS
        or positive.get("scale_radii_m") != EXPECTED_O0_SCALE_RADII_M
        or not isinstance(positive.get("scale_ids"), list)
        or len(positive.get("scale_ids")) != positive_scores.shape[1]
        or len(set(positive.get("scale_ids"))) != positive_scores.shape[1]
        or not isinstance(positive.get("scale_radii_m"), list)
        or len(positive.get("scale_radii_m")) != positive_scores.shape[1]
        or not all(
            isinstance(radius, (int, float))
            and bool(torch.isfinite(torch.tensor(radius)))
            for radius in positive.get("scale_radii_m")
        )
        or positive.get("geometry_fingerprint")
        != negative.get("geometry_fingerprint")
        or positive.get("field_checkpoint_sha256")
        != negative.get("field_checkpoint_sha256")
        or positive.get("readout_checkpoint_sha256")
        != negative.get("readout_checkpoint_sha256")
        or renderer != str(negative.get("renderer_geometry_checkpoint_sha256"))
        or physical_space_id
        != f"lerf:{SCENE_ID}:geometry-checkpoint-sha256:{renderer}"
        or not torch.equal(positive_xyz, negative_xyz)
        or not torch.equal(positive_valid, negative_valid)
        or positive_xyz.shape != (positive_scores.shape[0], 3)
        or positive_xyz.dtype != torch.float32
        or not bool(torch.isfinite(positive_xyz).all())
        or positive_valid.shape != (positive_scores.shape[0],)
        or positive_valid.dtype != torch.bool
        or not bool(positive_valid.any())
    ):
        raise ValueError("exact O0 positive/negative/physical-space lineage differs")


def _o0_relevance(positive: Mapping[str, Any], negative: Mapping[str, Any]) -> torch.Tensor:
    """Return the evaluator-exact KNN10/peak-scale/minmax-clipped O0 readout."""

    if (
        positive.get("contract") != "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
        or negative.get("contract") != positive.get("contract")
    ):
        raise ValueError("O0 query score contract differs")
    return exact_o0.exact_o0_readout(
        positive_scores=positive["query_scores"],
        negative_scores=negative["query_scores"],
        xyz=positive["xyz"],
        valid=positive["valid"],
        chunk_size=65536,
    ).final_scores


def _validate_descriptor_source_binding(
    descriptor: Mapping[str, Any],
    descriptor_execution: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    source_result = source["result"]["verified_record"]
    source_checkpoint = source["checkpoint"]["verified_record"]
    inputs = descriptor.get("input_authority")
    if (
        descriptor.get("source_selected_step") != EXPECTED_SOURCE_SELECTED_STEP
        or not isinstance(inputs, Mapping)
        or inputs.get("source_result") != source_result
        or inputs.get("source_checkpoint") != source_checkpoint
        or descriptor_execution["verified_source"]["result"]["verified_record"]
        != source_result
        or descriptor_execution["verified_source"]["checkpoint"]["verified_record"]
        != source_checkpoint
    ):
        raise ValueError("DBA-v2 descriptor source binding differs")


def _validate_legacy_query_axes(
    old: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    query_ids: list[str],
) -> None:
    relevance = old.get("region_absolute_relevance")
    if (
        old.get("scene_id") != descriptor.get("scene_id")
        or old.get("physical_space_id") != descriptor.get("physical_space_id")
        or list(old.get("region_row_ids", []))
        != list(descriptor.get("region_row_ids", []))
        or not torch.is_tensor(old.get("canonical_region_indices"))
        or not torch.is_tensor(descriptor.get("canonical_region_indices"))
        or not torch.equal(
            old["canonical_region_indices"], descriptor["canonical_region_indices"]
        )
        or list(old.get("region_fingerprints", []))
        != list(descriptor.get("region_fingerprints", []))
        or list(old.get("query_ids", [])) != query_ids
        or not torch.is_tensor(relevance)
        or relevance.shape != (4096, len(query_ids))
        or relevance.dtype != torch.float32
        or relevance.device.type != "cpu"
        or not bool(torch.isfinite(relevance).all())
    ):
        raise ValueError("DBA-v2 legacy relevance full axes differ")


def _audit_decision(checks: Mapping[str, object]) -> tuple[str, bool]:
    if not checks or any(type(value) is not bool for value in checks.values()):
        raise ValueError("DBA-v2 structural checks must be non-empty booleans")
    passed = all(checks.values())
    return ("PASS" if passed else "REJECT"), passed


def materialize_relevance_and_audit(args: argparse.Namespace) -> dict[str, Any]:
    execution = validate_query_authority(
        args.execution_authority, expected_sha256=args.execution_authority_sha256
    )
    relevance_output = _new(execution["relevance_output"], label="DBA-v2 relevance output")
    audit_output = _new(execution["audit_output"], label="DBA-v2 audit output")
    descriptor = execution["verified_descriptor"]
    legacy = execution["verified_legacy_query"]
    relevance = calibrated_v21_absolute_relevance(
        descriptor["semantic_descriptor"],
        positive_bank=legacy["verified_positive"],
        canonical_negative_bank=legacy["verified_negative"],
    ).detach().cpu().float().contiguous()
    query_ids = list(legacy["verified_positive"].query_ids)
    payload = {
        "schema": RELEVANCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(Path(__file__).resolve()),
        "query_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_result": dict(execution["verified_source"]["result"]["verified_record"]),
            "target_descriptor": dict(descriptor["verified_record"]),
            "legacy_exact_query_authority": dict(
                execution["legacy_exact_query_authority"]
            ),
        },
        "region_row_ids": list(descriptor["region_row_ids"]),
        "canonical_region_indices": descriptor["canonical_region_indices"].clone(),
        "region_fingerprints": list(descriptor["region_fingerprints"]),
        "query_ids": query_ids,
        "region_absolute_relevance": relevance,
        "access_audit": access_audit(query_opened=True),
        "tensor_sha256": tensor_sha256(relevance),
    }
    write_torch_noclobber(relevance_output, payload)

    old_raw, _, _ = load_torch_mapping(
        execution["legacy_contrast_v21_relevance"]["path"],
        expected_sha256=execution["legacy_contrast_v21_relevance"]["sha256"],
        map_location="cpu",
        label="legacy contrast-v21 relevance",
    )
    old_payload = legacy_query.formal.validate_query_relevance(old_raw)
    old = old_payload["region_absolute_relevance"]
    o0_positive, _, _ = load_torch_mapping(
        execution["o0_positive_query_scores"]["path"],
        expected_sha256=execution["o0_positive_query_scores"]["sha256"],
        map_location="cpu",
        label="O0 positive query scores",
    )
    o0_negative, _, _ = load_torch_mapping(
        execution["o0_negative_query_scores"]["path"],
        expected_sha256=execution["o0_negative_query_scores"]["sha256"],
        map_location="cpu",
        label="O0 negative query scores",
    )
    o0 = _o0_relevance(o0_positive, o0_negative)
    _validate_o0_pair(
        o0_positive,
        o0_negative,
        query_ids=query_ids,
        physical_space_id=descriptor["physical_space_id"],
    )
    if (
        query_ids != list(old_payload["query_ids"])
        or query_ids != list(o0_positive["query_ids"])
        or old_payload["scene_id"] != descriptor["scene_id"]
        or old_payload["physical_space_id"] != descriptor["physical_space_id"]
        or list(old_payload["region_row_ids"]) != list(descriptor["region_row_ids"])
        or not torch.equal(
            old_payload["canonical_region_indices"],
            descriptor["canonical_region_indices"],
        )
        or list(old_payload["region_fingerprints"])
        != list(descriptor["region_fingerprints"])
        or old.dtype != torch.float32
        or not bool(torch.isfinite(old).all())
    ):
        raise ValueError("DBA-v2, contrast-v21, and O0 query axes differ")
    candidate_counts = (relevance > CANDIDATE_BOUNDARY).sum(dim=0)
    old_counts = (old > CANDIDATE_BOUNDARY).sum(dim=0)
    valid = o0_positive["valid"][:, None]
    o0_counts = ((o0 > O0_STRONG_BOUNDARY) & valid).sum(dim=0)
    candidate_supported = candidate_counts >= MINIMUM_REGIONS_PER_SUPPORTED_QUERY
    old_supported = old_counts >= MINIMUM_REGIONS_PER_SUPPORTED_QUERY
    o0_supported = o0_counts >= MINIMUM_REGIONS_PER_SUPPORTED_QUERY
    overlap = candidate_supported & o0_supported

    old_descriptor_raw, _, _ = load_torch_mapping(
        legacy["target_descriptor"]["path"],
        expected_sha256=legacy["target_descriptor"]["sha256"],
        map_location="cpu",
        label="legacy contrast-v21 target descriptor",
    )
    candidate_statistics = health_stats.descriptor_statistics(
        descriptor["semantic_descriptor"]
    )
    old_statistics = health_stats.descriptor_statistics(
        old_descriptor_raw["semantic_descriptor"]
    )
    o0_total = int(o0_supported.sum())
    candidate_total = int(candidate_supported.sum())
    old_total = int(old_supported.sum())
    overlap_total = int(overlap.sum())
    checks = {
        "candidate_supported_queries_at_least_o0_strong_supported_queries": (
            candidate_total >= o0_total
        ),
        "o0_strong_query_recall_at_least_0p8": (
            overlap_total / max(o0_total, 1) >= MINIMUM_O0_QUERY_RECALL
        ),
        "candidate_coverage_gain_over_contrast_v21_at_least_5": (
            candidate_total >= old_total + MINIMUM_COVERAGE_GAIN_OVER_V21
        ),
        "candidate_positive_unit_rate_at_most_0p02": (
            float((relevance > CANDIDATE_BOUNDARY).float().mean())
            <= MAXIMUM_POSITIVE_UNIT_RATE
        ),
        "descriptor_centroid_squared_norm_not_regressed": (
            float(candidate_statistics["centroid_squared_norm"])
            <= float(old_statistics["centroid_squared_norm"])
            + MAXIMUM_CENTROID_SQUARED_NORM_INCREASE
        ),
        "descriptor_spread_at_least_0p75_contrast_v21": (
            float(candidate_statistics["centered_mean_squared_radius"])
            >= MINIMUM_SPREAD_RATIO_TO_V21
            * float(old_statistics["centered_mean_squared_radius"])
        ),
        "descriptor_effective_rank_at_least_0p75_contrast_v21": (
            float(candidate_statistics["centered_gram_effective_rank"])
            >= MINIMUM_EFFECTIVE_RANK_RATIO_TO_V21
            * float(old_statistics["centered_gram_effective_rank"])
        ),
    }
    status, passed = _audit_decision(checks)
    audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "structural_contract": structural_contract(),
        "structural_contract_sha256": STRUCTURAL_CONTRACT_SHA256,
        "query_execution_authority": dict(execution["verified_record"]),
        "relevance": file_record(relevance_output),
        "query_ids": query_ids,
        "coverage": {
            "candidate_supported_queries": candidate_total,
            "contrast_v21_supported_queries": old_total,
            "o0_strong_supported_queries": o0_total,
            "candidate_o0_overlap_queries": overlap_total,
            "o0_strong_query_recall": overlap_total / max(o0_total, 1),
            "candidate_positive_unit_rate": float(
                (relevance > CANDIDATE_BOUNDARY).float().mean()
            ),
            "candidate_supported_query_ids": [
                query_ids[i] for i in torch.where(candidate_supported)[0].tolist()
            ],
            "contrast_v21_supported_query_ids": [
                query_ids[i] for i in torch.where(old_supported)[0].tolist()
            ],
            "o0_strong_supported_query_ids": [
                query_ids[i] for i in torch.where(o0_supported)[0].tolist()
            ],
            "candidate_o0_overlap_query_ids": [
                query_ids[i] for i in torch.where(overlap)[0].tolist()
            ],
            "per_query_candidate_region_counts": {
                query_ids[i]: int(candidate_counts[i]) for i in range(len(query_ids))
            },
            "per_query_candidate_max": {
                query_ids[i]: float(relevance[:, i].max()) for i in range(len(query_ids))
            },
        },
        "descriptor_statistics": {
            "candidate": candidate_statistics,
            "contrast_v21": old_statistics,
        },
        "checks": checks,
        "metric_authority_recommended": passed,
        "metric_executed": False,
        "access_audit": access_audit(query_opened=True),
    }
    write_frozen_json(audit_output, audit)
    return {
        "status": audit["status"],
        "relevance": file_record(relevance_output),
        "audit": file_record(audit_output),
        "coverage": audit["coverage"],
        "checks": checks,
        "metric_authority_recommended": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_d = commands.add_parser("build-descriptor-authority")
    build_d.add_argument("--source-result", required=True)
    build_d.add_argument("--source-result-sha256", required=True)
    build_d.add_argument("--legacy-target-authority", required=True)
    build_d.add_argument("--legacy-target-authority-sha256", required=True)
    build_d.add_argument("--descriptor-output", required=True)
    build_d.add_argument("--authority-output", required=True)
    build_d.set_defaults(handler=build_descriptor_authority)
    run_d = commands.add_parser("materialize-descriptor")
    run_d.add_argument("--execution-authority", required=True)
    run_d.add_argument("--execution-authority-sha256", required=True)
    run_d.add_argument("--output", required=True)
    run_d.add_argument("--device", default="cuda:0")
    run_d.add_argument("--batch-size", type=int, default=256)
    run_d.set_defaults(handler=materialize_descriptor)
    build_q = commands.add_parser("build-query-authority")
    build_q.add_argument("--source-result", required=True)
    build_q.add_argument("--source-result-sha256", required=True)
    build_q.add_argument("--target-descriptor", required=True)
    build_q.add_argument("--target-descriptor-sha256", required=True)
    build_q.add_argument("--legacy-query-authority", required=True)
    build_q.add_argument("--legacy-query-authority-sha256", required=True)
    build_q.add_argument("--legacy-v21-relevance", required=True)
    build_q.add_argument("--legacy-v21-relevance-sha256", required=True)
    build_q.add_argument("--o0-positive", required=True)
    build_q.add_argument("--o0-positive-sha256", required=True)
    build_q.add_argument("--o0-negative", required=True)
    build_q.add_argument("--o0-negative-sha256", required=True)
    build_q.add_argument("--relevance-output", required=True)
    build_q.add_argument("--audit-output", required=True)
    build_q.add_argument("--authority-output", required=True)
    build_q.set_defaults(handler=build_query_authority)
    run_q = commands.add_parser("materialize-relevance-and-audit")
    run_q.add_argument("--execution-authority", required=True)
    run_q.add_argument("--execution-authority-sha256", required=True)
    run_q.set_defaults(handler=materialize_relevance_and_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_SCHEMA",
    "CANDIDATE_BOUNDARY",
    "DESCRIPTOR_SCHEMA",
    "MINIMUM_REGIONS_PER_SUPPORTED_QUERY",
    "O0_STRONG_BOUNDARY",
    "STRUCTURAL_CONTRACT_SHA256",
    "access_audit",
    "build_descriptor_authority",
    "build_query_authority",
    "materialize_descriptor",
    "materialize_relevance_and_audit",
    "structural_contract",
    "validate_source_result",
]
