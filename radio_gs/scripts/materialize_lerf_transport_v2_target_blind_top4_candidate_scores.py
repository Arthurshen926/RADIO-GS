#!/usr/bin/env python3
"""Figurines transport-v2 materializer using a validated canonical-top4 adapter."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from radio_gs.scripts import derive_lerf_teacher_agreement_from_top4 as _teacher
from radio_gs.scripts import materialize_lerf_o1_o2_frozen_text_rebind as _rebind
from radio_gs.scripts import materialize_lerf_transport_v2_target_blind_candidate_scores as _base
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_transport_v2_target_blind_top4_execution.v1"
RESULT_SCHEMA = "radio_gs.lerf_transport_v2_target_blind_top4_result.v1"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def method_contract() -> dict[str, Any]:
    value = copy.deepcopy(_base.method_contract())
    value.update(
        {
            "schema": AUTHORITY_SCHEMA,
            "teacher_payload": "validated_canonical_top4_deterministic_adapter",
            "teacher_adapter_contract": _teacher.contract(),
            "teacher_adapter_contract_sha256": _teacher.CONTRACT_SHA256,
        }
    )
    return value


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(record, label=label)
    return record


def _record_args(path: str, digest: str, *, label: str) -> dict[str, str]:
    return _record({"path": str(Path(path).expanduser().resolve()), "sha256": digest}, label=label)


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result) or result.exists() or result.is_symlink():
        raise ValueError(f"{label} output differs")
    return result


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256, label="top4 transport-v2 authority"
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256", "inputs", "outputs",
        "execution", "target_blind_materialization_authorized",
        "metric_execution_authorized", "access_audit",
    }
    if (
        set(raw) != required
        or raw.get("schema") != AUTHORITY_SCHEMA
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status") != "authorized_source_gated_transport_v2_top4_target_blind"
        or raw.get("scene_id") != "figurines"
        or raw.get("implementation") != file_record(Path(__file__).resolve())
        or raw.get("method_contract") != method_contract()
        or raw.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or raw.get("target_blind_materialization_authorized") is not True
        or raw.get("metric_execution_authorized") is not False
        or raw.get("access_audit") != _base.access_audit()
    ):
        raise ValueError("top4 transport-v2 authority differs")
    execution = raw.get("execution")
    if execution != {
        "physical_gpu": 0,
        "cuda_visible_devices": "0",
        "program_device": "cuda:0",
        "row_batch_size": _base.ROW_BATCH_SIZE,
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
        "maximum_power_limit_w": 300.5,
    }:
        raise ValueError("top4 transport-v2 execution differs")
    names = {
        "base_descriptor", "teacher_agreement_v2", "source_gate",
        "frozen_positive_bank", "frozen_negative_bank", "o0_positive", "o0_negative",
    }
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("top4 transport-v2 inputs differ")
    records = {name: _record(inputs[name], label=f"top4 transport {name}") for name in sorted(names)}
    base, rows = _base._old._o1o2._validate_base_descriptor_general(
        Path(records["base_descriptor"]["path"]), records["base_descriptor"]["sha256"]
    )
    teacher, teacher_digest, teacher_source = load_torch_mapping(
        records["teacher_agreement_v2"]["path"],
        expected_sha256=records["teacher_agreement_v2"]["sha256"], map_location="cpu",
        label="derived top4 teacher",
    )
    _teacher.validate_payload(teacher)
    if (
        teacher.get("scene_id") != "figurines"
        or teacher.get("base_descriptor") != records["base_descriptor"]
        or not torch.equal(teacher["global_rows"], rows)
    ):
        raise ValueError("derived top4 teacher/base differs")
    teacher_record = {"path": str(teacher_source), "sha256": teacher_digest}
    source_gate, source_gate_record = _base._validate_source_gate(
        records["source_gate"]["path"], records["source_gate"]["sha256"]
    )
    o0_positive, _, _ = load_torch_mapping(
        records["o0_positive"]["path"], expected_sha256=records["o0_positive"]["sha256"],
        map_location="cpu", label="top4 transport O0 positive"
    )
    o0_negative, _, _ = load_torch_mapping(
        records["o0_negative"]["path"], expected_sha256=records["o0_negative"]["sha256"],
        map_location="cpu", label="top4 transport O0 negative"
    )
    queries = list(o0_positive.get("query_ids", []))
    renderer_sha = o0_positive.get("renderer_geometry_checkpoint_sha256")
    if not queries or not isinstance(renderer_sha, str) or _SHA256.fullmatch(renderer_sha) is None:
        raise ValueError("top4 transport O0 query/renderer differs")
    _base._old._o1o2._validate_o0_pair(
        o0_positive, o0_negative, base=base, positive_queries=queries,
        renderer_sha256=renderer_sha,
    )
    frozen_positive, _, _ = load_torch_mapping(
        records["frozen_positive_bank"]["path"],
        expected_sha256=records["frozen_positive_bank"]["sha256"], map_location="cpu",
        label="top4 transport frozen positive",
    )
    frozen_negative, _, _ = load_torch_mapping(
        records["frozen_negative_bank"]["path"],
        expected_sha256=records["frozen_negative_bank"]["sha256"], map_location="cpu",
        label="top4 transport frozen negative",
    )
    positive_embeddings = _rebind.select_frozen_embeddings(frozen_positive, queries)
    negative_embeddings = _rebind.select_frozen_embeddings(
        frozen_negative, _base._old._o1o2.NEGATIVE_QUERIES
    )
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"positive", "negative", "result"}:
        raise ValueError("top4 transport outputs differ")
    resolved = {name: str(Path(str(value)).expanduser().resolve()) for name, value in outputs.items()}
    if any(str(outputs[name]) != value for name, value in resolved.items()):
        raise ValueError("top4 transport output path differs")
    return {
        "authority": dict(raw), "authority_record": {"path": str(source), "sha256": digest},
        "records": records, "base": base, "rows": rows, "teacher": teacher,
        "teacher_record": teacher_record, "source_gate": source_gate,
        "source_gate_record": source_gate_record, "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings, "o0_positive": o0_positive,
        "o0_negative": o0_negative, "outputs": resolved,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="top4 transport authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("top4 transport output root differs")
    outputs = {
        "positive": str(_new(output_root / "figurines_transport_v2_positive.pt", label="positive")),
        "negative": str(_new(output_root / "figurines_transport_v2_negative.pt", label="negative")),
        "result": str(_new(output_root / "figurines_transport_v2_result.json", label="result")),
    }
    inputs = {}
    for name in (
        "base_descriptor", "teacher_agreement_v2", "source_gate", "frozen_positive_bank",
        "frozen_negative_bank", "o0_positive", "o0_negative",
    ):
        inputs[name] = _record_args(getattr(args, name), getattr(args, f"{name}_sha256"), label=name)
    authority = {
        "schema": AUTHORITY_SCHEMA, "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_gated_transport_v2_top4_target_blind",
        "scene_id": "figurines", "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(), "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "inputs": inputs, "outputs": outputs,
        "execution": {
            "physical_gpu": 0, "cuda_visible_devices": "0", "program_device": "cuda:0",
            "row_batch_size": _base.ROW_BATCH_SIZE, "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0, "maximum_temperature_c": 88,
            "maximum_power_limit_w": 300.5,
        },
        "target_blind_materialization_authorized": True,
        "metric_execution_authorized": False, "access_audit": _base.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized_and_recursively_validated", "authority": record, "outputs": outputs}


def _candidate_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _base._candidate_cache(*args, **kwargs)
    authority = payload["authority"]
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["source_artifacts"]["materializer_source"] = file_record(Path(__file__).resolve())
    authority["source_artifacts"]["canonical_top4_teacher_adapter"] = dict(
        kwargs["teacher_record"]
    )
    authority["transport_v2_method_contract_sha256"] = METHOD_CONTRACT_SHA256
    return payload


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(args.execution_authority, expected_sha256=args.execution_authority_sha256)
    for path in prepared["outputs"].values():
        _new(path, label="top4 transport")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("top4 transport requires physical GPU0 only")
    teacher = prepared["teacher"]
    scored = _base.materialize_scores_lowmem(
        base_features_by_scale=prepared["base"]["features_by_scale"], global_rows=prepared["rows"],
        teacher_mean=teacher["teacher_mean"], teacher_valid=teacher["teacher_valid"],
        retained_view_count=teacher["retained_view_count"],
        directional_resultant=teacher["teacher_view_directional_resultant"],
        positive_embeddings=prepared["positive_embeddings"], negative_embeddings=prepared["negative_embeddings"],
        o0_positive_scores=prepared["o0_positive"]["query_scores"],
        o0_negative_scores=prepared["o0_negative"]["query_scores"], device=torch.device("cuda:0"),
    )
    positive = _candidate_cache(
        prepared["o0_positive"], scored["positive_scores"], teacher_record=prepared["teacher_record"],
        source_gate_record=prepared["source_gate_record"], text_record=prepared["records"]["frozen_positive_bank"],
        descriptor_sha256=scored["descriptor_sha256"],
    )
    negative = _candidate_cache(
        prepared["o0_negative"], scored["negative_scores"], teacher_record=prepared["teacher_record"],
        source_gate_record=prepared["source_gate_record"], text_record=prepared["records"]["frozen_negative_bank"],
        descriptor_sha256=scored["descriptor_sha256"],
    )
    write_torch_noclobber(prepared["outputs"]["positive"], positive)
    write_torch_noclobber(prepared["outputs"]["negative"], negative)
    outputs = {"positive": file_record(prepared["outputs"]["positive"]), "negative": file_record(prepared["outputs"]["negative"])}
    result = {
        "schema": RESULT_SCHEMA, "schema_version": SCHEMA_VERSION,
        "status": "complete_source_gated_transport_v2_top4_target_blind", "scene_id": "figurines",
        "execution_authority": prepared["authority_record"], "source_gate": prepared["source_gate_record"],
        "teacher_adapter": prepared["teacher_record"], "candidate_index": _base.SELECTED_CANDIDATE_INDEX,
        "maximum_angle_radians": _base.SELECTED_MAXIMUM_ANGLE_RADIANS,
        "gamma_policy": _base.SELECTED_GAMMA_POLICY, "outputs": outputs,
        "accepted_rows": int(prepared["rows"].numel()),
        "rows_with_teacher_applied": scored["rows_with_teacher_applied"],
        "rows_with_o0_descriptor_fallback": scored["rows_with_o0_descriptor_fallback"],
        "gamma_min": scored["gamma_min"], "gamma_max": scored["gamma_max"],
        "maximum_unit_norm_absolute_error": scored["maximum_unit_norm_absolute_error"],
        "descriptor_sha256": scored["descriptor_sha256"], "elapsed_seconds": time.monotonic() - started,
        "access_audit": _base.access_audit(), "metric_execution_authorized": False, "metric_executed": False,
    }
    write_frozen_json(prepared["outputs"]["result"], result)
    return {**result, "result": file_record(prepared["outputs"]["result"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for name in (
        "base_descriptor", "teacher_agreement_v2", "source_gate", "frozen_positive_bank",
        "frozen_negative_bank", "o0_positive", "o0_negative",
    ):
        option = name.replace("_", "-")
        build.add_argument(f"--{option}", required=True)
        build.add_argument(f"--{option}-sha256", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--authority-output", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--execution-authority-sha256", required=True)
    run.set_defaults(handler=materialize)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

