#!/usr/bin/env python3
"""Materialize the globally gated transport-v2 LERF raw-score candidate.

The source-only Ramen/Teatime gate fixes one global candidate: angle 0.45,
gamma k0.  This entrypoint only projects that query-independent descriptor
through frozen text banks.  It never opens target images, masks, labels, or
metrics and it does not modify the frozen evaluator.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_scale_residual_shrinkage_transport import (
    RESIDUAL_SHRINKAGE_CONTRACT_SHA256,
    residual_shrinkage_contract,
    scale_residual_shrinkage_transport,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as _frozen
from radio_gs.scripts import materialize_lerf_o1_o2_frozen_text_rebind as _rebind
from radio_gs.scripts import materialize_lerf_reliability_conditioned_candidate_scores as _old
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_transport_v2_target_blind_execution.v1"
RESULT_SCHEMA = "radio_gs.lerf_transport_v2_target_blind_result.v1"
SCHEMA_VERSION = 1
ROW_BATCH_SIZE = 256
SUPPORTED_PHYSICAL_GPUS = (0, 1)
SELECTED_CANDIDATE_INDEX = 11
SELECTED_MAXIMUM_ANGLE_RADIANS = 0.45
SELECTED_GAMMA_POLICY = "k0"
SOURCE_GATE_SHA256 = "13e5bcad24aa196fa63ba17a45a807e529bce3a19e749449c3430acc5c6b7587"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def access_audit() -> dict[str, bool]:
    return {
        "base_descriptor_opened": True,
        "teacher_agreement_v2_opened": True,
        "source_only_cross_scene_gate_opened": True,
        "frozen_text_embedding_banks_opened": True,
        "exact_o0_pair_opened_for_geometry_and_invalid_row_template": True,
        "target_images_opened": False,
        "target_ground_truth_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "target_metric_executed": False,
    }


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "residual_shrinkage_contract": residual_shrinkage_contract(),
        "residual_shrinkage_contract_sha256": RESIDUAL_SHRINKAGE_CONTRACT_SHA256,
        "source_gate_sha256": SOURCE_GATE_SHA256,
        "selected_candidate_index": SELECTED_CANDIDATE_INDEX,
        "selected_maximum_angle_radians": SELECTED_MAXIMUM_ANGLE_RADIANS,
        "selected_gamma_policy": SELECTED_GAMMA_POLICY,
        "immutable_backbone": "accepted_v2_o0_descriptor_three_scale_frame",
        "teacher_payload": "validated_teacher_agreement_v2_or_lowmem",
        "descriptor_materialization": "row_batch_only",
        "row_batch_size": ROW_BATCH_SIZE,
        "score_input": "l2_normalize(transport_v2_descriptor)",
        "score_dtype": "torch.float32",
        "accepted_rows_recomputed_against_frozen_text": True,
        "invalid_rows": "preserve_exact_o0_template_and_excluded_by_validity_domain",
        "scene_or_query_specific_parameters": False,
        "query_independent_transport": True,
        "target_data_or_metric_access": False,
        "metric_execution_authorized": False,
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _record_args(path: str, digest: str, *, label: str) -> dict[str, str]:
    return _record(
        {"path": str(Path(path).expanduser().resolve()), "sha256": digest},
        label=label,
    )


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def _typed_stream_hasher(shape: tuple[int, ...]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(torch.float32), "shape": list(shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    return digest


def _validate_source_gate(path: str | Path, digest: str) -> tuple[dict[str, Any], dict[str, str]]:
    if digest != SOURCE_GATE_SHA256:
        raise ValueError("transport-v2 source gate SHA differs")
    value, observed, source = load_json_object(
        path, expected_sha256=digest, label="transport-v2 cross-scene source gate"
    )
    gate = value.get("cross_scene_gate")
    selected = gate.get("selected_candidate") if isinstance(gate, Mapping) else None
    if (
        value.get("schema")
        != "radio_gs.lerf_transport_v2_ramen_teatime_cross_scene_source_gate_result.v1"
        or value.get("status") != "complete_source_gate_passed_no_target_execution"
        or value.get("target_candidate_authorized") is not False
        or value.get("target_metric_executed") is not False
        or not isinstance(gate, Mapping)
        or gate.get("source_scene_ids") != ["ramen", "teatime"]
        or gate.get("source_gate_passed") is not True
        or gate.get("selected_candidate_index") != SELECTED_CANDIDATE_INDEX
        or not isinstance(selected, Mapping)
        or selected.get("maximum_angle_radians") != SELECTED_MAXIMUM_ANGLE_RADIANS
        or selected.get("gamma_policy") != SELECTED_GAMMA_POLICY
        or selected.get("every_scene_mean_nonregression") is not True
        or selected.get("every_scene_p05_nonregression") is not True
        or float(selected.get("pooled_mean_delta_vs_baseline", 0.0)) <= 0.0
    ):
        raise ValueError("transport-v2 source gate selection differs")
    return dict(value), {"path": str(source), "sha256": observed}


@torch.inference_mode()
def materialize_scores_lowmem(
    *,
    base_features_by_scale: torch.Tensor,
    global_rows: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    directional_resultant: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    o0_positive_scores: torch.Tensor,
    o0_negative_scores: torch.Tensor,
    device: torch.device,
    row_batch_size: int = ROW_BATCH_SIZE,
) -> dict[str, Any]:
    base = torch.as_tensor(base_features_by_scale)
    rows = torch.as_tensor(global_rows)
    n_rows, _, dim = base.shape
    positive_text = F.normalize(torch.as_tensor(positive_embeddings).float(), dim=-1)
    negative_text = F.normalize(torch.as_tensor(negative_embeddings).float(), dim=-1)
    positive = torch.as_tensor(o0_positive_scores)
    negative = torch.as_tensor(o0_negative_scores)
    if (
        base.ndim != 3
        or base.shape[1] != 3
        or rows.shape != (n_rows,)
        or rows.dtype != torch.int64
        or teacher_mean.shape != (n_rows, dim)
        or teacher_valid.shape != (n_rows,)
        or retained_view_count.shape != (n_rows,)
        or directional_resultant.shape != (n_rows,)
        or positive_text.ndim != 2
        or negative_text.ndim != 2
        or positive_text.shape[1] != dim
        or negative_text.shape[1] != dim
        or positive.ndim != 3
        or negative.ndim != 3
        or positive.shape[1] != 3
        or negative.shape[1] != 3
        or positive.shape[2] != positive_text.shape[0]
        or negative.shape[2] != negative_text.shape[0]
        or positive.dtype != torch.float32
        or negative.dtype != torch.float32
        or not isinstance(row_batch_size, int)
        or row_batch_size < 1
        or (n_rows > 1 and not bool((rows[1:] > rows[:-1]).all()))
        or (n_rows > 0 and int(rows[-1]) >= positive.shape[0])
    ):
        raise ValueError("transport-v2 target score axes differ")
    candidate_positive = positive.clone()
    candidate_negative = negative.clone()
    positive_text = positive_text.to(device)
    negative_text = negative_text.to(device)
    hasher = _typed_stream_hasher((n_rows, 3, dim))
    rows_teacher_applied = 0
    rows_fallback = 0
    gamma_min = 1.0
    gamma_max = 0.0
    maximum_norm_error = 0.0
    for start in range(0, n_rows, row_batch_size):
        stop = min(start + row_batch_size, n_rows)
        base_batch = base[start:stop].to(device).float()
        output = scale_residual_shrinkage_transport(
            base_batch,
            teacher_mean[start:stop].to(device).float(),
            teacher_valid=teacher_valid[start:stop].to(device),
            retained_view_count=retained_view_count[start:stop].to(device),
            teacher_view_directional_resultant=directional_resultant[start:stop].to(device),
            maximum_angle_radians=SELECTED_MAXIMUM_ANGLE_RADIANS,
            gamma_policy=SELECTED_GAMMA_POLICY,
        )
        descriptor = output.descriptor
        unit = F.normalize(descriptor, dim=-1)
        global_batch = rows[start:stop]
        candidate_positive[global_batch] = torch.einsum(
            "bsd,qd->bsq", unit, positive_text
        ).cpu()
        candidate_negative[global_batch] = torch.einsum(
            "bsd,qd->bsq", unit, negative_text
        ).cpu()
        descriptor_cpu = descriptor.cpu().contiguous()
        hasher.update(descriptor_cpu.numpy().tobytes(order="C"))
        applied = output.teacher_applied.any(dim=-1)
        rows_teacher_applied += int(applied.sum().cpu())
        rows_fallback += int((~applied).sum().cpu())
        gamma_min = min(gamma_min, float(output.gamma.min().cpu()))
        gamma_max = max(gamma_max, float(output.gamma.max().cpu()))
        maximum_norm_error = max(
            maximum_norm_error,
            float((torch.linalg.vector_norm(unit, dim=-1) - 1.0).abs().max().cpu()),
        )
    if not bool(torch.isfinite(candidate_positive).all()) or not bool(
        torch.isfinite(candidate_negative).all()
    ):
        raise ValueError("transport-v2 target scores are nonfinite")
    return {
        "positive_scores": candidate_positive.contiguous(),
        "negative_scores": candidate_negative.contiguous(),
        "descriptor_sha256": hasher.hexdigest(),
        "rows_with_teacher_applied": rows_teacher_applied,
        "rows_with_o0_descriptor_fallback": rows_fallback,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "maximum_unit_norm_absolute_error": maximum_norm_error,
    }


def _candidate_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    teacher_record: Mapping[str, str],
    source_gate_record: Mapping[str, str],
    text_record: Mapping[str, str],
    descriptor_sha256: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in template.items() if key != "authority"}
    payload["query_scores"] = scores.contiguous().float()
    authority = copy.deepcopy(template["authority"])
    authority["contract"] = _old._o1o2.RAW_AUTHORITY_CONTRACT
    authority["score_semantics"] = "raw_independent_normalized_cosine"
    authority["score_formula"] = (
        "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
    )
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["score_dtype"] = "torch.float32"
    authority["query_scores_sha256"] = _old._o1o2.tensor_sha256_typed(
        payload["query_scores"]
    )
    authority["descriptor_axis"]["features_by_scale_sha256"] = descriptor_sha256
    authority["descriptor_axis"]["oracle"] = "source_gated_scale_residual_shrinkage_transport_v2"
    authority["descriptor_axis"]["execution_representation"] = (
        "row_streamed_three_scale_no_full_descriptor"
    )
    sources = authority["source_artifacts"]
    sources["descriptor_cache"] = dict(teacher_record)
    sources["text_query_cache"] = dict(text_record)
    sources["transport_v2_source_gate"] = dict(source_gate_record)
    sources["materializer_source"] = file_record(Path(__file__).resolve())
    authority["calibration_constraints"]["benchmark_metrics_opened"] = False
    authority["transport_v2_candidate_index"] = SELECTED_CANDIDATE_INDEX
    authority["transport_v2_maximum_angle_radians"] = SELECTED_MAXIMUM_ANGLE_RADIANS
    authority["transport_v2_gamma_policy"] = SELECTED_GAMMA_POLICY
    authority["residual_shrinkage_contract_sha256"] = RESIDUAL_SHRINKAGE_CONTRACT_SHA256
    authority["transport_v2_method_contract_sha256"] = METHOD_CONTRACT_SHA256
    payload["authority"] = authority
    return payload


def prepare_inputs(authority_path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="transport-v2 target-blind execution authority",
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
        or raw.get("status") != "authorized_source_gated_transport_v2_target_blind"
        or raw.get("implementation") != file_record(Path(__file__).resolve())
        or raw.get("method_contract") != method_contract()
        or raw.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or raw.get("target_blind_materialization_authorized") is not True
        or raw.get("metric_execution_authorized") is not False
        or raw.get("access_audit") != access_audit()
    ):
        raise ValueError("transport-v2 target-blind authority differs")
    scene_id = raw.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("transport-v2 target scene id differs")
    execution = raw.get("execution")
    physical = execution.get("physical_gpu") if isinstance(execution, Mapping) else None
    if physical not in SUPPORTED_PHYSICAL_GPUS or execution != {
        "physical_gpu": physical,
        "cuda_visible_devices": str(physical),
        "program_device": "cuda:0",
        "row_batch_size": ROW_BATCH_SIZE,
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
        "maximum_power_limit_w": 300.5,
    }:
        raise ValueError("transport-v2 target device differs")
    expected_names = {
        "base_descriptor", "teacher_agreement_v2", "source_gate",
        "frozen_positive_bank", "frozen_negative_bank", "o0_positive", "o0_negative",
    }
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_names:
        raise ValueError("transport-v2 target inputs differ")
    records = {name: _record(inputs[name], label=f"transport-v2 {name}") for name in sorted(expected_names)}
    base, rows = _old._o1o2._validate_base_descriptor_general(
        Path(records["base_descriptor"]["path"]), records["base_descriptor"]["sha256"]
    )
    teacher, teacher_record = _old._validate_teacher_payload(
        records["teacher_agreement_v2"]["path"],
        records["teacher_agreement_v2"]["sha256"],
        scene_id=scene_id,
        base_descriptor_record=records["base_descriptor"],
    )
    if not torch.equal(teacher["global_rows"], rows):
        raise ValueError("transport-v2 teacher/base lineage differs")
    source_gate, source_gate_record = _validate_source_gate(
        records["source_gate"]["path"], records["source_gate"]["sha256"]
    )
    o0_positive, _, _ = load_torch_mapping(
        records["o0_positive"]["path"], expected_sha256=records["o0_positive"]["sha256"],
        map_location="cpu", label="transport-v2 O0 positive cache"
    )
    o0_negative, _, _ = load_torch_mapping(
        records["o0_negative"]["path"], expected_sha256=records["o0_negative"]["sha256"],
        map_location="cpu", label="transport-v2 O0 negative cache"
    )
    positive_queries = list(o0_positive.get("query_ids", []))
    renderer_sha = o0_positive.get("renderer_geometry_checkpoint_sha256")
    if not positive_queries or not isinstance(renderer_sha, str) or _SHA256.fullmatch(renderer_sha) is None:
        raise ValueError("transport-v2 O0 query/renderer lineage differs")
    _old._o1o2._validate_o0_pair(
        o0_positive, o0_negative, base=base,
        positive_queries=positive_queries, renderer_sha256=renderer_sha
    )
    frozen_positive, _, _ = load_torch_mapping(
        records["frozen_positive_bank"]["path"],
        expected_sha256=records["frozen_positive_bank"]["sha256"], map_location="cpu",
        label="transport-v2 frozen positive bank"
    )
    frozen_negative, _, _ = load_torch_mapping(
        records["frozen_negative_bank"]["path"],
        expected_sha256=records["frozen_negative_bank"]["sha256"], map_location="cpu",
        label="transport-v2 frozen negative bank"
    )
    positive_embeddings = _rebind.select_frozen_embeddings(frozen_positive, positive_queries)
    negative_embeddings = _rebind.select_frozen_embeddings(
        frozen_negative, _old._o1o2.NEGATIVE_QUERIES
    )
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"positive", "negative", "result"}:
        raise ValueError("transport-v2 target outputs differ")
    resolved = {name: str(Path(str(value)).expanduser().resolve()) for name, value in outputs.items()}
    if any(str(outputs[name]) != value for name, value in resolved.items()):
        raise ValueError("transport-v2 target output must be canonical absolute")
    return {
        "authority": dict(raw),
        "authority_record": {"path": str(source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "teacher": teacher,
        "teacher_record": teacher_record,
        "source_gate": source_gate,
        "source_gate_record": source_gate_record,
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
        "o0_positive": o0_positive,
        "o0_negative": o0_negative,
        "outputs": resolved,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="transport-v2 authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("transport-v2 output directory must be canonical absolute")
    outputs = {
        "positive": str(_new(output_root / f"{args.scene_id}_transport_v2_positive.pt", label="positive output")),
        "negative": str(_new(output_root / f"{args.scene_id}_transport_v2_negative.pt", label="negative output")),
        "result": str(_new(output_root / f"{args.scene_id}_transport_v2_result.json", label="result output")),
    }
    inputs = {}
    for name in (
        "base_descriptor", "teacher_agreement_v2", "source_gate",
        "frozen_positive_bank", "frozen_negative_bank", "o0_positive", "o0_negative",
    ):
        inputs[name] = _record_args(
            getattr(args, name), getattr(args, f"{name}_sha256"), label=name
        )
    physical = int(args.physical_gpu)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_gated_transport_v2_target_blind",
        "scene_id": args.scene_id,
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": physical,
            "cuda_visible_devices": str(physical),
            "program_device": "cuda:0",
            "row_batch_size": ROW_BATCH_SIZE,
            "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0,
            "maximum_temperature_c": 88,
            "maximum_power_limit_w": 300.5,
        },
        "target_blind_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": access_audit(),
    }
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized_and_recursively_validated", "authority": record, "outputs": outputs}


def validate_runtime_device(physical_gpu: int) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("transport-v2 CUDA visibility differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("transport-v2 requires one visible CUDA device")
    return torch.device("cuda:0")


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(args.execution_authority, expected_sha256=args.execution_authority_sha256)
    for path in prepared["outputs"].values():
        _new(path, label="transport-v2 output")
    physical = int(prepared["authority"]["execution"]["physical_gpu"])
    device = validate_runtime_device(physical)
    teacher = prepared["teacher"]
    scored = materialize_scores_lowmem(
        base_features_by_scale=prepared["base"]["features_by_scale"],
        global_rows=prepared["rows"],
        teacher_mean=teacher["teacher_mean"],
        teacher_valid=teacher["teacher_valid"],
        retained_view_count=teacher["retained_view_count"],
        directional_resultant=teacher["teacher_view_directional_resultant"],
        positive_embeddings=prepared["positive_embeddings"],
        negative_embeddings=prepared["negative_embeddings"],
        o0_positive_scores=prepared["o0_positive"]["query_scores"],
        o0_negative_scores=prepared["o0_negative"]["query_scores"],
        device=device,
    )
    positive = _candidate_cache(
        prepared["o0_positive"], scored["positive_scores"],
        teacher_record=prepared["teacher_record"], source_gate_record=prepared["source_gate_record"],
        text_record=prepared["records"]["frozen_positive_bank"], descriptor_sha256=scored["descriptor_sha256"]
    )
    negative = _candidate_cache(
        prepared["o0_negative"], scored["negative_scores"],
        teacher_record=prepared["teacher_record"], source_gate_record=prepared["source_gate_record"],
        text_record=prepared["records"]["frozen_negative_bank"], descriptor_sha256=scored["descriptor_sha256"]
    )
    write_torch_noclobber(prepared["outputs"]["positive"], positive)
    write_torch_noclobber(prepared["outputs"]["negative"], negative)
    output_records = {
        "positive": file_record(prepared["outputs"]["positive"]),
        "negative": file_record(prepared["outputs"]["negative"]),
    }
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_gated_transport_v2_target_blind",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": prepared["authority_record"],
        "source_gate": prepared["source_gate_record"],
        "candidate_index": SELECTED_CANDIDATE_INDEX,
        "maximum_angle_radians": SELECTED_MAXIMUM_ANGLE_RADIANS,
        "gamma_policy": SELECTED_GAMMA_POLICY,
        "outputs": output_records,
        "accepted_rows": int(prepared["rows"].numel()),
        "rows_with_teacher_applied": scored["rows_with_teacher_applied"],
        "rows_with_o0_descriptor_fallback": scored["rows_with_o0_descriptor_fallback"],
        "gamma_min": scored["gamma_min"],
        "gamma_max": scored["gamma_max"],
        "maximum_unit_norm_absolute_error": scored["maximum_unit_norm_absolute_error"],
        "descriptor_sha256": scored["descriptor_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
        "metric_executed": False,
    }
    write_frozen_json(prepared["outputs"]["result"], result)
    return {**result, "result": file_record(prepared["outputs"]["result"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--scene-id", required=True)
    for name in (
        "base_descriptor", "teacher_agreement_v2", "source_gate",
        "frozen_positive_bank", "frozen_negative_bank", "o0_positive", "o0_negative",
    ):
        option = name.replace("_", "-")
        build.add_argument(f"--{option}", required=True)
        build.add_argument(f"--{option}-sha256", required=True)
    build.add_argument("--physical-gpu", type=int, choices=SUPPORTED_PHYSICAL_GPUS, required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--authority-output", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--execution-authority-sha256", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA", "METHOD_CONTRACT_SHA256", "RESULT_SCHEMA",
    "SELECTED_CANDIDATE_INDEX", "SOURCE_GATE_SHA256", "access_audit",
    "materialize_scores_lowmem", "method_contract", "validate_runtime_device",
]
