#!/usr/bin/env python3
"""Materialize a low-memory scale-equivariant transport LERF candidate.

This development entrypoint reuses the fully validated AcceptedV2 O0,
teacher-agreement-v2, text, and exact O0 cache lineages.  The historical
source-only selector is accepted only as a frozen candidate-grid/tentative
ceiling record: its independent-scale LOO statistic does not authorize this
new common-rotation transport.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_scale_equivariant_geodesic_transport import (
    CEILING_GRID_RADIANS,
    TRANSPORT_CONTRACT_SHA256,
    scale_equivariant_geodesic_transport,
    transport_contract,
)
from radio_gs.scripts import (
    materialize_lerf_reliability_conditioned_candidate_scores as _old,
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


AUTHORITY_SCHEMA = "radio_gs.lerf_scale_equivariant_transport_execution.v1"
RESULT_SCHEMA = "radio_gs.lerf_scale_equivariant_transport_result.v1"
SCHEMA_VERSION = 1
ROW_BATCH_SIZE = 256
SCALE_COUNT = 3
SUPPORTED_PHYSICAL_GPUS = (0, 1)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ACCESS_AUDIT = {
    "base_descriptor_opened": True,
    "teacher_agreement_v2_opened": True,
    "independent_scale_selector_opened_for_development_grid_only": True,
    "exact_query_axis_opened": True,
    "exact_o0_pair_opened": True,
    "target_images_opened": False,
    "target_ground_truth_opened": False,
    "target_masks_opened": False,
    "target_metrics_opened": False,
    "target_quality_readout_executed": False,
}


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "transport_contract": transport_contract(),
        "transport_contract_sha256": TRANSPORT_CONTRACT_SHA256,
        "immutable_backbone": "accepted_v2_o0_descriptor_three_scale_frame",
        "teacher_payload": "validated_teacher_agreement_v2_or_lowmem",
        "candidate_grid_radians": list(CEILING_GRID_RADIANS),
        "tentative_ceiling_source": (
            "old_independent_scale_source_only_selector_grid_and_value_only"
        ),
        "old_selector_authorizes_transport": False,
        "development_status": (
            "requires_new_source_only_transport_loo_before_formal_target_claim"
        ),
        "descriptor_materialization": "row_batch_only",
        "row_batch_size": ROW_BATCH_SIZE,
        "full_n_by_3_by_d_candidate_descriptor_allocated": False,
        "score_input": "l2_normalize(rotated_descriptor_only_for_cosine_readout)",
        "post_rotation_descriptor_renormalization": False,
        "score_dtype": "torch.float32",
        "score_semantics": "raw_independent_normalized_cosine",
        "fallback_score_semantics": "bitwise_exact_o0_cache",
        "one_global_ceiling": True,
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
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": digest}
    validate_file_record(record, label=label)
    return record


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


def transport_descriptor_batch(
    base: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    directional_resultant: torch.Tensor,
    *,
    global_ceiling_radians: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    output = scale_equivariant_geodesic_transport(
        base,
        teacher_mean,
        teacher_valid=teacher_valid,
        retained_view_count=retained_view_count,
        teacher_view_directional_resultant=directional_resultant,
        maximum_angle_radians=global_ceiling_radians,
    )
    return output.descriptor, output.teacher_applied, {
        "angular_budget_radians": output.angular_budget_radians,
        "reliability_score": output.reliability_score,
        "expanded_budget": output.expanded_budget,
        "interface_fallback": output.fallback_to_o0,
        "frame_mean_valid": output.frame_mean_valid,
        "same_direction": output.same_direction,
        "antipodal": output.antipodal,
    }


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
    global_ceiling_radians: float,
    device: torch.device,
    row_batch_size: int = ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the transport without allocating a scene-wide descriptor."""

    base = torch.as_tensor(base_features_by_scale)
    rows = torch.as_tensor(global_rows)
    n_rows = int(rows.numel())
    positive_text = F.normalize(torch.as_tensor(positive_embeddings).float(), dim=-1)
    negative_text = F.normalize(torch.as_tensor(negative_embeddings).float(), dim=-1)
    positive = torch.as_tensor(o0_positive_scores)
    negative = torch.as_tensor(o0_negative_scores)
    if (
        base.ndim != 3
        or base.shape[:2] != (n_rows, SCALE_COUNT)
        or base.shape[-1] <= 1
        or not base.is_floating_point()
        or not bool(torch.isfinite(base).all())
        or rows.shape != (n_rows,)
        or rows.dtype != torch.int64
        or (n_rows > 1 and not bool((rows[1:] > rows[:-1]).all()))
        or not isinstance(row_batch_size, int)
        or row_batch_size < 1
        or teacher_mean.shape != (n_rows, base.shape[-1])
        or teacher_valid.shape != (n_rows,)
        or teacher_valid.dtype != torch.bool
        or retained_view_count.shape != (n_rows,)
        or directional_resultant.shape != (n_rows,)
        or positive_text.ndim != 2
        or negative_text.ndim != 2
        or positive_text.shape[1] != base.shape[-1]
        or negative_text.shape[1] != base.shape[-1]
        or positive.ndim != 3
        or negative.ndim != 3
        or positive.shape[:2] != (negative.shape[0], SCALE_COUNT)
        or negative.shape[1] != SCALE_COUNT
        or positive.shape[2] != positive_text.shape[0]
        or negative.shape[2] != negative_text.shape[0]
        or positive.dtype != torch.float32
        or negative.dtype != torch.float32
        or (n_rows > 0 and int(rows[-1]) >= positive.shape[0])
        or not bool(torch.isfinite(positive).all())
        or not bool(torch.isfinite(negative).all())
    ):
        raise ValueError("transport score materialization axes differ")
    if float(global_ceiling_radians) not in CEILING_GRID_RADIANS:
        raise ValueError("transport ceiling differs from frozen grid")

    candidate_positive = positive.clone()
    candidate_negative = negative.clone()
    positive_text = positive_text.to(device)
    negative_text = negative_text.to(device)
    descriptor_hasher = _typed_stream_hasher(
        (n_rows, SCALE_COUNT, int(base.shape[-1]))
    )
    rows_replaced = 0
    scales_replaced = 0
    rows_expanded = 0
    maximum_batch_rows = 0
    maximum_norm_error = 0.0
    maximum_gram_error = 0.0
    for start in range(0, n_rows, row_batch_size):
        stop = min(n_rows, start + row_batch_size)
        maximum_batch_rows = max(maximum_batch_rows, stop - start)
        base_batch = base[start:stop].to(device).float()
        descriptor, replace, audit = transport_descriptor_batch(
            base_batch,
            teacher_mean[start:stop].to(device),
            teacher_valid[start:stop].to(device),
            retained_view_count[start:stop].to(device),
            directional_resultant[start:stop].to(device),
            global_ceiling_radians=global_ceiling_radians,
        )
        before_norm = torch.linalg.vector_norm(base_batch, dim=-1)
        after_norm = torch.linalg.vector_norm(descriptor, dim=-1)
        before_gram = torch.einsum("bsd,btd->bst", base_batch, base_batch)
        after_gram = torch.einsum("bsd,btd->bst", descriptor, descriptor)
        maximum_norm_error = max(
            maximum_norm_error,
            float((after_norm - before_norm).abs().max().cpu()),
        )
        maximum_gram_error = max(
            maximum_gram_error,
            float((after_gram - before_gram).abs().max().cpu()),
        )
        descriptor_cpu = descriptor.cpu()
        descriptor_hasher.update(descriptor_cpu.numpy().tobytes(order="C"))
        replace_cpu = replace.cpu()
        descriptor_unit = F.normalize(descriptor, dim=-1)
        positive_batch = (
            descriptor_unit[:, :, None, :] * positive_text[None, None, :, :]
        ).sum(dim=-1).cpu()
        negative_batch = (
            descriptor_unit[:, :, None, :] * negative_text[None, None, :, :]
        ).sum(dim=-1).cpu()
        output_rows = rows[start:stop]
        candidate_positive[output_rows] = torch.where(
            replace_cpu[..., None], positive_batch, candidate_positive[output_rows]
        )
        candidate_negative[output_rows] = torch.where(
            replace_cpu[..., None], negative_batch, candidate_negative[output_rows]
        )
        rows_replaced += int(replace_cpu.any(dim=1).sum())
        scales_replaced += int(replace_cpu.sum())
        rows_expanded += int(
            (audit["expanded_budget"] & replace.any(dim=1)).sum().cpu()
        )
        del descriptor, descriptor_cpu, descriptor_unit, replace, replace_cpu, audit
        del base_batch, positive_batch, negative_batch
    return {
        "positive_scores": candidate_positive.contiguous(),
        "negative_scores": candidate_negative.contiguous(),
        "descriptor_sha256": descriptor_hasher.hexdigest(),
        "rows_with_score_replacement": rows_replaced,
        "scales_with_score_replacement": scales_replaced,
        "rows_with_expanded_budget": rows_expanded,
        "maximum_batch_rows_observed": maximum_batch_rows,
        "maximum_per_scale_norm_absolute_error": maximum_norm_error,
        "maximum_scale_gram_absolute_error": maximum_gram_error,
    }


def _candidate_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    teacher_record: Mapping[str, str],
    selector_record: Mapping[str, str],
    text_record: Mapping[str, str],
    descriptor_sha256: str,
    global_ceiling_radians: float,
) -> dict[str, Any]:
    payload = _old._candidate_cache(
        template,
        scores,
        teacher_record=teacher_record,
        selector_record=selector_record,
        text_record=text_record,
        descriptor_sha256=descriptor_sha256,
        global_ceiling_radians=global_ceiling_radians,
    )
    authority = copy.deepcopy(payload["authority"])
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["descriptor_axis"]["oracle"] = (
        "scale_equivariant_geodesic_transport_development_v1"
    )
    authority["descriptor_axis"]["execution_representation"] = (
        "row_streamed_common_orthogonal_rotation_no_full_descriptor"
    )
    authority["source_artifacts"]["materializer_source"] = file_record(
        Path(__file__).resolve()
    )
    authority["transport_contract_sha256"] = TRANSPORT_CONTRACT_SHA256
    authority["transport_method_contract_sha256"] = METHOD_CONTRACT_SHA256
    authority["transport_ceiling_authorization"] = (
        "development_only_old_independent_scale_selector_not_formal_transport_authority"
    )
    payload["authority"] = authority
    return payload


def prepare_inputs(
    authority_path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="transport execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256", "inputs", "outputs",
        "execution", "development_materialization_authorized",
        "formal_transport_ceiling_authorized", "metric_execution_authorized",
        "access_audit",
    }
    if (
        set(raw) != required
        or raw.get("schema") != AUTHORITY_SCHEMA
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status")
        != "authorized_development_scale_equivariant_transport_sentinel"
        or raw.get("implementation") != file_record(Path(__file__).resolve())
        or raw.get("method_contract") != method_contract()
        or raw.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or raw.get("development_materialization_authorized") is not True
        or raw.get("formal_transport_ceiling_authorized") is not False
        or raw.get("metric_execution_authorized") is not False
        or raw.get("access_audit") != EXPECTED_ACCESS_AUDIT
    ):
        raise ValueError("transport execution authority differs")
    scene_id = raw.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("transport scene id differs")
    execution = raw.get("execution")
    physical = execution.get("physical_gpu") if isinstance(execution, Mapping) else None
    if execution != {
        "physical_gpu": physical,
        "cuda_visible_devices": str(physical),
        "program_device": "cuda:0",
        "row_batch_size": ROW_BATCH_SIZE,
        "thermal_safety_owner": "external_300s_hard88_guard",
        "maximum_temperature_c": 88,
    } or physical not in SUPPORTED_PHYSICAL_GPUS:
        raise ValueError("transport execution device differs")
    expected_inputs = {
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    }
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("transport input records differ")
    records = {
        name: _record(inputs[name], label=f"transport {name}")
        for name in sorted(expected_inputs)
    }
    base, rows = _old._o1o2._validate_base_descriptor_general(
        Path(records["base_descriptor"]["path"]),
        records["base_descriptor"]["sha256"],
    )
    teacher, teacher_record = _old._validate_teacher_payload(
        records["teacher_agreement_v2"]["path"],
        records["teacher_agreement_v2"]["sha256"],
        scene_id=scene_id,
        base_descriptor_record=records["base_descriptor"],
    )
    if not torch.equal(teacher["global_rows"], rows):
        raise ValueError("transport teacher/base lineage differs")
    selector, selector_record, ceiling = _old._validate_selector_candidate(
        records["global_ceiling_selector"]["path"],
        records["global_ceiling_selector"]["sha256"],
    )
    positive_raw, _, _ = load_torch_mapping(
        records["positive_text"]["path"],
        expected_sha256=records["positive_text"]["sha256"],
        map_location="cpu",
        label="transport positive text bank",
    )
    negative_raw, _, _ = load_torch_mapping(
        records["negative_text"]["path"],
        expected_sha256=records["negative_text"]["sha256"],
        map_location="cpu",
        label="transport negative text bank",
    )
    positive_queries = list(positive_raw.get("queries", []))
    positive_embeddings = _old._o1o2._validate_text_bank(
        positive_raw, expected_queries=positive_queries
    )
    negative_embeddings = _old._o1o2._validate_text_bank(
        negative_raw, expected_queries=list(_old._o1o2.NEGATIVE_QUERIES)
    )
    o0_positive, _, _ = load_torch_mapping(
        records["o0_positive"]["path"],
        expected_sha256=records["o0_positive"]["sha256"],
        map_location="cpu",
        label="transport O0 positive cache",
    )
    o0_negative, _, _ = load_torch_mapping(
        records["o0_negative"]["path"],
        expected_sha256=records["o0_negative"]["sha256"],
        map_location="cpu",
        label="transport O0 negative cache",
    )
    renderer_sha = o0_positive.get("renderer_geometry_checkpoint_sha256")
    if not isinstance(renderer_sha, str) or _SHA256.fullmatch(renderer_sha) is None:
        raise ValueError("transport O0 renderer lineage differs")
    _old._o1o2._validate_o0_pair(
        o0_positive,
        o0_negative,
        base=base,
        positive_queries=positive_queries,
        renderer_sha256=renderer_sha,
    )
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "positive", "negative", "result"
    }:
        raise ValueError("transport outputs differ")
    resolved_outputs = {
        name: str(Path(str(path)).expanduser().resolve())
        for name, path in outputs.items()
    }
    if any(str(outputs[name]) != value for name, value in resolved_outputs.items()):
        raise ValueError("transport output must be canonical absolute")
    return {
        "authority": dict(raw),
        "authority_record": {"path": str(source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "teacher": teacher,
        "teacher_record": teacher_record,
        "selector": selector,
        "selector_record": selector_record,
        "global_ceiling_radians": ceiling,
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
        "o0_positive": o0_positive,
        "o0_negative": o0_negative,
        "outputs": resolved_outputs,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="transport authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("transport output directory must be canonical absolute")
    outputs = {
        "positive": str(_new(output_root / f"{args.scene_id}_scale_transport_development_positive.pt", label="transport positive output")),
        "negative": str(_new(output_root / f"{args.scene_id}_scale_transport_development_negative.pt", label="transport negative output")),
        "result": str(_new(output_root / f"{args.scene_id}_scale_transport_development_result.json", label="transport result output")),
    }
    names = (
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    )
    inputs = {
        name: _record_args(
            getattr(args, name), getattr(args, f"{name}_sha256"), label=name
        )
        for name in names
    }
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_development_scale_equivariant_transport_sentinel",
        "scene_id": args.scene_id,
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": str(args.physical_gpu),
            "program_device": "cuda:0",
            "row_batch_size": ROW_BATCH_SIZE,
            "thermal_safety_owner": "external_300s_hard88_guard",
            "maximum_temperature_c": 88,
        },
        "development_materialization_authorized": True,
        "formal_transport_ceiling_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": dict(EXPECTED_ACCESS_AUDIT),
    }
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized_development", "authority": record, "outputs": outputs}


def validate_runtime_device(
    execution: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    cuda_available: bool | None = None,
) -> torch.device:
    physical = execution.get("physical_gpu")
    environment = os.environ if environ is None else environ
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if (
        physical not in SUPPORTED_PHYSICAL_GPUS
        or execution.get("cuda_visible_devices") != str(physical)
        or execution.get("program_device") != "cuda:0"
        or environment.get("CUDA_VISIBLE_DEVICES") != str(physical)
        or available is not True
    ):
        raise RuntimeError("transport runtime CUDA device authority differs")
    return torch.device("cuda:0")


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    for name, path in prepared["outputs"].items():
        _new(path, label=f"transport {name} output")
    device = validate_runtime_device(prepared["authority"]["execution"])
    teacher = prepared["teacher"]
    scored = materialize_scores_lowmem(
        base_features_by_scale=prepared["base"]["features_by_scale"],
        global_rows=prepared["rows"],
        teacher_mean=teacher["teacher_mean"],
        teacher_valid=teacher["teacher_valid"],
        retained_view_count=teacher["retained_view_count"],
        directional_resultant=teacher[_old.VIEW_AGREEMENT_SCALAR],
        positive_embeddings=prepared["positive_embeddings"],
        negative_embeddings=prepared["negative_embeddings"],
        o0_positive_scores=prepared["o0_positive"]["query_scores"],
        o0_negative_scores=prepared["o0_negative"]["query_scores"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
        device=device,
    )
    positive = _candidate_cache(
        prepared["o0_positive"],
        scored["positive_scores"],
        teacher_record=prepared["teacher_record"],
        selector_record=prepared["selector_record"],
        text_record=prepared["records"]["positive_text"],
        descriptor_sha256=scored["descriptor_sha256"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
    )
    negative = _candidate_cache(
        prepared["o0_negative"],
        scored["negative_scores"],
        teacher_record=prepared["teacher_record"],
        selector_record=prepared["selector_record"],
        text_record=prepared["records"]["negative_text"],
        descriptor_sha256=scored["descriptor_sha256"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
    )
    write_torch_noclobber(prepared["outputs"]["positive"], positive)
    positive_record = file_record(prepared["outputs"]["positive"])
    write_torch_noclobber(prepared["outputs"]["negative"], negative)
    negative_record = file_record(prepared["outputs"]["negative"])
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_development_scale_equivariant_transport_sentinel",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": prepared["authority_record"],
        "teacher_agreement_v2": prepared["teacher_record"],
        "development_grid_selector": prepared["selector_record"],
        "formal_transport_ceiling_authorized": False,
        "global_ceiling_radians": prepared["global_ceiling_radians"],
        "outputs": {"positive": positive_record, "negative": negative_record},
        "accepted_rows": int(prepared["rows"].numel()),
        "rows_with_score_replacement": scored["rows_with_score_replacement"],
        "scales_with_score_replacement": scored["scales_with_score_replacement"],
        "rows_with_expanded_budget": scored["rows_with_expanded_budget"],
        "maximum_batch_rows_observed": scored["maximum_batch_rows_observed"],
        "maximum_per_scale_norm_absolute_error": scored[
            "maximum_per_scale_norm_absolute_error"
        ],
        "maximum_scale_gram_absolute_error": scored[
            "maximum_scale_gram_absolute_error"
        ],
        "descriptor_sha256": scored["descriptor_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "transport_contract_sha256": TRANSPORT_CONTRACT_SHA256,
        "access_audit": dict(EXPECTED_ACCESS_AUDIT),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "next_gate": "explicit_development_metric_authority_or_new_source_only_transport_loo",
    }
    write_frozen_json(prepared["outputs"]["result"], result)
    return {**result, "result": file_record(prepared["outputs"]["result"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--scene-id", required=True)
    for name in (
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    ):
        option = name.replace("_", "-")
        build.add_argument(f"--{option}", required=True)
        build.add_argument(f"--{option}-sha256", required=True)
    build.add_argument(
        "--physical-gpu", type=int, choices=SUPPORTED_PHYSICAL_GPUS, required=True
    )
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
    print(json.dumps(args.handler(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "EXPECTED_ACCESS_AUDIT",
    "METHOD_CONTRACT_SHA256",
    "RESULT_SCHEMA",
    "ROW_BATCH_SIZE",
    "materialize_scores_lowmem",
    "method_contract",
    "transport_descriptor_batch",
    "validate_runtime_device",
]
