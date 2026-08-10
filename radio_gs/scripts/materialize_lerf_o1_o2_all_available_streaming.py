#!/usr/bin/env python3
"""Materialize LERF O1/O2 from every available frozen source feature view."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from radio_gs.querying.all_available_source_view_authority import (
    file_record_value,
    load_reference_inputs,
    validate_supplemental_responsibility,
)
from radio_gs.querying.all_available_source_views import (
    validate_composite_frame_axis,
)
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as legacy
from radio_gs.scripts.materialize_lerf_o1_o2_streaming_unpaced_gpu1_lowmem import (
    TEACHER_MEAN_CHUNK_ROWS,
    _chunked_teacher_mean,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_o1_o2_all_available_streaming_execution.v1"
MEAN_SCHEMA = "radio_gs.lerf_source_teacher_mean_siglip_all_available.v1"
RESULT_SCHEMA = "radio_gs.lerf_o1_o2_all_available_streaming_result.v1"
SCHEMA_VERSION = 1


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_view_domain": "feature_manifest_order_minus_frozen_excluded_frames",
        "source_view_count": "runtime_exact_all_available_count",
        "legacy_default_and_contract_modified": False,
        "composition": "sealed_legacy120_plus_exact_omitted_only_supplement",
        "supplement_overlap_policy": "forbidden",
        "teacher_retention": "global_top4_exact_marginal_mass_desc_then_frame_id_asc",
        "teacher_aggregation": "equal_view_normalized_mean",
        "durable_teacher_dtype": "torch.float16",
        "teacher_mean_finalization": "row_chunked_fp32_canonical_top4_sum_v1",
        "teacher_mean_finalization_chunk_rows": TEACHER_MEAN_CHUNK_ROWS,
        "O1": {
            "operation": "closed_form_unit_sphere_geodesic_projection",
            "maximum_angle_radians": legacy.O1_MAXIMUM_ANGLE_RADIANS,
            "per_scale": True,
        },
        "O2": {
            "operation": "normalized_equal_view_teacher_mean",
            "repeated_scale_slots": 3,
        },
        "scale_ids": list(legacy.SCALE_IDS),
        "scale_radii_m": list(legacy.SCALE_RADII_M),
        "canonical_negative_queries": list(legacy.NEGATIVE_QUERIES),
        "score_dtype": "torch.float32",
        "score_semantics": "raw_independent_normalized_cosine",
        "per_scene_or_per_query_hyperparameters": False,
        "target_metric_execution_authorized": False,
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def access_audit() -> dict[str, bool]:
    return {
        "source_feature_bundle_opened": True,
        "legacy_source_responsibility_opened": True,
        "supplemental_source_responsibility_opened_if_required": True,
        "exact_query_axis_opened": True,
        "exact_o0_pair_opened": True,
        "target_images_opened": False,
        "target_ground_truth_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "target_quality_readout_executed": False,
    }


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def _outputs(output_root: Path, scene_id: str) -> dict[str, str]:
    names = {
        "teacher_mean": f"{scene_id}_all_available_teacher_mean_fp16.pt",
        "o1_positive": f"{scene_id}_all_available_o1_positive.pt",
        "o1_negative": f"{scene_id}_all_available_o1_negative.pt",
        "o2_positive": f"{scene_id}_all_available_o2_positive.pt",
        "o2_negative": f"{scene_id}_all_available_o2_negative.pt",
        "result": f"{scene_id}_all_available_o1_o2_result.json",
    }
    return {
        name: str(_new(output_root / filename, label=f"all-available {name}"))
        for name, filename in names.items()
    }


def _load_supplement(
    record: dict[str, str], *, reference: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    raw, _, source = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="all-available supplemental responsibility",
    )
    if not isinstance(raw, Mapping):
        raise ValueError("supplemental responsibility is not a mapping")
    validated = validate_supplemental_responsibility(
        raw,
        source_path=source,
        audit=reference["domain_audit"],
        reference=reference,
    )
    return validated, source


def prepare_inputs(
    path: str | Path,
    *,
    expected_sha256: str,
    load_tensor_payloads: bool = True,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="all-available O1/O2 execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "method_contract",
        "method_contract_sha256",
        "reference_execution_authority",
        "supplemental_responsibility_authority",
        "outputs",
        "execution",
        "source_only_materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("all-available O1/O2 authority fields differ")
    authority = dict(raw)
    implementation = file_record_value(
        authority["implementation"], label="all-available O1/O2 implementation"
    )
    reference_record = file_record_value(
        authority["reference_execution_authority"],
        label="all-available reference authority",
    )
    execution = authority.get("execution")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_all_available_o1_o2_streaming"
        or implementation != file_record(Path(__file__).resolve())
        or authority.get("method_contract") != method_contract()
        or authority.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or not isinstance(execution, Mapping)
        or set(execution)
        != {
            "physical_gpu",
            "cuda_visible_devices",
            "program_device",
            "projection_batch_candidates",
            "pacing_seconds_per_projection_batch",
            "thermal_poll_seconds",
            "maximum_temperature_c",
        }
        or int(execution.get("physical_gpu", -1)) not in (0, 1)
        or str(execution.get("cuda_visible_devices"))
        != str(execution.get("physical_gpu"))
        or execution.get("program_device") != "cuda:0"
        or list(execution.get("projection_batch_candidates", []))
        != list(legacy.PREFLIGHT_BATCH_CANDIDATES)
        or float(execution.get("pacing_seconds_per_projection_batch", -1)) != 0.0
        or int(execution.get("thermal_poll_seconds", -1)) != 300
        or int(execution.get("maximum_temperature_c", -1)) != 88
        or authority.get("source_only_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != access_audit()
    ):
        raise ValueError("all-available O1/O2 authority header differs")
    reference = load_reference_inputs(
        reference_record["path"],
        expected_sha256=reference_record["sha256"],
        load_tensor_payloads=load_tensor_payloads,
    )
    if authority.get("scene_id") != reference["authority"]["scene_id"]:
        raise ValueError("all-available scene differs from reference")

    supplement_value = authority.get("supplemental_responsibility_authority")
    supplemental_record: dict[str, str] | None
    supplemental: dict[str, Any] | None
    supplemental_path: Path | None
    if reference["domain_audit"].omitted_frames:
        supplemental_record = file_record_value(
            supplement_value, label="all-available supplemental responsibility"
        )
        supplemental, supplemental_path = _load_supplement(
            supplemental_record, reference=reference
        )
    else:
        if supplement_value is not None:
            raise ValueError("all-available control scene must not carry a supplement")
        supplemental_record = None
        supplemental = None
        supplemental_path = None
        validate_composite_frame_axis(reference["domain_audit"], [])

    outputs = authority.get("outputs")
    expected_output_names = {
        "teacher_mean",
        "o1_positive",
        "o1_negative",
        "o2_positive",
        "o2_negative",
        "result",
    }
    if not isinstance(outputs, Mapping) or set(outputs) != expected_output_names:
        raise ValueError("all-available output axis differs")
    resolved_outputs: dict[str, str] = {}
    for name, value in outputs.items():
        raw_output = str(value)
        resolved = str(Path(raw_output).expanduser().resolve())
        if raw_output != resolved:
            raise ValueError(f"all-available {name} output is not canonical")
        resolved_outputs[name] = resolved

    sources: dict[int, tuple[Mapping[str, Any], Path, Mapping[str, Any]]] = {}
    for record in reference["responsibility"]["views"]:
        sources[int(record["frame_index"])] = (
            reference["responsibility"],
            Path(reference["responsibility_path"]),
            record,
        )
    if supplemental is not None and supplemental_path is not None:
        for record in supplemental["views"]:
            frame = int(record["frame_index"])
            if frame in sources:
                raise ValueError("supplement overlaps legacy responsibility")
            sources[frame] = (supplemental, supplemental_path, record)
    combined_frames = validate_composite_frame_axis(
        reference["domain_audit"],
        [] if supplemental is None else supplemental["frame_indices"],
    )
    if set(sources) != set(combined_frames):
        raise ValueError("composite responsibility records are incomplete")
    composite_views = [sources[frame] for frame in combined_frames]
    return {
        **reference,
        "authority": authority,
        "authority_record": {"path": str(source), "sha256": digest},
        "reference_authority_record": reference_record,
        "legacy_authority": reference["authority"],
        "supplemental_record": supplemental_record,
        "supplemental_responsibility": supplemental,
        "composite_views": composite_views,
        "combined_frame_ids": combined_frames,
        "outputs": resolved_outputs,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="all-available authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("all-available output directory must be canonical absolute")
    reference_record = {
        "path": str(Path(args.reference_execution_authority).expanduser().resolve()),
        "sha256": str(args.reference_execution_authority_sha256),
    }
    reference = load_reference_inputs(
        reference_record["path"],
        expected_sha256=reference_record["sha256"],
        load_tensor_payloads=False,
    )
    supplement_path = str(args.supplemental_responsibility_authority or "")
    supplement_sha = str(args.supplemental_responsibility_authority_sha256 or "")
    if reference["domain_audit"].omitted_frames:
        if not supplement_path or not supplement_sha:
            raise ValueError("omitted source views require an exact supplement")
        supplement_record: dict[str, str] | None = {
            "path": str(Path(supplement_path).expanduser().resolve()),
            "sha256": supplement_sha,
        }
        _load_supplement(supplement_record, reference=reference)
    else:
        if supplement_path or supplement_sha:
            raise ValueError("all-covered control scene forbids a supplement")
        supplement_record = None
    physical_gpu = int(args.physical_gpu)
    if physical_gpu not in (0, 1):
        raise ValueError("physical GPU must be 0 or 1")
    scene_id = str(reference["authority"]["scene_id"])
    payload = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_only_all_available_o1_o2_streaming",
        "scene_id": scene_id,
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "reference_execution_authority": reference_record,
        "supplemental_responsibility_authority": supplement_record,
        "outputs": _outputs(output_root, scene_id),
        "execution": {
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": str(physical_gpu),
            "program_device": "cuda:0",
            "projection_batch_candidates": list(legacy.PREFLIGHT_BATCH_CANDIDATES),
            "pacing_seconds_per_projection_batch": 0.0,
            "thermal_poll_seconds": 300,
            "maximum_temperature_c": 88,
        },
        "source_only_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": access_audit(),
    }
    write_frozen_json(authority_output, payload)
    record = file_record(authority_output)
    prepared = prepare_inputs(
        record["path"],
        expected_sha256=record["sha256"],
        load_tensor_payloads=False,
    )
    return {
        "status": "authorized",
        "authority": record,
        "scene_id": scene_id,
        "legacy_frame_count": len(prepared["domain_audit"].legacy_frames),
        "supplemental_frame_count": len(prepared["domain_audit"].omitted_frames),
        "all_available_frame_count": len(prepared["combined_frame_ids"]),
        "outputs": prepared["outputs"],
    }


def _project_composite_view(
    *,
    prepared: Mapping[str, Any],
    item: tuple[Mapping[str, Any], Path, Mapping[str, Any]],
    global_to_accepted: torch.Tensor,
    head: torch.nn.Module,
    device: torch.device,
    projection_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    responsibility, path, record = item
    local = dict(prepared)
    local["responsibility"] = responsibility
    local["responsibility_path"] = path
    return legacy._project_view(
        prepared=local,
        record=record,
        global_to_accepted=global_to_accepted,
        head=head,
        device=device,
        projection_batch=projection_batch,
        pace=False,
    )


def _raw_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    oracle: str,
    representation: Mapping[str, str],
    text_cache: Mapping[str, str],
    descriptor_sha256: str,
) -> dict[str, Any]:
    payload = legacy._raw_cache(
        template,
        scores,
        oracle=oracle,
        representation=representation,
        text_cache=text_cache,
        descriptor_sha256=descriptor_sha256,
    )
    authority = copy.deepcopy(payload["authority"])
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["descriptor_axis"]["execution_representation"] = (
        "source_teacher_mean_all_available_streaming_v1"
    )
    authority["source_artifacts"]["materializer_source"] = file_record(
        Path(__file__).resolve()
    )
    payload["authority"] = authority
    return payload


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    execution = prepared["authority"]["execution"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != execution["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from all-available authority")
    if not torch.cuda.is_available():
        raise RuntimeError("all-available O1/O2 streaming requires CUDA")
    for name, path in prepared["outputs"].items():
        _new(path, label=f"all-available {name} output")
    device = torch.device("cuda:0")
    rows = prepared["rows"]
    n_rows = int(rows.numel())
    global_to_accepted = torch.full(
        (int(prepared["base"]["xyz"].shape[0]),), -1, dtype=torch.long
    )
    global_to_accepted[rows] = torch.arange(n_rows)
    head = legacy.SigLIP2SummaryHead.from_radio_checkpoint(
        prepared["records"]["official_radio_checkpoint"]["path"],
        expected_sha256=prepared["records"]["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    densest = max(
        prepared["composite_views"], key=lambda item: int(item[2]["num_hits"])
    )

    def preflight(candidate: int) -> int:
        torch.cuda.reset_peak_memory_stats(device)
        active, descriptors, mass = _project_composite_view(
            prepared=prepared,
            item=densest,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=candidate,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        del active, descriptors, mass
        torch.cuda.empty_cache()
        return peak

    projection_batch, preflight_peak = legacy.select_projection_batch(
        legacy.PREFLIGHT_BATCH_CANDIDATES, preflight
    )
    print(
        json.dumps(
            {
                "event": "preflight_pass",
                "selected_projection_batch": projection_batch,
                "peak_cuda_bytes": preflight_peak,
                "densest_frame_id": int(densest[2]["frame_index"]),
            }
        ),
        flush=True,
    )
    top_descriptors = torch.zeros(
        n_rows, legacy.TOP_VIEW_COUNT, 1536, dtype=torch.float16
    )
    top_mass = torch.zeros(n_rows, legacy.TOP_VIEW_COUNT, dtype=torch.float32)
    top_frame_ids = torch.full(
        (n_rows, legacy.TOP_VIEW_COUNT), -1, dtype=torch.int32
    )
    total_views = len(prepared["composite_views"])
    progress_step = max(1, total_views // 4)
    view_started = time.monotonic()
    for position, item in enumerate(prepared["composite_views"], start=1):
        active, descriptors, mass = _project_composite_view(
            prepared=prepared,
            item=item,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=projection_batch,
        )
        legacy._update_top_views(
            top_descriptors=top_descriptors,
            top_mass=top_mass,
            top_frame_ids=top_frame_ids,
            rows=active,
            descriptors=descriptors,
            mass=mass,
            frame_id=int(item[2]["frame_index"]),
        )
        if position % progress_step == 0 or position == total_views:
            elapsed = time.monotonic() - view_started
            print(
                json.dumps(
                    {
                        "event": "all_available_view_progress",
                        "views_complete": position,
                        "views_total": total_views,
                        "fraction": position / total_views,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / position * (total_views - position),
                    }
                ),
                flush=True,
            )
        del active, descriptors, mass
    top_descriptors, top_mass, top_frame_ids = legacy._canonicalize_view_axis(
        top_descriptors, top_mass, top_frame_ids
    )
    teacher_mean_half, teacher_valid, observed_counts = _chunked_teacher_mean(
        top_descriptors, top_frame_ids
    )
    del top_descriptors, top_mass, top_frame_ids
    mean_payload = {
        "schema": MEAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": prepared["authority"]["scene_id"],
        "global_rows": rows.clone(),
        "teacher_mean": teacher_mean_half,
        "teacher_valid": teacher_valid,
        "retained_view_count": observed_counts,
        "producer": file_record(Path(__file__).resolve()),
        "execution_authority": dict(prepared["authority_record"]),
        "input_authority": {
            "reference_execution_authority": dict(
                prepared["reference_authority_record"]
            ),
            "legacy_responsibility_authority": dict(
                prepared["records"]["responsibility_authority"]
            ),
            "supplemental_responsibility_authority": prepared[
                "supplemental_record"
            ],
            "feature_manifest": dict(prepared["records"]["feature_manifest"]),
            "official_radio_checkpoint": dict(
                prepared["records"]["official_radio_checkpoint"]
            ),
        },
        "all_available_frame_ids": list(prepared["combined_frame_ids"]),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "teacher_mean_sha256": legacy.tensor_sha256_typed(teacher_mean_half),
        "access_audit": access_audit(),
    }
    legacy.write_torch_noclobber(prepared["outputs"]["teacher_mean"], mean_payload)
    mean_record = file_record(prepared["outputs"]["teacher_mean"])

    base_features = prepared["base"]["features_by_scale"]
    positive_scores_o1 = prepared["o0_positive"]["query_scores"].clone()
    negative_scores_o1 = prepared["o0_negative"]["query_scores"].clone()
    positive_scores_o2 = positive_scores_o1.clone()
    negative_scores_o2 = negative_scores_o1.clone()
    positive_embeddings = prepared["positive_embeddings"].to(device)
    negative_embeddings = prepared["negative_embeddings"].to(device)
    o1_hasher = legacy._typed_stream_hasher((n_rows, 3, 1536), torch.float32)
    o2_hasher = legacy._typed_stream_hasher((n_rows, 1536), torch.float32)
    for start in range(0, n_rows, 256):
        stop = min(n_rows, start + 256)
        global_rows = rows[start:stop]
        o1, mean = legacy._score_descriptors(
            base=base_features[start:stop],
            teacher_mean=teacher_mean_half[start:stop],
            teacher_valid=teacher_valid[start:stop],
        )
        o1_hasher.update(o1.contiguous().numpy().tobytes(order="C"))
        o2_hasher.update(mean.contiguous().numpy().tobytes(order="C"))
        active = teacher_valid[start:stop]
        if bool(active.any()):
            selected_rows = global_rows[active]
            o1_active = o1[active].to(device)
            mean_active = mean[active].to(device)
            positive_scores_o1[selected_rows] = torch.einsum(
                "bsd,qd->bsq", o1_active, positive_embeddings
            ).cpu()
            negative_scores_o1[selected_rows] = torch.einsum(
                "bsd,qd->bsq", o1_active, negative_embeddings
            ).cpu()
            positive_scores_o2[selected_rows] = torch.einsum(
                "bd,qd->bq", mean_active, positive_embeddings
            )[:, None].expand(-1, 3, -1).cpu()
            negative_scores_o2[selected_rows] = torch.einsum(
                "bd,qd->bq", mean_active, negative_embeddings
            )[:, None].expand(-1, 3, -1).cpu()
    cache_specs = (
        ("o1_positive", prepared["o0_positive"], positive_scores_o1, "O1", prepared["records"]["positive_text"], o1_hasher.hexdigest()),
        ("o1_negative", prepared["o0_negative"], negative_scores_o1, "O1", prepared["records"]["negative_text"], o1_hasher.hexdigest()),
        ("o2_positive", prepared["o0_positive"], positive_scores_o2, "O2", prepared["records"]["positive_text"], o2_hasher.hexdigest()),
        ("o2_negative", prepared["o0_negative"], negative_scores_o2, "O2", prepared["records"]["negative_text"], o2_hasher.hexdigest()),
    )
    output_records: dict[str, dict[str, str]] = {"teacher_mean": mean_record}
    for name, template, scores, oracle, text_record, descriptor_sha in cache_specs:
        payload = _raw_cache(
            template,
            scores,
            oracle=oracle,
            representation=mean_record,
            text_cache=text_record,
            descriptor_sha256=descriptor_sha,
        )
        legacy.write_torch_noclobber(prepared["outputs"][name], payload)
        output_records[name] = file_record(prepared["outputs"][name])
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_premetric_all_available_o1_o2",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": dict(prepared["authority_record"]),
        "outputs": output_records,
        "accepted_rows": n_rows,
        "legacy_frame_count": len(prepared["domain_audit"].legacy_frames),
        "supplemental_frame_count": len(prepared["domain_audit"].omitted_frames),
        "all_available_frame_count": total_views,
        "rows_with_teacher": int(teacher_valid.sum()),
        "rows_without_teacher_bitwise_o0_fallback": int((~teacher_valid).sum()),
        "selected_projection_batch": projection_batch,
        "preflight_peak_cuda_bytes": preflight_peak,
        "elapsed_seconds": time.monotonic() - started,
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "next_gate": "source_only_ab_gate_before_any_target_metric_authority",
    }
    legacy.write_frozen_json(prepared["outputs"]["result"], result)
    return {**result, "result": file_record(prepared["outputs"]["result"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--reference-execution-authority", required=True)
    build.add_argument("--reference-execution-authority-sha256", required=True)
    build.add_argument("--supplemental-responsibility-authority")
    build.add_argument("--supplemental-responsibility-authority-sha256")
    build.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
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
    "AUTHORITY_SCHEMA",
    "MEAN_SCHEMA",
    "METHOD_CONTRACT_SHA256",
    "RESULT_SCHEMA",
    "access_audit",
    "method_contract",
    "prepare_inputs",
]
