#!/usr/bin/env python3
"""Regenerate exact source views and emit only transport-v2 scalar LOO data.

This entrypoint deliberately excludes text banks, query-score caches, frozen
target configs, images, masks, and metrics.  It reuses the exact 120-view
responsibility/projection lineage, retains canonical top four views only in
host memory, invokes the transport-v2 CPU hook, and durably writes one JSON
summary.  No per-view descriptor is written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from radio_gs.config import load_config
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.scripts import (
    materialize_lerf_transport_v2_source_loo_streaming_hook as _hook,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_transport_v2_source_loo_exact_execution.v1"
RESULT_SCHEMA = "radio_gs.lerf_transport_v2_source_loo_exact_result.v1"
SCHEMA_VERSION = 1
PHYSICAL_GPU = 1
CUDA_VISIBLE_DEVICES = "1"
PROGRAM_DEVICE = "cuda:0"
PROJECTION_BATCH_CANDIDATES = (128, 64)
ROW_CHUNK = 2048
P05_CANDIDATE_COUNT = 25
QUANTILE_WORKSPACE_MATRIX_MULTIPLIER = 2
MINIMUM_HOST_HEADROOM_BYTES = 8 * 1024**3


def access_audit() -> dict[str, bool]:
    return {
        "source_lineage_authority_opened": True,
        "source_base_descriptor_opened": True,
        "source_feature_bundle_opened": True,
        "source_responsibility_opened": True,
        "transport_v2_preregistration_opened": True,
        "query_embeddings_or_text_opened": False,
        "o0_query_score_cache_opened": False,
        "frozen_target_config_opened": False,
        "target_images_opened": False,
        "target_labels_or_masks_opened": False,
        "target_metrics_opened": False,
        "target_metric_executed": False,
    }


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_projection_core": file_record(Path(_core.__file__).resolve()),
        "transport_v2_streaming_hook": file_record(Path(_hook.__file__).resolve()),
        "transport_v2_hook_contract_sha256": _hook.HOOK_CONTRACT_SHA256,
        "source_view_count": _core.SOURCE_VIEW_COUNT,
        "teacher_retention": "top4_exact_marginal_mass_desc_then_frame_id_asc",
        "top4_descriptor_dtype": "torch.float16",
        "top4_descriptors_durable": False,
        "source_view_projection": "official_c_radio_v4_h_siglip2_summary_head",
        "source_loo": {
            "candidate_count": P05_CANDIDATE_COUNT,
            "statistics": ["mean_cosine", "exact_linear_p05_cosine"],
            "row_chunk": ROW_CHUNK,
            "compute_device": "cpu",
            "candidate_by_observation_matrix": "transient_float32",
        },
        "outputs": "one_scalar_only_json_result",
        "query_embeddings_or_text_consumed": False,
        "o0_query_score_cache_consumed": False,
        "target_images_labels_masks_metrics_consumed": False,
        "metric_execution_authorized": False,
        "target_candidate_authorized": False,
        "execution": {
            "physical_gpu": PHYSICAL_GPU,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "program_device": PROGRAM_DEVICE,
            "projection_batch_candidates": list(PROJECTION_BATCH_CANDIDATES),
            "projection_pacing_seconds": 0.0,
            "thermal_safety_owner": "external_300s_hard88_no_soft_guard",
            "gpu_owner_pid_namespace_mode": ("exclusive-singleton-after-clear-v1"),
        },
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def host_memory_preflight(accepted_rows: int) -> dict[str, Any]:
    if (
        not isinstance(accepted_rows, int)
        or isinstance(accepted_rows, bool)
        or accepted_rows < 1
    ):
        raise ValueError("transport-v2 source LOO accepted-row count differs")
    maximum_predictions = accepted_rows * _core.TOP_VIEW_COUNT
    maximum_observations = maximum_predictions * 3
    existing_top4_bytes = accepted_rows * _core.TOP_VIEW_COUNT * 1536 * 2
    scalar_matrix_bytes = P05_CANDIDATE_COUNT * maximum_observations * 4
    quantile_workspace_bytes = (
        QUANTILE_WORKSPACE_MATRIX_MULTIPLIER * scalar_matrix_bytes
    )
    chunk_bytes = ROW_CHUNK * _core.TOP_VIEW_COUNT * 1536 * 4
    additional_bytes = scalar_matrix_bytes + quantile_workspace_bytes + chunk_bytes
    return {
        "accepted_rows": accepted_rows,
        "maximum_heldout_predictions": maximum_predictions,
        "maximum_heldout_scale_observations": maximum_observations,
        "existing_lowmem_top4_fp16_bytes": existing_top4_bytes,
        "transport_v2_scalar_matrix_bytes": scalar_matrix_bytes,
        "quantile_workspace_upper_bound_bytes": quantile_workspace_bytes,
        "row_chunk_fp32_upper_bound_bytes": chunk_bytes,
        "transport_v2_additional_host_bytes_upper_bound": additional_bytes,
        "additional_below_existing_lowmem_top4_allocation": (
            additional_bytes < existing_top4_bytes
        ),
        "minimum_available_host_bytes_at_launch": (
            additional_bytes + MINIMUM_HOST_HEADROOM_BYTES
        ),
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _canonical_output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be canonical absolute")
    return resolved


def _available_host_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _validate_source_lineage(
    record: Mapping[str, str],
    source_inputs: Mapping[str, Mapping[str, str]],
    *,
    feature_output_bundle_sha256: str,
    scene_id: str,
) -> None:
    lineage, _, _ = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="transport-v2 historical source lineage",
    )
    lineage_inputs = lineage.get("inputs")
    lineage_contract = lineage.get("method_contract")
    if (
        lineage.get("scene_id") != scene_id
        or lineage.get("query_free_materialization_authorized") is not True
        or lineage.get("metric_execution_authorized") is not False
        or not isinstance(lineage_inputs, Mapping)
        or not isinstance(lineage_contract, Mapping)
        or lineage.get("method_contract_sha256")
        != canonical_json_sha256(lineage_contract)
        or lineage.get("feature_output_bundle_sha256") != feature_output_bundle_sha256
        or any(
            lineage_inputs.get(name) != value for name, value in source_inputs.items()
        )
    ):
        raise ValueError("transport-v2 historical source lineage differs")
    _record(lineage.get("implementation"), label="historical source implementation")
    for name in (
        "streaming_core_implementation",
        "streaming_entrypoint_implementation",
        "teacher_agreement_v2_numerical_implementation",
        "lowmem_allocation_reference_implementation",
    ):
        if name in lineage_contract:
            _record(lineage_contract[name], label=f"historical source {name}")


def prepare_inputs(
    authority_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="transport-v2 source LOO execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "method_contract",
        "method_contract_sha256",
        "source_lineage_authority",
        "transport_v2_preregistration",
        "feature_output_bundle_sha256",
        "source_inputs",
        "outputs",
        "execution",
        "access_audit",
        "metric_execution_authorized",
        "target_candidate_authorized",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("transport-v2 source LOO authority fields differ")
    authority = dict(raw)
    expected_execution = {
        "physical_gpu": PHYSICAL_GPU,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "program_device": PROGRAM_DEVICE,
        "projection_batch_candidates": list(PROJECTION_BATCH_CANDIDATES),
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
        "maximum_power_limit_w": 300.5,
        "gpu_owner_pid_namespace_mode": "exclusive-singleton-after-clear-v1",
    }
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status") != "authorized_ramen_source_only_transport_v2_loo"
        or authority.get("scene_id") != "ramen"
        or _record(authority.get("implementation"), label="transport-v2 implementation")
        != file_record(Path(__file__).resolve())
        or authority.get("method_contract") != method_contract()
        or authority.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or authority.get("execution") != expected_execution
        or authority.get("access_audit") != access_audit()
        or authority.get("metric_execution_authorized") is not False
        or authority.get("target_candidate_authorized") is not False
    ):
        raise ValueError("transport-v2 source LOO authority header differs")

    lineage_record = _record(
        authority.get("source_lineage_authority"),
        label="transport-v2 source lineage authority",
    )
    prereg_record = _record(
        authority.get("transport_v2_preregistration"),
        label="transport-v2 preregistration",
    )
    prereg, _, _ = load_json_object(
        prereg_record["path"],
        expected_sha256=prereg_record["sha256"],
        label="transport-v2 preregistration",
    )
    if (
        prereg.get("status")
        != "implementation_and_tests_complete_source_and_target_not_executed"
        or prereg.get("implementation", {}).get("residual_shrinkage_contract_sha256")
        != _hook.hook_contract()["residual_shrinkage_contract_sha256"]
        or prereg.get("implementation", {}).get("streaming_hook_contract_sha256")
        != _hook.HOOK_CONTRACT_SHA256
        or prereg.get("execution_state_at_seal", {}).get("source_loo_executed")
        is not False
        or prereg.get("execution_state_at_seal", {}).get("target_metric_executed")
        is not False
    ):
        raise ValueError("transport-v2 preregistration contract differs")

    source_inputs_raw = authority.get("source_inputs")
    expected_source_inputs = {
        "base_descriptor",
        "responsibility_authority",
        "feature_manifest",
        "scene_config",
        "renderer_geometry_checkpoint",
        "official_radio_checkpoint",
    }
    if (
        not isinstance(source_inputs_raw, Mapping)
        or set(source_inputs_raw) != expected_source_inputs
    ):
        raise ValueError("transport-v2 source inputs differ")
    records = {
        name: _record(source_inputs_raw[name], label=f"transport-v2 {name}")
        for name in sorted(expected_source_inputs)
    }
    feature_bundle_sha256 = str(authority.get("feature_output_bundle_sha256", ""))
    _validate_source_lineage(
        lineage_record,
        records,
        feature_output_bundle_sha256=feature_bundle_sha256,
        scene_id="ramen",
    )

    base, rows = _core._validate_base_descriptor_general(
        Path(records["base_descriptor"]["path"]),
        records["base_descriptor"]["sha256"],
    )
    feature_path = Path(records["feature_manifest"]["path"])
    if feature_path.name != "frame_manifest.json":
        raise ValueError("transport-v2 feature manifest name differs")
    feature_manifest, validation, tensor_records = _core._validated_feature_bundle(
        feature_path.parent,
        expected_output_bundle_sha256=feature_bundle_sha256,
    )
    if validation["manifest_sha256"] != records["feature_manifest"]["sha256"]:
        raise ValueError("transport-v2 feature manifest SHA differs")
    feature_frames = {int(item["frame_idx"]) for item in feature_manifest["frames"]}
    responsibility_raw, _, responsibility_path = load_json_object(
        records["responsibility_authority"]["path"],
        expected_sha256=records["responsibility_authority"]["sha256"],
        label="transport-v2 exact responsibility authority",
    )
    responsibility = _core._validate_responsibility_payload(
        responsibility_raw,
        descriptor_xyz_sha256=str(base["metadata"]["field_geometry_xyz_sha256"]),
        feature_frame_ids=feature_frames,
    )
    responsibility_root = Path(responsibility_path).parent
    for view in responsibility["views"]:
        sidecar = (responsibility_root / str(view["relative_path"])).resolve()
        if responsibility_root not in sidecar.parents:
            raise ValueError("transport-v2 responsibility sidecar escapes root")
        validate_file_record(
            {"path": str(sidecar), "sha256": str(view["sha256"])},
            label=f"transport-v2 responsibility view {view['view_index']}",
        )
    metadata = responsibility["metadata"]
    if (
        records["scene_config"]["path"] != str(Path(metadata["config"]).resolve())
        or records["renderer_geometry_checkpoint"]["path"]
        != str(Path(metadata["checkpoint"]).resolve())
        or records["renderer_geometry_checkpoint"]["sha256"]
        != metadata["geometry_checkpoint_sha256"]
        or int(responsibility["num_gaussians"]) != int(base["xyz"].shape[0])
        or int(responsibility["num_pixels"])
        != int(metadata["feature_height"]) * int(metadata["feature_width"])
        or base["metadata"].get("official_radio_checkpoint_sha256")
        != records["official_radio_checkpoint"]["sha256"]
    ):
        raise ValueError("transport-v2 source geometry lineage differs")
    for name in (
        "scene_config",
        "renderer_geometry_checkpoint",
        "official_radio_checkpoint",
    ):
        if sha256_file(records[name]["path"]) != records[name]["sha256"]:
            raise ValueError(f"transport-v2 {name} SHA differs")
    source_config = load_config(records["scene_config"]["path"])
    if (
        Path(str(getattr(source_config, "feature_dir", ""))).expanduser().resolve()
        != feature_path.parent
    ):
        raise ValueError("transport-v2 source config feature directory differs")

    outputs_raw = authority.get("outputs")
    if not isinstance(outputs_raw, Mapping) or set(outputs_raw) != {
        "result",
        "thermal_telemetry",
        "gpu_owner_audit",
    }:
        raise ValueError("transport-v2 source LOO outputs differ")
    outputs = {
        name: _canonical_output(value, label=f"transport-v2 {name} output")
        for name, value in outputs_raw.items()
    }
    memory = host_memory_preflight(int(rows.numel()))
    if memory["additional_below_existing_lowmem_top4_allocation"] is not True:
        raise RuntimeError("transport-v2 source LOO exceeds existing lowmem top4")
    return {
        "authority": authority,
        "authority_record": {"path": str(source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "feature_manifest": feature_manifest,
        "tensor_records": tensor_records,
        "responsibility": responsibility,
        "responsibility_path": responsibility_path,
        "outputs": outputs,
        "host_memory_preflight": memory,
    }


def validate_runtime_device() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise RuntimeError("transport-v2 source LOO requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("transport-v2 source LOO requires one visible CUDA device")
    return torch.device(PROGRAM_DEVICE)


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    result_path = Path(prepared["outputs"]["result"])
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError(f"transport-v2 source result exists: {result_path}")
    available_host_bytes = _available_host_bytes()
    required_host_bytes = int(
        prepared["host_memory_preflight"]["minimum_available_host_bytes_at_launch"]
    )
    if available_host_bytes < required_host_bytes:
        raise RuntimeError("transport-v2 source LOO host-memory preflight failed")
    device = validate_runtime_device()
    rows = prepared["rows"]
    n_rows = int(rows.numel())
    global_to_accepted = torch.full(
        (int(prepared["base"]["xyz"].shape[0]),), -1, dtype=torch.long
    )
    global_to_accepted[rows] = torch.arange(n_rows)
    head = (
        SigLIP2SummaryHead.from_radio_checkpoint(
            prepared["records"]["official_radio_checkpoint"]["path"],
            expected_sha256=prepared["records"]["official_radio_checkpoint"]["sha256"],
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    densest = max(
        prepared["responsibility"]["views"],
        key=lambda item: int(item["num_hits"]),
    )

    def gpu_preflight(candidate: int) -> int:
        torch.cuda.reset_peak_memory_stats(device)
        active, descriptors, mass = _core._project_view(
            prepared=prepared,
            record=densest,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=candidate,
            pace=False,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        del active, descriptors, mass
        torch.cuda.empty_cache()
        return peak

    projection_batch, preflight_peak = _core.select_projection_batch(
        PROJECTION_BATCH_CANDIDATES, gpu_preflight
    )
    print(
        json.dumps(
            {
                "event": "transport_v2_source_preflight_pass",
                "scene_id": "ramen",
                "selected_projection_batch": projection_batch,
                "peak_cuda_bytes": preflight_peak,
                "available_host_bytes": available_host_bytes,
                "host_memory_preflight": prepared["host_memory_preflight"],
            }
        ),
        flush=True,
    )

    top_descriptors = torch.zeros(
        n_rows, _core.TOP_VIEW_COUNT, 1536, dtype=torch.float16
    )
    top_mass = torch.zeros(n_rows, _core.TOP_VIEW_COUNT, dtype=torch.float32)
    top_frame_ids = torch.full((n_rows, _core.TOP_VIEW_COUNT), -1, dtype=torch.int32)
    view_started = time.monotonic()
    for position, record in enumerate(prepared["responsibility"]["views"], start=1):
        active, descriptors, mass = _core._project_view(
            prepared=prepared,
            record=record,
            global_to_accepted=global_to_accepted,
            head=head,
            device=device,
            projection_batch=projection_batch,
            pace=False,
        )
        _core._update_top_views(
            top_descriptors=top_descriptors,
            top_mass=top_mass,
            top_frame_ids=top_frame_ids,
            rows=active,
            descriptors=descriptors,
            mass=mass,
            frame_id=int(record["frame_index"]),
        )
        if position in _core.PROGRESS_VIEW_MILESTONES:
            elapsed = time.monotonic() - view_started
            print(
                json.dumps(
                    {
                        "event": "transport_v2_source_view_progress",
                        "views_complete": position,
                        "views_total": _core.SOURCE_VIEW_COUNT,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed
                        / position
                        * (_core.SOURCE_VIEW_COUNT - position),
                    }
                ),
                flush=True,
            )
        del active, descriptors, mass
    top_descriptors, top_mass, top_frame_ids = _core._canonicalize_view_axis(
        top_descriptors, top_mass, top_frame_ids
    )
    del head, global_to_accepted, top_mass
    torch.cuda.empty_cache()
    capture = _hook.capture_source_only_transport_v2_loo(
        scene_id="ramen",
        top_descriptors=top_descriptors,
        top_frame_ids=top_frame_ids,
        o0_descriptor_by_scale=prepared["base"]["features_by_scale"],
        row_chunk=ROW_CHUNK,
    )
    del top_descriptors, top_frame_ids
    audit = capture["source_only_loo_audit"]
    single_scene_eligible_indices = [
        index
        for index, candidate in enumerate(audit["candidate_grid"])
        if candidate["mean_delta_vs_baseline"] > 0.0
        and candidate["p05_nonregression_vs_baseline"] is True
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_ramen_source_only_transport_v2_loo",
        "scene_id": "ramen",
        "execution_authority": dict(prepared["authority_record"]),
        "accepted_rows": n_rows,
        "selected_projection_batch": projection_batch,
        "preflight_peak_cuda_bytes": preflight_peak,
        "available_host_bytes_at_launch": available_host_bytes,
        "host_memory_preflight": prepared["host_memory_preflight"],
        "source_loo_capture": capture,
        "single_scene_mean_improvement_and_p05_nonregression_candidate_indices": (
            single_scene_eligible_indices
        ),
        "cross_scene_gate_evaluated": False,
        "cross_scene_candidate_eligible": None,
        "target_candidate_authorized": False,
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "elapsed_seconds": time.monotonic() - started,
        "next_gate": "report_ramen_then_wait_before_teatime",
    }
    write_frozen_json(result_path, result)
    return {**result, "result": file_record(result_path)}


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_path = Path(args.authority_output).expanduser().resolve()
    if authority_path.exists() or authority_path.is_symlink():
        raise FileExistsError(f"transport-v2 authority exists: {authority_path}")
    lineage_record = {
        "path": str(Path(args.source_lineage_authority).expanduser().resolve()),
        "sha256": str(args.source_lineage_authority_sha256),
    }
    lineage, _, _ = load_json_object(
        lineage_record["path"],
        expected_sha256=lineage_record["sha256"],
        label="transport-v2 source lineage authority",
    )
    lineage_inputs = lineage.get("inputs")
    source_names = {
        "base_descriptor",
        "responsibility_authority",
        "feature_manifest",
        "scene_config",
        "renderer_geometry_checkpoint",
        "official_radio_checkpoint",
    }
    if not isinstance(lineage_inputs, Mapping) or not source_names <= set(
        lineage_inputs
    ):
        raise ValueError("transport-v2 source lineage inputs differ")
    source_inputs = {name: dict(lineage_inputs[name]) for name in sorted(source_names)}
    output_dir = Path(args.output_dir).expanduser().resolve()
    if str(output_dir) != str(args.output_dir):
        raise ValueError("transport-v2 output directory must be canonical absolute")
    outputs = {
        "result": str(output_dir / "ramen_transport_v2_source_loo_result.json"),
        "thermal_telemetry": str(output_dir / "ramen_transport_v2_gpu1_telemetry.csv"),
        "gpu_owner_audit": str(output_dir / "ramen_transport_v2_gpu1_owner_audit.csv"),
    }
    for name, value in outputs.items():
        path = Path(value)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"transport-v2 {name} output exists: {path}")
    contract = method_contract()
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_ramen_source_only_transport_v2_loo",
        "scene_id": "ramen",
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": contract,
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "source_lineage_authority": lineage_record,
        "transport_v2_preregistration": {
            "path": str(Path(args.transport_v2_preregistration).expanduser().resolve()),
            "sha256": str(args.transport_v2_preregistration_sha256),
        },
        "feature_output_bundle_sha256": str(
            lineage.get("feature_output_bundle_sha256", "")
        ),
        "source_inputs": source_inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": PHYSICAL_GPU,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "program_device": PROGRAM_DEVICE,
            "projection_batch_candidates": list(PROJECTION_BATCH_CANDIDATES),
            "thermal_poll_seconds": 300,
            "soft_pause_temperature_c": 0,
            "maximum_temperature_c": 88,
            "maximum_power_limit_w": 300.5,
            "gpu_owner_pid_namespace_mode": ("exclusive-singleton-after-clear-v1"),
        },
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
        "target_candidate_authorized": False,
    }
    write_frozen_json(authority_path, authority)
    record = file_record(authority_path)
    prepared = prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {
        "status": "authorized_and_recursively_validated",
        "authority": record,
        "outputs": outputs,
        "host_memory_preflight": prepared["host_memory_preflight"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--source-lineage-authority", required=True)
    build.add_argument("--source-lineage-authority-sha256", required=True)
    build.add_argument("--transport-v2-preregistration", required=True)
    build.add_argument("--transport-v2-preregistration-sha256", required=True)
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
    "METHOD_CONTRACT_SHA256",
    "RESULT_SCHEMA",
    "access_audit",
    "build_authority",
    "host_memory_preflight",
    "materialize",
    "method_contract",
    "prepare_inputs",
    "validate_runtime_device",
]
