#!/usr/bin/env python3
"""Materialize teacher-agreement v2 with bounded host finalization memory.

This independent GPU0 entrypoint preserves the frozen teacher-agreement-v2
method.  Canonical top-four selection, directional resultant, the source-only
leave-one-view-out audit, O1/O2 construction, and score materialization all
delegate to their hash-bound implementations.  The only change is allocation
schedule: the teacher mean promotes, sums, normalizes, and stores one bounded
row chunk at a time instead of promoting the scene-wide top-four tensor.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.scripts import (
    materialize_lerf_o1_o2_streaming_unpaced_gpu1_lowmem as _lowmem_reference,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as _v2,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


CUDA_VISIBLE_DEVICES = "0"
TEACHER_MEAN_CHUNK_ROWS = 4096
ENTRYPOINT_IMPLEMENTATION = file_record(Path(__file__).resolve())
CORE_IMPLEMENTATION = file_record(Path(_core.__file__).resolve())
TEACHER_AGREEMENT_V2_IMPLEMENTATION = file_record(Path(_v2.__file__).resolve())
LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION = file_record(
    Path(_lowmem_reference.__file__).resolve()
)
_BOUND_CORE_PATH = Path(CORE_IMPLEMENTATION["path"])

# Capture untouched callables before installation mutates the shared core
# module in this private entrypoint process.
_CORE_PREPARE_INPUTS = _core.prepare_inputs
_CORE_BUILD_AUTHORITY = _core.build_authority
_CORE_PROJECT_VIEW = _core._project_view
_CORE_CANONICALIZE_VIEW_AXIS = _core._canonicalize_view_axis
_CORE_RAW_CACHE = _core._raw_cache


def method_contract() -> dict[str, Any]:
    """Bind v2 method numerics and the low-memory allocation schedule."""

    contract = copy.deepcopy(_v2.method_contract())
    contract.update(
        {
            "streaming_entrypoint_implementation": dict(
                ENTRYPOINT_IMPLEMENTATION
            ),
            "streaming_core_implementation": dict(CORE_IMPLEMENTATION),
            "teacher_agreement_v2_numerical_implementation": dict(
                TEACHER_AGREEMENT_V2_IMPLEMENTATION
            ),
            "lowmem_allocation_reference_implementation": dict(
                LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION
            ),
            "projection_pacing_seconds_per_batch": 0.0,
            "projection_pacing_affects_method_numerics": False,
            "thermal_safety_owner": "external_300s_hard88_guard",
            "execution_device_authority": {
                "implemented_physical_gpu": 0,
                "required_cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                "program_device": "cuda:0",
                "other_physical_gpu_authorized": False,
            },
            "teacher_mean_finalization": (
                "row_chunked_fp32_canonical_top4_sum_normalize_fp16_v1"
            ),
            "teacher_mean_finalization_chunk_rows": TEACHER_MEAN_CHUNK_ROWS,
            "teacher_mean_chunking_affects_method_numerics": False,
            "teacher_mean_chunking_changes_only_allocation_schedule": True,
            "agreement_and_loo_implementation_reused_without_change": True,
        }
    )
    return contract


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def _validate_local_contract() -> None:
    """Fail closed if this entrypoint or any bound upstream file changed."""

    if (
        METHOD_CONTRACT_SHA256 != canonical_json_sha256(method_contract())
        or ENTRYPOINT_IMPLEMENTATION != file_record(Path(__file__).resolve())
        or CORE_IMPLEMENTATION != file_record(_BOUND_CORE_PATH)
        or TEACHER_AGREEMENT_V2_IMPLEMENTATION
        != file_record(Path(_v2.__file__).resolve())
        or LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION
        != file_record(Path(_lowmem_reference.__file__).resolve())
    ):
        raise RuntimeError("teacher-agreement v2 low-memory contract differs")


def _chunked_teacher_mean(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    *,
    chunk_rows: int = TEACHER_MEAN_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the frozen dense teacher expression row-for-row and bit-for-bit."""

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    if (
        descriptors.ndim != 3
        or descriptors.shape[1] != _core.TOP_VIEW_COUNT
        or descriptors.shape[2] != 1536
        or descriptors.dtype != torch.float16
        or frame_ids.shape != descriptors.shape[:2]
        or frame_ids.dtype == torch.bool
        or not bool(torch.isfinite(descriptors).all())
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunked teacher mean inputs differ")
    mask = frame_ids >= 0
    if bool((descriptors[~mask] != 0).any()):
        raise ValueError("unretained teacher descriptor must be exact zero")
    counts = mask.sum(dim=1)
    valid = counts > 0
    mean_half = torch.empty(
        descriptors.shape[0], descriptors.shape[2], dtype=torch.float16
    )
    for start in range(0, descriptors.shape[0], chunk_rows):
        stop = min(descriptors.shape[0], start + chunk_rows)
        chunk = F.normalize(
            (
                descriptors[start:stop].float()
                * mask[start:stop, :, None]
            ).sum(dim=1),
            dim=-1,
        )
        chunk[~valid[start:stop]] = 0
        mean_half[start:stop] = chunk.half()
    return (
        mean_half.contiguous(),
        valid.contiguous(),
        counts.to(torch.uint8).contiguous(),
    )


def finalize_teacher_statistics_lowmem(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    *,
    chunk_rows: int = TEACHER_MEAN_CHUNK_ROWS,
) -> dict[str, Any]:
    """Compute the exact v2 statistics without scene-wide FP32 descriptors."""

    agreement, agreement_counts = (
        _v2.directional_resultant_from_canonical_top_views(
            top_descriptors, top_frame_ids
        )
    )
    loo_audit = _v2.source_only_leave_one_view_out_ceiling_audit(
        top_descriptors, top_frame_ids, o0_descriptor_by_scale
    )
    teacher_mean, teacher_valid, retained_count = _chunked_teacher_mean(
        top_descriptors, top_frame_ids, chunk_rows=chunk_rows
    )
    if not torch.equal(agreement_counts, retained_count):
        raise RuntimeError("teacher agreement and mean retained counts differ")
    return {
        "teacher_mean": teacher_mean,
        "teacher_valid": teacher_valid,
        "retained_view_count": retained_count,
        _v2.VIEW_AGREEMENT_SCALAR: agreement,
        _v2.LOO_AUDIT_FIELD: loo_audit,
    }


def validate_teacher_payload_lowmem(payload: Mapping[str, Any]) -> None:
    """Apply the full v2 validator plus this entrypoint's lineage contract."""

    if (
        payload.get("producer") != ENTRYPOINT_IMPLEMENTATION
        or payload.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
    ):
        raise ValueError("teacher-agreement v2 low-memory payload lineage differs")
    surrogate = dict(payload)
    surrogate["producer"] = dict(_v2.ENTRYPOINT_IMPLEMENTATION)
    surrogate["method_contract_sha256"] = _v2.METHOD_CONTRACT_SHA256
    _v2.validate_teacher_payload_v2(surrogate)


def _raw_cache_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _CORE_RAW_CACHE(*args, **kwargs)
    payload["authority"]["descriptor_axis"]["execution_representation"] = (
        "source_teacher_mean_with_directional_resultant_streaming_v2"
    )
    return payload


@torch.inference_mode()
def materialize(args: Any) -> dict[str, Any]:
    """Run the frozen v2 method on physical GPU0 with bounded CPU memory."""

    _validate_local_contract()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "teacher-agreement v2 low-memory materialization requires "
            "CUDA_VISIBLE_DEVICES=0"
        )
    started = time.monotonic()
    prepared = _CORE_PREPARE_INPUTS(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    outputs = prepared["outputs"]
    for name, path in outputs.items():
        _core._new(path, label=f"O1/O2 {name} output")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-agreement v2 streaming requires GPU0")
    config = _core.load_config(prepared["records"]["scene_config"]["path"])
    expected_feature_dir = Path(
        prepared["records"]["feature_manifest"]["path"]
    ).parent
    if (
        Path(str(getattr(config, "feature_dir", ""))).expanduser().resolve()
        != expected_feature_dir
    ):
        raise ValueError("O1/O2 scene config feature directory differs")
    rows = prepared["rows"]
    n_rows = int(rows.numel())
    global_to_accepted = torch.full(
        (int(prepared["base"]["xyz"].shape[0]),), -1, dtype=torch.long
    )
    global_to_accepted[rows] = torch.arange(n_rows)
    head = _core.SigLIP2SummaryHead.from_radio_checkpoint(
        prepared["records"]["official_radio_checkpoint"]["path"],
        expected_sha256=prepared["records"]["official_radio_checkpoint"][
            "sha256"
        ],
    ).to(device).eval().requires_grad_(False)
    densest = max(
        prepared["responsibility"]["views"],
        key=lambda row: int(row["num_hits"]),
    )

    def preflight(candidate: int) -> int:
        torch.cuda.reset_peak_memory_stats(device)
        active, descriptors, mass = _CORE_PROJECT_VIEW(
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
        _core.PREFLIGHT_BATCH_CANDIDATES, preflight
    )
    print(
        json.dumps(
            {
                "event": "preflight_pass",
                "selected_projection_batch": projection_batch,
                "peak_cuda_bytes": preflight_peak,
                "densest_frame_id": int(densest["frame_index"]),
            }
        ),
        flush=True,
    )

    top_descriptors = torch.zeros(
        n_rows, _core.TOP_VIEW_COUNT, 1536, dtype=torch.float16
    )
    top_mass = torch.zeros(n_rows, _core.TOP_VIEW_COUNT, dtype=torch.float32)
    top_frame_ids = torch.full(
        (n_rows, _core.TOP_VIEW_COUNT), -1, dtype=torch.int32
    )
    view_started = time.monotonic()
    for position, record in enumerate(
        prepared["responsibility"]["views"], start=1
    ):
        active, descriptors, mass = _CORE_PROJECT_VIEW(
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
            eta = elapsed / position * (_core.SOURCE_VIEW_COUNT - position)
            print(
                json.dumps(
                    {
                        "event": "source_view_progress",
                        "views_complete": position,
                        "views_total": _core.SOURCE_VIEW_COUNT,
                        "fraction": position / _core.SOURCE_VIEW_COUNT,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
                ),
                flush=True,
            )
        del active, descriptors, mass

    top_descriptors, top_mass, top_frame_ids = _CORE_CANONICALIZE_VIEW_AXIS(
        top_descriptors, top_mass, top_frame_ids
    )
    statistics = finalize_teacher_statistics_lowmem(
        top_descriptors,
        top_frame_ids,
        prepared["base"]["features_by_scale"],
    )
    del top_descriptors, top_mass, top_frame_ids
    teacher_mean_half = statistics["teacher_mean"]
    teacher_valid = statistics["teacher_valid"]
    observed_counts = statistics["retained_view_count"]
    agreement = statistics[_v2.VIEW_AGREEMENT_SCALAR]
    loo_audit = statistics[_v2.LOO_AUDIT_FIELD]
    mean_payload = {
        "schema": _v2.MEAN_SCHEMA,
        "schema_version": _v2.SCHEMA_VERSION,
        "scene_id": prepared["authority"]["scene_id"],
        "global_rows": rows.clone(),
        "teacher_mean": teacher_mean_half,
        "teacher_valid": teacher_valid,
        "retained_view_count": observed_counts,
        _v2.VIEW_AGREEMENT_SCALAR: agreement,
        "producer": dict(ENTRYPOINT_IMPLEMENTATION),
        "execution_authority": dict(prepared["authority_record"]),
        "input_authority": {
            "base_descriptor": dict(prepared["records"]["base_descriptor"]),
            "responsibility_authority": dict(
                prepared["records"]["responsibility_authority"]
            ),
            "feature_manifest": dict(prepared["records"]["feature_manifest"]),
            "official_radio_checkpoint": dict(
                prepared["records"]["official_radio_checkpoint"]
            ),
        },
        "method_contract_sha256": prepared["authority"][
            "method_contract_sha256"
        ],
        "teacher_mean_sha256": _core.tensor_sha256_typed(teacher_mean_half),
        _v2.VIEW_AGREEMENT_SHA256_FIELD: _core.tensor_sha256_typed(agreement),
        _v2.LOO_AUDIT_FIELD: loo_audit,
        _v2.LOO_AUDIT_SHA256_FIELD: canonical_json_sha256(loo_audit),
        "access_audit": _core.access_audit(),
    }
    validate_teacher_payload_lowmem(mean_payload)
    _core.write_torch_noclobber(outputs["teacher_mean"], mean_payload)
    mean_record = file_record(outputs["teacher_mean"])
    print(
        json.dumps(
            {
                "event": "teacher_agreement_v2_lowmem_complete",
                "rows": n_rows,
                "chunk_rows": TEACHER_MEAN_CHUNK_ROWS,
            }
        ),
        flush=True,
    )

    base_features = prepared["base"]["features_by_scale"]
    positive_scores_o1 = prepared["o0_positive"]["query_scores"].clone()
    negative_scores_o1 = prepared["o0_negative"]["query_scores"].clone()
    positive_scores_o2 = positive_scores_o1.clone()
    negative_scores_o2 = negative_scores_o1.clone()
    positive_embeddings = prepared["positive_embeddings"].to(device)
    negative_embeddings = prepared["negative_embeddings"].to(device)
    o1_hasher = _core._typed_stream_hasher((n_rows, 3, 1536), torch.float32)
    o2_hasher = _core._typed_stream_hasher((n_rows, 1536), torch.float32)
    score_batch = 256
    for start in range(0, n_rows, score_batch):
        stop = min(n_rows, start + score_batch)
        global_rows = rows[start:stop]
        o1, mean = _core._score_descriptors(
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
            o2_positive = torch.einsum(
                "bd,qd->bq", mean_active, positive_embeddings
            )[:, None].expand(-1, 3, -1)
            o2_negative = torch.einsum(
                "bd,qd->bq", mean_active, negative_embeddings
            )[:, None].expand(-1, 3, -1)
            positive_scores_o2[selected_rows] = o2_positive.cpu()
            negative_scores_o2[selected_rows] = o2_negative.cpu()
    o1_descriptor_sha = o1_hasher.hexdigest()
    o2_descriptor_sha = o2_hasher.hexdigest()
    cache_specs = (
        (
            "o1_positive", prepared["o0_positive"], positive_scores_o1,
            "O1", prepared["records"]["positive_text"], o1_descriptor_sha,
        ),
        (
            "o1_negative", prepared["o0_negative"], negative_scores_o1,
            "O1", prepared["records"]["negative_text"], o1_descriptor_sha,
        ),
        (
            "o2_positive", prepared["o0_positive"], positive_scores_o2,
            "O2", prepared["records"]["positive_text"], o2_descriptor_sha,
        ),
        (
            "o2_negative", prepared["o0_negative"], negative_scores_o2,
            "O2", prepared["records"]["negative_text"], o2_descriptor_sha,
        ),
    )
    output_records: dict[str, dict[str, str]] = {"teacher_mean": mean_record}
    for name, template, scores, oracle, text_record, descriptor_sha in cache_specs:
        payload = _raw_cache_v2(
            template,
            scores,
            oracle=oracle,
            representation=mean_record,
            text_cache=text_record,
            descriptor_sha256=descriptor_sha,
        )
        _core.write_torch_noclobber(outputs[name], payload)
        output_records[name] = file_record(outputs[name])
    result = {
        "schema": _v2.RESULT_SCHEMA,
        "schema_version": _v2.SCHEMA_VERSION,
        "status": "complete_source_only_premetric_o1_o2_streaming",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": dict(prepared["authority_record"]),
        "outputs": output_records,
        "accepted_rows": n_rows,
        "rows_with_teacher": int(teacher_valid.sum()),
        "rows_without_teacher_bitwise_o0_fallback": int((~teacher_valid).sum()),
        "selected_projection_batch": projection_batch,
        "preflight_peak_cuda_bytes": preflight_peak,
        "elapsed_seconds": time.monotonic() - started,
        "method_contract_sha256": prepared["authority"][
            "method_contract_sha256"
        ],
        "access_audit": _core.access_audit(),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "next_gate": (
            "source_only_global_ceiling_preregistration_without_target_metrics"
        ),
    }
    _core.write_frozen_json(outputs["result"], result)
    return {**result, "result": file_record(outputs["result"])}


def _install_lowmem_v2_contract() -> None:
    _validate_local_contract()
    _core.AUTHORITY_SCHEMA = _v2.AUTHORITY_SCHEMA
    _core.MEAN_SCHEMA = _v2.MEAN_SCHEMA
    _core.RESULT_SCHEMA = _v2.RESULT_SCHEMA
    _core.SCHEMA_VERSION = _v2.SCHEMA_VERSION
    _core.PACING_SECONDS_PER_PROJECTION_BATCH = 0.0
    _core.method_contract = method_contract
    _core.METHOD_CONTRACT_SHA256 = METHOD_CONTRACT_SHA256
    _core.prepare_inputs = _CORE_PREPARE_INPUTS
    _core.build_authority = _CORE_BUILD_AUTHORITY
    _core.materialize = materialize
    _core.__file__ = str(Path(__file__).resolve())


def main() -> None:
    _install_lowmem_v2_contract()
    _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "CORE_IMPLEMENTATION",
    "CUDA_VISIBLE_DEVICES",
    "ENTRYPOINT_IMPLEMENTATION",
    "LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION",
    "METHOD_CONTRACT_SHA256",
    "TEACHER_AGREEMENT_V2_IMPLEMENTATION",
    "TEACHER_MEAN_CHUNK_ROWS",
    "finalize_teacher_statistics_lowmem",
    "main",
    "materialize",
    "method_contract",
    "validate_teacher_payload_lowmem",
]
