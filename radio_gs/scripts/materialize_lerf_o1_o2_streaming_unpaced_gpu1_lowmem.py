#!/usr/bin/env python3
"""Run Waldo-sized O1/O2 streaming with row-chunked CPU finalization.

This entrypoint binds the already frozen numerical core, no-pacing wrapper,
and explicit host-GPU1 wrapper.  It changes only the allocation schedule of
the per-row teacher mean: each row still sums its canonical top-four FP32
descriptors in the same order, uses the same ``F.normalize`` semantics, and
is stored as FP16.  Chunking avoids simultaneous scene-wide FP32 temporaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.scripts import (
    materialize_lerf_o1_o2_streaming_unpaced_gpu1 as _gpu1,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


GPU1_IMPLEMENTATION = file_record(Path(_gpu1.__file__).resolve())
TEACHER_MEAN_CHUNK_ROWS = 4096
_GPU1_PREPARE_INPUTS = _gpu1.prepare_inputs
_GPU1_BUILD_AUTHORITY = _gpu1.build_authority
_GPU1_METHOD_CONTRACT = _gpu1.method_contract


def method_contract() -> dict[str, Any]:
    contract = dict(_GPU1_METHOD_CONTRACT())
    contract.update(
        {
            "gpu1_streaming_entrypoint": dict(GPU1_IMPLEMENTATION),
            "teacher_mean_finalization": "row_chunked_fp32_top4_sum_v1",
            "teacher_mean_finalization_chunk_rows": TEACHER_MEAN_CHUNK_ROWS,
            "teacher_mean_chunking_affects_method_numerics": False,
            "teacher_mean_chunking_changes_only_allocation_schedule": True,
        }
    )
    return contract


def _chunked_teacher_mean(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    *,
    chunk_rows: int = TEACHER_MEAN_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the frozen dense expression without scene-wide FP32 temporaries."""

    if (
        top_descriptors.ndim != 3
        or top_descriptors.shape[1] != _core.TOP_VIEW_COUNT
        or top_descriptors.dtype != torch.float16
        or top_frame_ids.shape != top_descriptors.shape[:2]
        or chunk_rows < 1
    ):
        raise ValueError("chunked teacher mean inputs differ")
    teacher_mask = top_frame_ids >= 0
    observed_counts = teacher_mask.sum(dim=1)
    teacher_valid = observed_counts > 0
    n_rows, _, dimension = top_descriptors.shape
    teacher_mean_half = torch.empty(
        int(n_rows), int(dimension), dtype=torch.float16
    )
    for start in range(0, int(n_rows), int(chunk_rows)):
        stop = min(int(n_rows), start + int(chunk_rows))
        chunk_mask = teacher_mask[start:stop]
        mean_float = F.normalize(
            (
                top_descriptors[start:stop].float()
                * chunk_mask[:, :, None]
            ).sum(dim=1),
            dim=-1,
        )
        mean_float[~teacher_valid[start:stop]] = 0
        teacher_mean_half[start:stop] = mean_float.half()
    return (
        teacher_mean_half.contiguous(),
        teacher_valid.contiguous(),
        observed_counts.to(torch.uint8).contiguous(),
    )


@torch.inference_mode()
def materialize(args: Any) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != _gpu1.CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "GPU1 low-memory O1/O2 materialization requires "
            "CUDA_VISIBLE_DEVICES=1"
        )
    started = time.monotonic()
    prepared = _GPU1_PREPARE_INPUTS(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    outputs = prepared["outputs"]
    for name, path in outputs.items():
        _core._new(path, label=f"O1/O2 {name} output")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("O1/O2 streaming requires CUDA_VISIBLE_DEVICES=1")
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
        expected_sha256=prepared["records"]["official_radio_checkpoint"]["sha256"],
    ).to(device).eval().requires_grad_(False)
    densest = max(
        prepared["responsibility"]["views"],
        key=lambda row: int(row["num_hits"]),
    )

    def preflight(candidate: int) -> int:
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
    top_descriptors, top_mass, top_frame_ids = _core._canonicalize_view_axis(
        top_descriptors, top_mass, top_frame_ids
    )
    teacher_mean_half, teacher_valid, observed_counts = _chunked_teacher_mean(
        top_descriptors, top_frame_ids
    )
    del top_descriptors, top_mass, top_frame_ids
    mean_payload = {
        "schema": _core.MEAN_SCHEMA,
        "schema_version": _core.SCHEMA_VERSION,
        "scene_id": prepared["authority"]["scene_id"],
        "global_rows": rows.clone(),
        "teacher_mean": teacher_mean_half,
        "teacher_valid": teacher_valid,
        "retained_view_count": observed_counts,
        "producer": file_record(Path(__file__).resolve()),
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
        "access_audit": _core.access_audit(),
    }
    _core.write_torch_noclobber(outputs["teacher_mean"], mean_payload)
    mean_record = file_record(outputs["teacher_mean"])
    print(
        json.dumps(
            {
                "event": "teacher_mean_lowmem_complete",
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
        ("o1_positive", prepared["o0_positive"], positive_scores_o1, "O1", prepared["records"]["positive_text"], o1_descriptor_sha),
        ("o1_negative", prepared["o0_negative"], negative_scores_o1, "O1", prepared["records"]["negative_text"], o1_descriptor_sha),
        ("o2_positive", prepared["o0_positive"], positive_scores_o2, "O2", prepared["records"]["positive_text"], o2_descriptor_sha),
        ("o2_negative", prepared["o0_negative"], negative_scores_o2, "O2", prepared["records"]["negative_text"], o2_descriptor_sha),
    )
    output_records: dict[str, dict[str, str]] = {"teacher_mean": mean_record}
    for name, template, scores, oracle, text_record, descriptor_sha in cache_specs:
        payload = _core._raw_cache(
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
        "schema": _core.RESULT_SCHEMA,
        "schema_version": _core.SCHEMA_VERSION,
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
        "next_gate": "independent_exact_o0_control_replay_before_any_metric_authority",
    }
    _core.write_frozen_json(outputs["result"], result)
    return {**result, "result": file_record(outputs["result"])}


def _install_lowmem_contract() -> None:
    _gpu1._install_gpu1_contract()
    _gpu1.method_contract = method_contract
    _gpu1.prepare_inputs = _GPU1_PREPARE_INPUTS
    _gpu1.__file__ = str(Path(__file__).resolve())
    _core.method_contract = method_contract
    _core.METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())
    _core.prepare_inputs = _GPU1_PREPARE_INPUTS
    _core.build_authority = _GPU1_BUILD_AUTHORITY
    _core.materialize = materialize
    _core.__file__ = str(Path(__file__).resolve())


def main() -> None:
    _install_lowmem_contract()
    _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "GPU1_IMPLEMENTATION",
    "TEACHER_MEAN_CHUNK_ROWS",
    "main",
    "materialize",
    "method_contract",
]
