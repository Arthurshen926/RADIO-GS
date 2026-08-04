#!/usr/bin/env python3
"""Derive a disposable text-space cache from the canonical field and v3 readout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import surface_region_geometry_v2
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_canonical_primitive_semantic_cache import (
    canonical_reconstruction_confidence,
)
from radio_gs.scripts.build_primitive_text_score_cache import (
    apply_completion_evidence,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    load_torch_payload,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _atomic_torch_save(payload: object, output: Path) -> None:
    """Durably publish a first-writer-wins cache without replacement."""

    write_torch_noclobber(output, payload)


class _ResumeStateError(ValueError):
    pass


def _resume_failure(resume_dir: Path, reason: str) -> _ResumeStateError:
    quarantine = resume_dir.with_name(f"{resume_dir.name}.quarantine-required")
    return _ResumeStateError(
        f"stale/corrupt semantic resume state: {reason}; "
        f"quarantine path: {quarantine} (not deleted automatically)"
    )


def _load_or_create_resume_contract(
    resume_dir: Path,
    payload: Mapping[str, Any],
) -> str:
    resume_dir.mkdir(parents=True, exist_ok=True)
    contract_path = resume_dir / "contract.json"
    expected = dict(payload)
    expected_digest = canonical_json_sha256(expected)
    if contract_path.exists() or contract_path.is_symlink():
        try:
            observed, _, _ = load_json_object(
                contract_path,
                label="semantic resume contract",
            )
        except Exception as exc:
            raise _resume_failure(resume_dir, "contract cannot be reopened") from exc
        if observed != expected:
            raise _resume_failure(resume_dir, "contract differs from this stage")
    else:
        write_frozen_json(contract_path, expected)
    return expected_digest


def _resume_paths(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
) -> tuple[Path, Path]:
    stem = f"{phase}_{int(start):09d}_{int(stop):09d}"
    return resume_dir / f"{stem}.pt", resume_dir / f"{stem}.complete.json"


def _load_resume_tensor(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
    contract_sha256: str,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor | None:
    shard, terminal = _resume_paths(
        resume_dir,
        phase=phase,
        start=start,
        stop=stop,
    )
    shard_present = shard.exists() or shard.is_symlink()
    terminal_present = terminal.exists() or terminal.is_symlink()
    if not shard_present and not terminal_present:
        return None
    if shard_present != terminal_present:
        raise _resume_failure(resume_dir, f"partial {phase} batch {start}:{stop}")
    try:
        marker, _, _ = load_json_object(terminal, label=f"{phase} resume terminal")
        if (
            set(marker)
            != {
                "schema_version",
                "artifact_type",
                "phase",
                "start",
                "stop",
                "contract_sha256",
                "tensor",
                "shape",
                "dtype",
            }
            or marker.get("schema_version") != 1
            or marker.get("artifact_type") != "surface_semantic_resume_batch"
            or marker.get("phase") != phase
            or marker.get("start") != start
            or marker.get("stop") != stop
            or marker.get("contract_sha256") != contract_sha256
            or marker.get("shape") != list(expected_shape)
            or marker.get("dtype") != str(expected_dtype)
        ):
            raise _resume_failure(resume_dir, f"{phase} batch marker differs")
        tensor_path = validate_file_record(
            marker["tensor"],
            label=f"{phase} resume tensor",
        )
        if tensor_path != shard.resolve():
            raise _resume_failure(resume_dir, f"{phase} batch path differs")
        value, _, _ = load_torch_payload(
            shard,
            expected_sha256=marker["tensor"]["sha256"],
            map_location="cpu",
            label=f"{phase} resume tensor",
        )
    except _ResumeStateError:
        raise
    except Exception as exc:
        raise _resume_failure(resume_dir, f"{phase} batch cannot be reopened") from exc
    if (
        not torch.is_tensor(value)
        or tuple(value.shape) != expected_shape
        or value.dtype != expected_dtype
        or not bool(torch.isfinite(value).all())
    ):
        raise _resume_failure(resume_dir, f"{phase} batch tensor differs")
    return value


def _commit_resume_tensor(
    resume_dir: Path,
    *,
    phase: str,
    start: int,
    stop: int,
    contract_sha256: str,
    value: torch.Tensor,
) -> None:
    shard, terminal = _resume_paths(
        resume_dir,
        phase=phase,
        start=start,
        stop=stop,
    )
    tensor = value.detach().cpu().contiguous()
    write_torch_noclobber(shard, tensor)
    marker = {
        "schema_version": 1,
        "artifact_type": "surface_semantic_resume_batch",
        "phase": phase,
        "start": int(start),
        "stop": int(stop),
        "contract_sha256": contract_sha256,
        "tensor": file_record(shard),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    write_frozen_json(terminal, marker)


def _pace_after_commit(device: torch.device, seconds: float) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time.sleep(float(seconds))


def _validate_resume_inventory(
    resume_dir: Path,
    *,
    row_count: int,
    radio_batch_size: int,
    semantic_batch_size: int,
    semantic_phase: str,
) -> None:
    allowed = {"contract.json"}
    for phase, batch_size in (
        ("radio", int(radio_batch_size)),
        (str(semantic_phase), int(semantic_batch_size)),
    ):
        for start in range(0, int(row_count), batch_size):
            stop = min(int(row_count), start + batch_size)
            shard, terminal = _resume_paths(
                resume_dir,
                phase=phase,
                start=start,
                stop=stop,
            )
            allowed.update((shard.name, terminal.name))
    unexpected = sorted(path.name for path in resume_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise _resume_failure(
            resume_dir,
            f"unexpected files {unexpected[:5]}",
        )


def _adjacency(graph: dict, neighbors: int) -> torch.Tensor:
    """Keep strongest surface-conditioned outgoing edges plus a self slot."""
    count = int(graph["xyz"].shape[0]); k = int(neighbors)
    edge = torch.as_tensor(graph["edge_index"]).long()
    affinity = torch.as_tensor(graph["raw_affinity"]).float()
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    for src, dst, weight in zip(edge[0].tolist(), edge[1].tolist(), affinity.tolist()):
        buckets[src].append((float(weight), int(dst)))
    result = torch.arange(count)[:, None].expand(-1, k).clone()
    for row, entries in enumerate(buckets):
        selected = [dst for _weight, dst in sorted(entries, reverse=True)[:k]]
        if selected:
            result[row, :len(selected)] = torch.tensor(selected)
    return result


def two_hop_physical_regions(
    centers: torch.Tensor, adjacency: torch.Tensor, xyz: torch.Tensor, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique two-hop surface candidates clipped by physical radius."""
    center = torch.as_tensor(centers, device=adjacency.device).long()
    first = adjacency[center]
    second = adjacency[first].flatten(1)
    rows = torch.cat([center[:, None], first, second], dim=1)
    rows, _ = rows.sort(dim=1)
    unique = torch.ones_like(rows, dtype=torch.bool)
    unique[:, 1:] = rows[:, 1:] != rows[:, :-1]
    distance = torch.linalg.vector_norm(xyz[rows] - xyz[center, None], dim=-1)
    mask = unique & (distance <= float(radius))
    # The center is guaranteed to survive sorting and the radius test.
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("physical surface region lost its center")
    return rows, mask


def completion_primary_valid(
    mpr: dict,
    output_valid: torch.Tensor,
) -> torch.Tensor | None:
    """Recover the explicit primary/fallback partition from a fused MPR."""

    metadata = dict(mpr.get("metadata", {}))
    if (
        metadata.get("construction")
        != "dominant_primary_with_query_free_support_completion"
    ):
        return None
    observed = torch.as_tensor(output_valid).bool().reshape(-1)
    reliability = torch.as_tensor(mpr.get("reliability")).float()
    if reliability.ndim != 2 or reliability.shape[0] != observed.numel():
        raise ValueError("completed MPR reliability does not align with rows")
    if reliability.shape[1] < 3:
        raise ValueError("completed MPR lacks its primary indicator channel")
    primary = observed & (reliability[:, 2] > 0.5)
    expected = metadata.get("primary_valid_count")
    if expected is None or int(primary.sum()) != int(expected):
        raise ValueError("completed MPR primary partition count mismatch")
    if not bool(primary.any()) or torch.equal(primary, observed):
        raise ValueError("completed MPR does not contain a distinct fallback partition")
    return primary


def preserve_primary_region_tokens(
    rows: torch.Tensor,
    mask: torch.Tensor,
    centers: torch.Tensor,
    primary_valid: torch.Tensor | None,
) -> torch.Tensor:
    """Prevent fallback rows from changing an existing primary descriptor."""

    active = torch.as_tensor(mask).bool()
    if primary_valid is None:
        return active
    neighbors = torch.as_tensor(rows).long()
    center_rows = torch.as_tensor(centers).long().reshape(-1)
    primary = torch.as_tensor(primary_valid).bool()
    if neighbors.shape != active.shape or center_rows.shape != (
        neighbors.shape[0],
    ):
        raise ValueError("surface region rows, mask, and centers must align")
    if neighbors.numel() and int(neighbors.max()) >= primary.numel():
        raise ValueError("surface region rows exceed the primary partition")
    return active & (
        primary[neighbors] | (neighbors == center_rows[:, None])
    )


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    canonical_radio_source = str(
        getattr(args, "canonical_radio_source", "field_decode")
    )
    if canonical_radio_source not in {"field_decode", "mpr_teacher"}:
        raise ValueError("unsupported canonical RADIO source")
    registration_arg = str(getattr(args, "experiment_registration", "")).strip()
    registration_record = file_record(registration_arg) if registration_arg else None
    if canonical_radio_source == "mpr_teacher" and registration_record is None:
        raise ValueError("mpr_teacher capacity diagnostics require preregistration")
    field_path, graph_path, readout_path = map(Path, (
        args.field_checkpoint, args.support_graph, args.readout_checkpoint,
    ))
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"semantic output already exists: {output}")
    radio_batch_size = int(args.radio_batch_size)
    semantic_batch_size = int(args.semantic_batch_size)
    pacing_seconds = float(args.thermal_pacing_seconds_per_batch)
    if (
        radio_batch_size <= 0
        or semantic_batch_size <= 0
        or not math.isfinite(pacing_seconds)
        or pacing_seconds <= 0.0
    ):
        raise ValueError("batch sizes and thermal pacing must be positive")

    field_expected = str(getattr(args, "field_checkpoint_sha256", "")).strip() or None
    graph_expected = str(getattr(args, "support_graph_sha256", "")).strip() or None
    readout_expected = str(getattr(args, "readout_checkpoint_sha256", "")).strip() or None
    mpr_expected = str(getattr(args, "mpr_cache_sha256", "")).strip() or None
    radio_expected = str(getattr(args, "radio_checkpoint_sha256", "")).strip() or None
    field, field_payload = load_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=field_expected,
    )
    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=graph_expected,
        map_location="cpu",
        label="surface support graph",
    )
    readout, readout_payload, _, _ = load_surface_region_summary_readout_v2(
        readout_path,
        expected_sha256=readout_expected,
        map_location="cpu",
    )
    if readout_payload["provenance"].get("uses_benchmark_scenes", True):
        raise ValueError("readout provenance is benchmark contaminated")
    mpr_path = Path(field_payload["mpr_cache"]).resolve()
    mpr, _, _ = load_torch_mapping(
        mpr_path,
        expected_sha256=mpr_expected,
        map_location="cpu",
        label="canonical field MPR cache",
    )
    field_record = file_record(field_path)
    graph_record = file_record(graph_path)
    readout_record = file_record(readout_path)
    mpr_record = file_record(mpr_path)
    radio_record = file_record(args.radio_checkpoint)
    for expected, record, label in (
        (field_expected, field_record, "field"),
        (graph_expected, graph_record, "support graph"),
        (readout_expected, readout_record, "readout"),
        (mpr_expected, mpr_record, "MPR"),
        (radio_expected, radio_record, "RADIO"),
    ):
        if expected is not None and record["sha256"] != expected:
            raise ValueError(f"{label} checkpoint SHA-256 differs")
    xyz_global = torch.as_tensor(mpr["xyz"]).float().cpu()
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    if not torch.equal(xyz, xyz_global[global_rows]):
        raise ValueError("support graph and canonical field geometry differ")
    output_valid = torch.zeros(len(xyz_global), dtype=torch.bool)
    output_valid[global_rows] = True
    primary_valid = completion_primary_valid(mpr, output_valid)
    primary_local = (
        primary_valid[global_rows] if primary_valid is not None else None
    )
    provenance = readout_payload["provenance"]
    training_scope = str(provenance.get("training_scope", ""))
    if (
        not training_scope.startswith("global_cross_scene")
        or provenance.get("uses_benchmark_scenes", True)
        or provenance.get("uses_benchmark_test_vocabulary", True)
        or provenance.get("scene_disjoint") is not True
    ):
        raise ValueError("readout provenance is not frozen global cross-scene training")
    contract = SurfaceRegionContractV2(**{
        **provenance["region_contract"],
        "radii_m": tuple(provenance["region_contract"]["radii_m"]),
    })
    if str(args.region_radii).strip():
        requested = tuple(
            float(value) for value in str(args.region_radii).replace(",", " ").split()
        )
        if requested != contract.radii_m:
            raise ValueError("CLI radii differ from the frozen readout contract")
    contract.assert_compatible({
        "region_contract_version": contract.version,
        "region_contract_sha256": provenance["region_contract_sha256"],
    })
    stream_text = bool(str(args.stream_text_queries).strip())
    preserve_streamed_text_scales = bool(
        getattr(args, "preserve_streamed_text_scales", False)
    )
    if preserve_streamed_text_scales and not stream_text:
        raise ValueError(
            "--preserve-streamed-text-scales requires --stream-text-queries"
        )
    resume_dir = (
        Path(args.resume_dir).resolve()
        if str(args.resume_dir).strip()
        else output.with_name(f"{output.name}.resume")
    )
    text_record = (
        file_record(args.text_embedding_cache)
        if stream_text and str(args.text_embedding_cache).strip()
        else None
    )
    resume_contract = {
        "schema_version": 1,
        "artifact_type": "surface_semantic_resume_contract",
        "output": str(output),
        "inputs": {
            "field": field_record,
            "mpr": mpr_record,
            "support_graph": graph_record,
            "readout": readout_record,
            "radio": radio_record,
            "text": text_record,
            **(
                {"experiment_registration": registration_record}
                if canonical_radio_source == "mpr_teacher"
                else {}
            ),
        },
        "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest,
        "radio_batch_size": radio_batch_size,
        "semantic_batch_size": semantic_batch_size,
        "stream_text_queries": str(args.stream_text_queries),
        **(
            {"preserve_streamed_text_scales": True}
            if preserve_streamed_text_scales
            else {}
        ),
        **(
            {"canonical_radio_source": canonical_radio_source}
            if canonical_radio_source == "mpr_teacher"
            else {}
        ),
        "device_type": device.type,
        "thermal_pacing_seconds_per_batch": pacing_seconds,
        "implementation": file_record(Path(__file__).resolve()),
    }
    resume_contract_sha256 = _load_or_create_resume_contract(
        resume_dir,
        resume_contract,
    )
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"], edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"], local_sigma=graph["local_sigma"],
        num_nodes=len(xyz), edge_channels=graph.get("edge_channels", {}),
    )
    prepared_graph = contract.prepare_graph(support, xyz)
    field, readout = field.to(device).eval(), readout.to(device).eval()
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        args.radio_checkpoint,
        **({"expected_sha256": radio_expected} if radio_expected else {}),
    ).to(device).eval()
    for module in (field, readout, head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    semantic_phase = (
        "text_scores_multiscale"
        if preserve_streamed_text_scales
        else ("text_scores" if stream_text else "semantic")
    )
    _validate_resume_inventory(
        resume_dir,
        row_count=len(global_rows),
        radio_batch_size=radio_batch_size,
        semantic_batch_size=semantic_batch_size,
        semantic_phase=semantic_phase,
    )
    radio = torch.empty(len(global_rows), 1280, dtype=torch.float16, device=device)
    for start in range(0, len(global_rows), radio_batch_size):
        stop = min(start + radio_batch_size, len(global_rows))
        cached = _load_resume_tensor(
            resume_dir,
            phase="radio",
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            expected_shape=(stop - start, 1280),
            expected_dtype=torch.float16,
        )
        if cached is not None:
            radio[start:stop] = cached.to(device)
            continue
        if canonical_radio_source == "field_decode":
            computed = field.radio_features(global_rows[start:stop].to(device)).half()
        else:
            computed = torch.as_tensor(mpr["features"])[
                global_rows[start:stop]
            ].to(device=device, dtype=torch.float16)
        radio[start:stop] = computed
        _commit_resume_tensor(
            resume_dir,
            phase="radio",
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            value=computed,
        )
        _pace_after_commit(device, pacing_seconds)
    semantic_confidence = None
    if primary_valid is not None:
        teacher_radio = torch.as_tensor(mpr["features"])[global_rows]
        observation_counts = torch.as_tensor(mpr["view_counts"])[global_rows]
        local_confidence = torch.zeros(
            len(global_rows), dtype=torch.float16
        )
        for start in range(0, len(global_rows), radio_batch_size):
            stop = min(start + radio_batch_size, len(global_rows))
            local_confidence[start:stop] = canonical_reconstruction_confidence(
                radio[start:stop].float(),
                teacher_radio[start:stop].to(
                    device=device, dtype=torch.float32
                ),
                torch.ones(stop - start, dtype=torch.bool, device=device),
                primary_local[start:stop].to(device),
                observation_counts[start:stop].to(device),
            ).half().cpu()
        semantic_confidence = torch.zeros(
            len(xyz_global), dtype=torch.float16
        )
        semantic_confidence[global_rows] = local_confidence
    if contract.reliability_semantics == "uniform_valid":
        reliability = torch.ones(len(global_rows), dtype=torch.float32)
    else:
        reliability_source = torch.as_tensor(mpr.get("reliability")).float()[
            global_rows
        ]
        if reliability_source.ndim != 2 or reliability_source.shape[1] < 2:
            raise ValueError(
                "canonical MPR reliability needs coverage/agreement channels"
            )
        reliability = (
            reliability_source[:, :2].clamp_min(1e-6).log().mean(-1).exp()
        )
        reliability[(reliability_source[:, :2] <= 0).any(-1)] = 0.0
    reliability = reliability.to(device)
    local_scale = torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4).to(device)
    xyz_device = xyz.to(device)
    radii = contract.radii_m
    text_queries: list[str] = []
    text_embeddings = None
    if stream_text:
        if not args.text_embedding_cache:
            raise ValueError("streaming text queries require --text-embedding-cache")
        text_payload, _, _ = load_torch_mapping(
            args.text_embedding_cache,
            expected_sha256=(text_record or {}).get("sha256"),
            map_location="cpu",
            label="streamed text embedding cache",
        )
        available = [str(value) for value in text_payload.get("queries", [])]
        text_queries = [
            value.strip() for value in str(args.stream_text_queries).split(",")
            if value.strip()
        ]
        if preserve_streamed_text_scales and len(set(text_queries)) != len(
            text_queries
        ):
            raise ValueError("multiscale streamed text query IDs must be unique")
        lookup = {name: index for index, name in enumerate(available)}
        missing = [name for name in text_queries if name not in lookup]
        if missing:
            raise ValueError(f"streaming text queries are absent: {missing}")
        text_embeddings = F.normalize(
            torch.as_tensor(text_payload["embeddings"])[
                torch.tensor([lookup[name] for name in text_queries])
            ].float(), dim=-1, eps=1e-8,
        )
        streamed_scores = torch.zeros(
            (
                (len(xyz_global), len(radii), len(text_queries))
                if preserve_streamed_text_scales
                else (len(xyz_global), len(text_queries))
            ),
            dtype=torch.float16,
        )
        descriptors_by_scale = None
    else:
        descriptors_by_scale = torch.zeros(
            len(global_rows), len(radii), 1536, dtype=torch.float16
        )
    for start in range(0, len(global_rows), semantic_batch_size):
        stop = min(start + semantic_batch_size, len(global_rows))
        cached_shape = (
            (
                (stop - start, len(radii), len(text_queries))
                if preserve_streamed_text_scales
                else (stop - start, len(text_queries))
            )
            if stream_text
            else (stop - start, len(radii), 1536)
        )
        cached = _load_resume_tensor(
            resume_dir,
            phase=semantic_phase,
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            expected_shape=cached_shape,
            expected_dtype=torch.float16,
        )
        if cached is not None:
            if stream_text:
                streamed_scores[global_rows[start:stop]] = cached
            else:
                assert descriptors_by_scale is not None
                descriptors_by_scale[start:stop] = cached
            continue
        centers_cpu = torch.arange(start, stop)
        batch_streamed_scores = None
        for scale_index, radius in enumerate(radii):
            regions = contract.expand_batch(
                support, xyz, centers_cpu.tolist(), radius,
                prepared_graph=prepared_graph,
            )
            batch = len(regions); width = contract.maximum_tokens
            rows = torch.zeros(batch, width, dtype=torch.long)
            mask = torch.zeros(batch, width, dtype=torch.bool)
            core = torch.zeros(batch, width, dtype=torch.bool)
            anchor_local = torch.zeros(batch, dtype=torch.long)
            for offset, (region_rows, region_core, _distance) in enumerate(regions):
                count = len(region_rows)
                rows[offset, :count] = region_rows
                mask[offset, :count] = True
                core[offset, :count] = region_core
                anchor_local[offset] = int(torch.where(region_rows == centers_cpu[offset])[0][0])
            mask = preserve_primary_region_tokens(
                rows,
                mask,
                centers_cpu,
                primary_local,
            )
            core &= mask
            rows, mask, core, anchor_local = (
                rows.to(device), mask.to(device), core.to(device), anchor_local.to(device)
            )
            token_xyz = xyz_device[rows]
            token_scale = local_scale[rows, None].expand(-1, -1, 3)
            token_reliability = reliability[rows, None]
            geometry = surface_region_geometry_v2(
                token_xyz, token_scale, token_reliability, float(radius),
                anchor_index=anchor_local, core_mask=core, token_mask=mask,
            )
            summary = readout(
                radio[rows], geometry, token_mask=mask,
                reliability=token_reliability, anchor_index=anchor_local,
            )
            descriptor = F.normalize(
                head(summary[:, None])[:, 0].float(), dim=-1
            ).half()
            if stream_text:
                # Match the warm-cache compiler exactly: descriptors are
                # quantized to fp16 before normalized cosine and scores are
                # finally stored as fp16 primitive unaries.
                assert text_embeddings is not None
                # Deliberately perform this tiny Q-way dot product on CPU.
                # The warm-cache compiler also reloads fp16 descriptors on
                # CPU, so this makes cold/warm unaries bitwise reproducible
                # instead of merely close across CUDA/CPU reduction kernels.
                scale_scores = F.normalize(
                    descriptor.cpu().float(), dim=-1, eps=1e-8
                ) @ text_embeddings.T
                if preserve_streamed_text_scales:
                    if batch_streamed_scores is None:
                        batch_streamed_scores = torch.empty(
                            batch,
                            len(radii),
                            len(text_queries),
                            dtype=torch.float32,
                        )
                    batch_streamed_scores[:, scale_index] = scale_scores
                else:
                    batch_streamed_scores = (
                        scale_scores
                        if batch_streamed_scores is None
                        else torch.maximum(batch_streamed_scores, scale_scores)
                    )
            else:
                assert descriptors_by_scale is not None
                descriptors_by_scale[start:stop, scale_index] = descriptor.cpu()
        if stream_text:
            assert batch_streamed_scores is not None
            committed = batch_streamed_scores.half()
            streamed_scores[global_rows[start:stop]] = committed
        else:
            assert descriptors_by_scale is not None
            committed = descriptors_by_scale[start:stop].clone()
        _commit_resume_tensor(
            resume_dir,
            phase=semantic_phase,
            start=start,
            stop=stop,
            contract_sha256=resume_contract_sha256,
            value=committed,
        )
        _pace_after_commit(device, pacing_seconds)
    readout_sha256 = readout_record["sha256"]
    radio_sha256 = radio_record["sha256"]
    metadata = {
        "schema_version": 5, "feature_space": "official_siglip2_summary_descriptor_multiscale",
        "source": "canonical_radio_surface_region_readout",
        "construction": "canonical_radio_surface_region_readout_then_official_summary_head",
        "canonical_radio_source": (
            "field_decode_only"
            if canonical_radio_source == "field_decode"
            else "frozen_mpr_full_1280_teacher"
        ),
        "mpr_radio_features_opened": canonical_radio_source == "mpr_teacher",
        **(
            {
                "capacity_diagnostic_only": True,
                "experiment_registration": registration_record,
            }
            if canonical_radio_source == "mpr_teacher"
            else {}
        ),
        "readout_checkpoint": str(readout_path.resolve()),
        "readout_checkpoint_sha256": readout_sha256,
        "bridge_checkpoint_sha256": readout_sha256,
        "bridge_training_scope": "global_cross_scene",
        "bridge_training_scope_detail": training_scope,
        "field_checkpoint": field_record["path"],
        "field_checkpoint_sha256": field_record["sha256"],
        "mpr_cache": mpr_record["path"],
        "mpr_cache_sha256": mpr_record["sha256"],
        "field_geometry_xyz_sha256": field_payload.get(
            "geometry_fingerprint", {}
        ).get("xyz_sha256"),
        "support_graph": str(graph_path.resolve()),
        "support_graph_sha256": graph_record["sha256"],
        "official_radio_checkpoint_sha256": radio_sha256,
        "radio_checkpoint_sha256": radio_sha256,
        "region_radii_m": list(radii), "region_topology": contract.expansion,
        "readout_batch_size": int(args.semantic_batch_size),
        "region_contract": contract.to_dict(),
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
        "query_set_invariant": True, "benchmark_images_opened": False,
        "official_summary_head": True, "custom_text_projection": False,
        "benchmark_masks_opened": False, "text_queries_opened": False,
        "cache_role": "disposable_derivative_not_scene_memory",
        "row_storage": "sparse_valid_rows_with_global_row_index",
        "scale_storage": "all_scales_preserved; mean_descriptor_legacy_only",
        "resume_contract_sha256": resume_contract_sha256,
        "thermal_pacing_seconds_per_batch": pacing_seconds,
        "completion_context_policy": (
            "primary_plus_center"
            if primary_valid is not None
            else "all_valid"
        ),
        "primary_valid_count": (
            int(primary_valid.sum()) if primary_valid is not None else None
        ),
        "semantic_confidence": (
            {
                "source": "canonical_radio_reconstruction_fidelity",
                "nonzero_count": int((semantic_confidence > 0).sum()),
                "mean_valid": float(
                    semantic_confidence[output_valid].float().mean()
                ),
            }
            if semantic_confidence is not None
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if stream_text:
        if preserve_streamed_text_scales:
            # The frozen LERF Direct3D protocol consumes all three raw scales
            # and performs its own fixed KNN10/peak-scale readout.  Completion
            # or scale reduction here would change that protocol.
            streamed_scores[~output_valid] = 0
            completion = {
                "applied": False,
                "reason": "frozen_direct3d_requires_raw_unreduced_scale_scores",
            }
        else:
            streamed_scores, completion = apply_completion_evidence(
                streamed_scores,
                output_valid,
                semantic_confidence=semantic_confidence,
                primary_valid=primary_valid,
                routing=str(
                    getattr(args, "completion_routing", "primary_first")
                ),
                primary_support_threshold=float(
                    getattr(args, "primary_support_threshold", 0.5)
                ),
                primary_support_mode=str(
                    getattr(args, "primary_support_mode", "relative_peak")
                ),
                primary_support_margin=float(
                    getattr(args, "primary_support_margin", 0.02)
                ),
            )
        score_metadata = {
            "schema_version": 3 if preserve_streamed_text_scales else 2,
            "feature_space": (
                "primitive_text_query_scores_multiscale_unreduced"
                if preserve_streamed_text_scales
                else "primitive_text_query_scores"
            ),
            "construction": (
                "cold_streaming_surface_region_readout_then_independent_cosine"
                if preserve_streamed_text_scales
                else "cold_streaming_surface_region_readout_then_cosine_max"
            ),
            "scoring": (
                "raw_independent_normalized_cosine"
                if preserve_streamed_text_scales
                else "cosine"
            ),
            "scale_aggregation": (
                "none_frozen_downstream_only"
                if preserve_streamed_text_scales
                else "max"
            ),
            "scale_count": len(radii),
            **(
                {"scale_radii_m": list(radii)}
                if preserve_streamed_text_scales
                else {}
            ),
            "score_chunk_size": int(args.semantic_batch_size),
            "query_names": text_queries,
            "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
            **(
                {
                    "text_embedding_cache_sha256": (text_record or {}).get(
                        "sha256"
                    ),
                    "streaming_implementation": file_record(
                        Path(__file__).resolve()
                    ),
                }
                if preserve_streamed_text_scales
                else {}
            ),
            "semantic_cache_materialized": False,
            "completion": completion,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": True,
            "semantic_provenance": metadata,
        }
        score_payload = {
            "xyz": xyz_global,
            "features": streamed_scores,
            "valid": output_valid,
            "metadata": score_metadata,
        }
        if primary_valid is not None:
            score_payload["primary_valid"] = primary_valid
        if semantic_confidence is not None:
            score_payload["semantic_confidence"] = semantic_confidence
        _atomic_torch_save(score_payload, output)
        report = {
            "output": str(output.resolve()),
            "valid_primitives": int(output_valid.sum()),
            "total_primitives": len(output_valid),
            "num_queries": len(text_queries),
            "semantic_cache_materialized": False,
            "metadata": score_metadata,
        }
        write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
        return report
    assert descriptors_by_scale is not None
    descriptors = F.normalize(descriptors_by_scale.float().mean(1), dim=-1).half()
    # Semantic descriptors dominate cache size (1536 fp16 values per row).  Do
    # not materialize zero descriptors for invalid/background primitives.  The
    # global geometry and explicit row index retain an exact, lossless mapping;
    # consumers expand only when their downstream score representation needs it.
    semantic_payload = {
        "xyz": xyz_global,
        "features": descriptors,
        "summary_features": descriptors,
        "global_rows": global_rows,
        "features_by_scale": descriptors_by_scale,
        "valid": output_valid,
        "metadata": metadata,
    }
    if primary_valid is not None:
        semantic_payload["primary_valid"] = primary_valid
    if semantic_confidence is not None:
        semantic_payload["semantic_confidence"] = semantic_confidence
    _atomic_torch_save(semantic_payload, output)
    # Pose-free image querying only consumes the already aggregated descriptor,
    # never the retained per-scale tensor.  Save an exact, provenance-identical
    # derivative alongside the full cache so each query does not repeatedly
    # deserialize several otherwise unused gigabytes.  This is an execution
    # representation change only: ``features`` is byte-for-byte the tensor
    # stored in the full semantic cache.
    query_output = (
        Path(args.query_output)
        if str(args.query_output).strip()
        else output.with_name(f"{output.stem}_query{output.suffix}")
    )
    query_payload = {
        "xyz": xyz_global,
        "features": descriptors,
        "global_rows": global_rows,
        "valid": output_valid,
        "metadata": metadata,
    }
    if primary_valid is not None:
        query_payload["primary_valid"] = primary_valid
    if semantic_confidence is not None:
        query_payload["semantic_confidence"] = semantic_confidence
    _atomic_torch_save(query_payload, query_output)
    report = {"output": str(output.resolve()), "valid_primitives": int(output_valid.sum()),
              "total_primitives": len(output_valid), "metadata": metadata}
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    write_frozen_json(
        output.with_suffix(output.suffix + ".provenance.json"),
        {"cache": str(output.resolve()), "inputs": metadata},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument(
        "--canonical-radio-source",
        choices=("field_decode", "mpr_teacher"),
        default="field_decode",
        help=(
            "Diagnostic source for the 1280-D RADIO rows. The default preserves "
            "the compact canonical-field path; mpr_teacher is a full-capacity "
            "label-free teacher upper-bound and must not be reported as the field."
        ),
    )
    parser.add_argument(
        "--experiment-registration",
        default="",
        help=(
            "Immutable preregistration receipt. Required for the label-free "
            "mpr_teacher full-capacity diagnostic."
        ),
    )
    parser.add_argument("--field-checkpoint-sha256", default="")
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", default="")
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--readout-checkpoint-sha256", default="")
    parser.add_argument("--mpr-cache-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--query-output", default="",
        help=(
            "Optional compact descriptor-only sidecar for pose-free querying; "
            "defaults next to --output."
        ),
    )
    parser.add_argument("--region-radii", default="")
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument("--radio-batch-size", type=int, default=4096)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    parser.add_argument("--resume-dir", default="")
    parser.add_argument(
        "--thermal-pacing-seconds-per-batch",
        type=float,
        default=1.0,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-embedding-cache", default="")
    parser.add_argument(
        "--completion-routing",
        choices=("primary_first", "direct", "primary_only"),
        default="primary_first",
    )
    parser.add_argument(
        "--primary-support-mode",
        choices=("absolute", "relative_peak"),
        default="relative_peak",
    )
    parser.add_argument("--primary-support-threshold", type=float, default=0.5)
    parser.add_argument("--primary-support-margin", type=float, default=0.02)
    parser.add_argument(
        "--stream-text-queries", default="",
        help=(
            "Optional ordered comma-separated queries. When set, execute the "
            "readout and cosine scoring as a cold stream and save only scalar "
            "primitive unaries, never a 1536D semantic cache."
        ),
    )
    parser.add_argument(
        "--preserve-streamed-text-scales",
        action="store_true",
        help=(
            "Keep raw [primitive,3,query] cosine scores for the frozen LERF "
            "Direct3D protocol. This disables both scale reduction and "
            "completion in the streamed derivative."
        ),
    )
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    parser.add_argument("--radio-checkpoint-sha256", default="")
    args = parser.parse_args(); print(json.dumps(build(args), indent=2))


if __name__ == "__main__": main()
