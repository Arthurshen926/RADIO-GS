#!/usr/bin/env python3
"""Derive a disposable text-space cache from the canonical field and v3 readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadoutV2, surface_region_geometry_v2,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_canonical_primitive_semantic_cache import (
    canonical_reconstruction_confidence,
)
from radio_gs.scripts.build_primitive_text_score_cache import (
    apply_completion_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(payload: object, output: Path) -> None:
    """Publish a cache only after its entire tensor payload is durable.

    Full semantic caches are several GB, while the PFIR field workers and the
    compact-query cache materializer may run concurrently.  ``torch.save``
    creates its destination immediately, so file-existence checks alone do not
    establish that a reader sees a complete archive.  A same-filesystem rename
    does, and leaves a prior valid cache untouched on interruption.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    field_path, graph_path, readout_path = map(Path, (
        args.field_checkpoint, args.support_graph, args.readout_checkpoint,
    ))
    field, field_payload = load_canonical_field_checkpoint(field_path, map_location="cpu")
    graph = torch.load(graph_path, map_location="cpu")
    readout, readout_payload = SurfaceRegionSummaryReadoutV2.from_checkpoint(readout_path)
    if readout_payload["provenance"].get("uses_benchmark_scenes", True):
        raise ValueError("readout provenance is benchmark contaminated")
    mpr = torch.load(Path(field_payload["mpr_cache"]), map_location="cpu")
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
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"], edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"], local_sigma=graph["local_sigma"],
        num_nodes=len(xyz), edge_channels=graph.get("edge_channels", {}),
    )
    prepared_graph = contract.prepare_graph(support, xyz)
    field, readout = field.to(device).eval(), readout.to(device).eval()
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    for module in (field, readout, head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    radio = torch.empty(len(global_rows), 1280, dtype=torch.float16, device=device)
    for start in range(0, len(global_rows), int(args.radio_batch_size)):
        stop = min(start + int(args.radio_batch_size), len(global_rows))
        radio[start:stop] = field.radio_features(global_rows[start:stop].to(device)).half()
    semantic_confidence = None
    if primary_valid is not None:
        teacher_radio = torch.as_tensor(mpr["features"])[global_rows]
        observation_counts = torch.as_tensor(mpr["view_counts"])[global_rows]
        local_confidence = torch.zeros(
            len(global_rows), dtype=torch.float16
        )
        for start in range(0, len(global_rows), int(args.radio_batch_size)):
            stop = min(start + int(args.radio_batch_size), len(global_rows))
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
    stream_text = bool(str(args.stream_text_queries).strip())
    text_queries: list[str] = []
    text_embeddings = None
    if stream_text:
        if not args.text_embedding_cache:
            raise ValueError("streaming text queries require --text-embedding-cache")
        text_payload = torch.load(args.text_embedding_cache, map_location="cpu")
        available = [str(value) for value in text_payload.get("queries", [])]
        text_queries = [
            value.strip() for value in str(args.stream_text_queries).split(",")
            if value.strip()
        ]
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
            len(xyz_global), len(text_queries), dtype=torch.float16
        )
        descriptors_by_scale = None
    else:
        descriptors_by_scale = torch.zeros(
            len(global_rows), len(radii), 1536, dtype=torch.float16
        )
    for start in range(0, len(global_rows), int(args.semantic_batch_size)):
        stop = min(start + int(args.semantic_batch_size), len(global_rows))
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
                batch_streamed_scores = (
                    scale_scores if batch_streamed_scores is None
                    else torch.maximum(batch_streamed_scores, scale_scores)
                )
            else:
                assert descriptors_by_scale is not None
                descriptors_by_scale[start:stop, scale_index] = descriptor.cpu()
        if stream_text:
            assert batch_streamed_scores is not None
            streamed_scores[global_rows[start:stop]] = batch_streamed_scores.half()
    readout_sha256 = _sha256(readout_path)
    radio_sha256 = _sha256(Path(args.radio_checkpoint))
    metadata = {
        "schema_version": 5, "feature_space": "official_siglip2_summary_descriptor_multiscale",
        "source": "canonical_radio_surface_region_readout",
        "construction": "canonical_radio_surface_region_readout_then_official_summary_head",
        "canonical_radio_source": "field_decode_only", "mpr_radio_features_opened": False,
        "readout_checkpoint": str(readout_path.resolve()),
        "readout_checkpoint_sha256": readout_sha256,
        "bridge_checkpoint_sha256": readout_sha256,
        "bridge_training_scope": "global_cross_scene",
        "bridge_training_scope_detail": training_scope,
        "field_checkpoint": str(field_path.resolve()),
        "field_checkpoint_sha256": _sha256(field_path),
        "support_graph": str(graph_path.resolve()),
        "support_graph_sha256": _sha256(graph_path),
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
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if stream_text:
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
            "schema_version": 2,
            "feature_space": "primitive_text_query_scores",
            "construction": "cold_streaming_surface_region_readout_then_cosine_max",
            "scoring": "cosine",
            "scale_aggregation": "max",
            "scale_count": len(radii),
            "score_chunk_size": int(args.semantic_batch_size),
            "query_names": text_queries,
            "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
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
        output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
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
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    output.with_suffix(output.suffix + ".provenance.json").write_text(
        json.dumps({"cache": str(output.resolve()), "inputs": metadata}, indent=2)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
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
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args(); print(json.dumps(build(args), indent=2))


if __name__ == "__main__": main()
