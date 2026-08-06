#!/usr/bin/env python3
"""Materialize one formal AcceptedV2 canonical-region/e0 authority.

This is a source-only producer, not a semantic-cache converter.  It reopens a
caller-SHA-bound factorized field/state, query-free support graph, immutable
AcceptedV2 checkpoint, and official C-RADIO checkpoint.  Canonical V2 regions
are selected directly from the frozen region contract and their readout tokens
are projected through the singleton official SigLIP2 summary head.

No text query, benchmark label, mask, metric, or semantic-cache final payload
is accepted.  Publication is first-writer-wins and never overwrites a path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
)
from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    SAMPLING_CONTRACT_SHA256,
    region_fingerprint,
    select_scale_stratified_indices,
)
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_metadata,
)
from radio_gs.interfaces.surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    SurfaceRegionSummaryReadoutV4,
    surface_region_geometry_v2,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.build_surface_region_semantic_cache import (
    load_surface_factorized_state_bundle,
)
from radio_gs.scripts.train_surface_region_full_scalar_residual import (
    canonical_physical_space_id,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


UPSTREAM_CHAIN = (
    "clean source RGB/poses -> geometry checkpoint -> exact-marginal "
    "factorized RADIO cache -> schema-v2 factorized field/state -> official "
    "DINO/SAM capability bank -> clean canonical support graph -> this producer"
)
RESPONSIBILITY_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
)
RESPONSIBILITY_VIEW_SCHEMA = (
    "radio_gs.sparse_exact_marginal_responsibility_view.v1"
)


@dataclass(frozen=True)
class _Runtime:
    scene_id: str
    field: torch.nn.Module
    state: Any
    graph_payload: Mapping[str, Any]
    support: PrimitiveSupportGraph
    contract: SurfaceRegionContractV2
    readout: torch.nn.Module
    input_authority: Mapping[str, Any]
    input_records: Mapping[str, Mapping[str, Any]]
    anchor_visible: torch.Tensor


def _require_expected_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"{label} is missing: {source}. Required producer chain: "
            f"{UPSTREAM_CHAIN}"
        )
    expected = shard._require_sha256(expected_sha256, label=label)
    observed = sha256_file(source)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return {
        "path": str(source),
        "sha256": observed,
        "size_bytes": source.stat().st_size,
    }


def _load_exact_anchor_visibility(
    authority_path: str | Path,
    *,
    expected_sha256: str,
    num_gaussians: int,
    xyz_sha256: str,
) -> tuple[torch.Tensor, dict[str, str]]:
    """Validate every sparse responsibility shard and return global visibility."""

    value, observed, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="exact-marginal responsibility authority",
    )
    if not isinstance(value, Mapping):
        raise ValueError("exact-marginal responsibility authority must be a mapping")
    payload = dict(value)
    metadata = payload.get("metadata")
    formula = payload.get("formula_contract")
    views = payload.get("views")
    formula_sha = str(payload.get("formula_sha256", ""))
    if (
        set(payload) != {
            "schema", "schema_version", "formula_contract", "formula_sha256",
            "frame_indices", "metadata", "num_gaussians", "num_pixels",
            "total_hits", "views",
        }
        or payload.get("schema") != RESPONSIBILITY_SCHEMA
        or payload.get("schema_version") != 1
        or int(payload.get("num_gaussians", -1)) != int(num_gaussians)
        or not isinstance(metadata, Mapping)
        or metadata.get("xyz_sha256") != xyz_sha256
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or not isinstance(formula, Mapping)
        or formula.get("query_independent") is not True
        or formula.get("feature_independent") is not True
        or formula_sha != canonical_json_sha256(formula)
        or metadata.get("formula_sha256") != formula_sha
        or not isinstance(views, list)
        or not views
    ):
        raise ValueError("exact-marginal responsibility selection authority differs")
    visible = torch.zeros(int(num_gaussians), dtype=torch.bool)
    last: tuple[int, int] | None = None
    hit_total = 0
    frames: list[int] = []
    frozen_views: list[dict[str, Any]] = []
    for raw in views:
        if not isinstance(raw, Mapping) or set(raw) != {
            "frame_index", "num_hits", "relative_path", "sha256", "view_index"
        }:
            raise ValueError("responsibility selection view record differs")
        frame = int(raw["frame_index"])
        view = int(raw["view_index"])
        key = (frame, view)
        relative = Path(str(raw["relative_path"]))
        count = int(raw["num_hits"])
        digest = shard._require_sha256(
            raw["sha256"], label="responsibility selection view"
        )
        if (
            count <= 0
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or (last is not None and key <= last)
        ):
            raise ValueError("responsibility selection view identity/order differs")
        last = key
        path = (source.parent / relative).resolve()
        if source.parent not in path.parents:
            raise ValueError("responsibility selection view escapes authority root")
        view_value, _, _ = load_torch_mapping(
            path,
            expected_sha256=digest,
            map_location="cpu",
            label="responsibility selection view",
        )
        if not isinstance(view_value, Mapping) or set(view_value) != {
            "schema", "schema_version", "formula_sha256", "view_index",
            "frame_index", "num_gaussians", "num_pixels", "gaussian_ids",
            "pixel_ids", "base_weights",
        }:
            raise ValueError("responsibility selection view payload differs")
        gaussian = torch.as_tensor(view_value["gaussian_ids"]).long().cpu()
        pixels = torch.as_tensor(view_value["pixel_ids"]).long().cpu()
        weights = torch.as_tensor(view_value["base_weights"]).float().cpu()
        num_pixels = int(view_value.get("num_pixels", -1))
        if (
            view_value.get("schema") != RESPONSIBILITY_VIEW_SCHEMA
            or view_value.get("schema_version") != 1
            or view_value.get("formula_sha256") != formula_sha
            or int(view_value.get("view_index", -1)) != view
            or int(view_value.get("frame_index", -1)) != frame
            or int(view_value.get("num_gaussians", -1)) != int(num_gaussians)
            or gaussian.shape != (count,)
            or pixels.shape != (count,)
            or weights.shape != (count,)
            or num_pixels <= 0
            or bool((gaussian < 0).any())
            or bool((gaussian >= num_gaussians).any())
            or bool((pixels < 0).any())
            or bool((pixels >= num_pixels).any())
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise ValueError("responsibility selection view tensor differs")
        visible[gaussian[weights > 0]] = True
        hit_total += count
        frames.append(frame)
        frozen_views.append(
            {
                "frame_index": frame,
                "num_hits": count,
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "view_index": view,
            }
        )
    if (
        payload.get("frame_indices") != frames
        or int(payload.get("total_hits", -1)) != hit_total
    ):
        raise ValueError("responsibility selection aggregate differs")
    return visible, {
        "kind": "exact_marginal_anchor_visibility_sparse_selection_v1",
        "exact_marginal_responsibility_authority_file_sha256": observed,
        "exact_marginal_formula_sha256": formula_sha,
        "responsibility_view_records_sha256": canonical_json_sha256(frozen_views),
        "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
    }


def _validate_source_only_graph(
    graph: Mapping[str, Any],
    *,
    state: Any,
    field_checkpoint_sha256: str,
    factorized_radio_cache_sha256: str,
) -> tuple[PrimitiveSupportGraph, PrimitiveRowAuthority]:
    required = {
        "schema_version",
        "global_rows",
        "num_global_rows",
        "xyz",
        "edge_index",
        "edge_weight",
        "raw_affinity",
        "edge_channels",
        "local_sigma",
        "metadata",
    }
    if set(graph) != required or graph.get("schema_version") != 1:
        raise ValueError("clean support graph payload fields differ")
    metadata = graph.get("metadata")
    if not isinstance(metadata, Mapping) or any(
        metadata.get(key) is not False
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError("clean support graph is query/benchmark contaminated")
    channels = graph.get("edge_channels")
    if (
        metadata.get("source")
        != "canonical_official_dino_sam3_multichannel_support_graph"
        or not isinstance(channels, Mapping)
        or not {"appearance", "boundary"}.issubset(channels)
    ):
        raise ValueError("clean support graph lacks official DINO/SAM relations")
    capability = metadata.get("capability_metadata")
    training_authority = (
        capability.get("capability_training_authority")
        if isinstance(capability, Mapping)
        else None
    )
    exact_sources = (
        training_authority.get("exact_source_capabilities")
        if isinstance(training_authority, Mapping)
        else None
    )
    if (
        not isinstance(capability, Mapping)
        or capability.get("field_checkpoint_schema_version") != 2
        or capability.get("field_checkpoint_sha256")
        != field_checkpoint_sha256
        or capability.get("factorized_radio_cache_sha256")
        != factorized_radio_cache_sha256
        or capability.get("query_independent") is not True
        or capability.get("custom_adaptor_head") is not False
        or any(
            capability.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
        or metadata.get("capability_authority_bootstrap") is not None
        or not isinstance(training_authority, Mapping)
        or training_authority.get("source")
        != "formal_exact_marginal_capability_training_authority_v1"
        or not isinstance(exact_sources, Mapping)
        or set(exact_sources) != {"appearance", "boundary"}
        or any(
            not isinstance(exact_sources.get(role), Mapping)
            or len(str(exact_sources[role].get("sha256", ""))) != 64
            for role in ("appearance", "boundary")
        )
        or any(
            training_authority.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("clean support graph field/cache lineage differs")
    row_authority = PrimitiveRowAuthority.from_mapping(
        metadata.get("primitive_row_authority")
    )
    row_authority.validate(state.xyz, state.valid)
    expected_rows = torch.as_tensor(state.global_rows).long().cpu()
    graph_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    graph_xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    state_xyz = torch.as_tensor(state.xyz).float().cpu()
    if (
        int(graph["num_global_rows"]) != int(state.valid.numel())
        or not torch.equal(graph_rows, expected_rows)
        or not torch.equal(graph_xyz, state_xyz[expected_rows])
    ):
        raise ValueError("clean support graph geometry/global rows differ")
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=int(expected_rows.numel()),
        edge_channels=graph["edge_channels"],
    )
    return support, row_authority


def accepted_region_input_authority(
    *,
    state_file_sha256: str,
    field_checkpoint_sha256: str,
    factorized_radio_cache_sha256: str,
    support_graph_sha256: str,
    row_authority: PrimitiveRowAuthority,
    geometry_fingerprint: Mapping[str, Any],
    selection_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "geometry_authority": {
            "kind": "factorized_primitive_state_v2",
            "factorized_primitive_state_file_sha256": state_file_sha256,
            "factorized_primitive_state_contract_sha256": (
                shard.FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
            ),
            "factorized_field_checkpoint_file_sha256": (
                field_checkpoint_sha256
            ),
            "factorized_radio_cache_file_sha256": (
                factorized_radio_cache_sha256
            ),
            "primitive_row_authority_sha256": row_authority.digest,
            "geometry_fingerprint": dict(geometry_fingerprint),
        },
        "support_graph_authority": {
            "kind": "canonical_query_free_support_graph_v1",
            "support_graph_file_sha256": support_graph_sha256,
            "primitive_row_authority_sha256": row_authority.digest,
        },
        "selection_authority": dict(selection_authority),
        "accepted_v2_checkpoint_authority": shard.trainer._accepted_v2_authority(),
        "official_summary_head_authority": (
            shard.accepted_region_official_head_authority()
        ),
    }


def validate_accepted_v2_graph_contract(
    graph: Mapping[str, Any],
    contract: SurfaceRegionContractV2,
) -> None:
    """Require every mathematical graph parameter frozen by AcceptedV2."""

    metadata = graph.get("metadata")
    observed = dict(
        metadata.get("graph_config", {}) if isinstance(metadata, Mapping) else {}
    )
    expected = asdict(contract.graph_config())
    # Affinity chunking changes only construction memory/throughput, not the
    # mathematical graph.  Every relation/topology parameter remains exact.
    observed.pop("affinity_chunk_size", None)
    expected.pop("affinity_chunk_size", None)
    if observed != expected:
        raise ValueError("clean support graph differs from AcceptedV2 graph contract")


def build_authority_payload(
    *,
    scene_id: str,
    geometry_fingerprint: Mapping[str, Any],
    accepted_base_valid: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    region_fingerprints: list[str],
    selection_audit: Mapping[str, Any],
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
    scale_indices: torch.Tensor,
    accepted_v2_e0: torch.Tensor,
    input_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Sort aligned region tensors and build exactly the shard-validator payload."""

    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    mask = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    anchors = torch.as_tensor(anchor_index).detach().long().cpu().contiguous()
    scales = torch.as_tensor(scale_indices).detach().long().cpu().contiguous()
    e0 = torch.as_tensor(accepted_v2_e0).detach().float().cpu().contiguous()
    if (
        rows.ndim != 2
        or mask.shape != rows.shape
        or anchors.shape != (rows.shape[0],)
        or scales.shape != (rows.shape[0],)
        or e0.shape != (rows.shape[0], shard.trainer.DESCRIPTOR_DIM)
    ):
        raise ValueError("AcceptedV2 region producer tensors are not aligned")
    canonical = torch.as_tensor(canonical_region_indices).long().cpu().contiguous()
    if (
        canonical.shape != (rows.shape[0],)
        or (canonical.numel() > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or len(region_fingerprints) != rows.shape[0]
    ):
        raise ValueError("AcceptedV2 sparse canonical selection differs")
    contract = shard.accepted_region_authority_contract()
    payload: dict[str, Any] = {
        "schema": shard.ACCEPTED_REGION_SCHEMA,
        "schema_version": shard.ACCEPTED_REGION_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": shard.canonical_json_sha256(contract),
        "scene_id": str(scene_id),
        "physical_space_id": canonical_physical_space_id(str(scene_id)),
        "accepted_v2_authority": shard.trainer._accepted_v2_authority(),
        "geometry_fingerprint": dict(geometry_fingerprint),
        "accepted_base_valid": (
            torch.as_tensor(accepted_base_valid)
            .detach()
            .bool()
            .cpu()
            .contiguous()
        ),
        "canonical_region_indices": canonical,
        "region_fingerprints": list(region_fingerprints),
        "selection_audit": dict(selection_audit),
        "region_rows": rows,
        "token_mask": mask,
        "anchor_index": anchors,
        "scale_indices": scales,
        "accepted_v2_e0": e0,
        "input_authority": dict(input_authority),
        "source_access": shard._authority_access(source_rgb_used=False),
    }
    payload["channel_sha256"] = shard.accepted_region_channel_sha256(payload)
    return shard.validate_accepted_region_authority(payload)


def preflight(args: argparse.Namespace) -> _Runtime:
    records = {
        "field_checkpoint": _require_expected_file(
            args.factorized_field_checkpoint,
            args.expected_factorized_field_checkpoint_sha256,
            label="factorized field checkpoint",
        ),
        "factorized_state": _require_expected_file(
            args.factorized_primitive_state,
            args.expected_factorized_primitive_state_sha256,
            label="factorized primitive state",
        ),
        "support_graph": _require_expected_file(
            args.support_graph,
            args.expected_support_graph_sha256,
            label="clean support graph",
        ),
        "exact_marginal_responsibility": _require_expected_file(
            args.exact_marginal_responsibility_authority,
            args.expected_exact_marginal_responsibility_authority_sha256,
            label="exact-marginal responsibility authority",
        ),
        "accepted_v2_checkpoint": _require_expected_file(
            args.accepted_v2_checkpoint,
            args.expected_accepted_v2_checkpoint_sha256,
            label="AcceptedV2 checkpoint",
        ),
        "official_radio_checkpoint": _require_expected_file(
            args.official_radio_checkpoint,
            args.expected_official_radio_checkpoint_sha256,
            label="official C-RADIOv4-H checkpoint",
        ),
    }
    if (
        records["accepted_v2_checkpoint"]["sha256"]
        != ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256
        or records["official_radio_checkpoint"]["sha256"]
        != shard.OFFICIAL_RADIO_CHECKPOINT_SHA256
    ):
        raise ValueError("AcceptedV2 or official RADIO singleton authority differs")
    support_bundle, state = load_surface_factorized_state_bundle(
        records["field_checkpoint"]["path"],
        expected_field_checkpoint_sha256=records["field_checkpoint"]["sha256"],
        expected_factorized_radio_cache_sha256=shard._require_sha256(
            args.expected_factorized_radio_cache_sha256,
            label="factorized RADIO cache",
        ),
        state_path=records["factorized_state"]["path"],
        expected_state_sha256=records["factorized_state"]["sha256"],
    )
    if (
        state.schema != FACTORIZED_PRIMITIVE_STATE_SCHEMA
        or state.schema_version != FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
        or state.metadata.get("query_independent") is not True
        or any(
            state.metadata.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("factorized primitive state is not clean source-only v2")
    anchor_visible, selection_authority = _load_exact_anchor_visibility(
        records["exact_marginal_responsibility"]["path"],
        expected_sha256=records["exact_marginal_responsibility"]["sha256"],
        num_gaussians=int(state.valid.numel()),
        xyz_sha256=str(state.metadata["geometry_fingerprint"]["xyz_sha256"]),
    )
    graph, _, _ = load_torch_mapping(
        records["support_graph"]["path"],
        expected_sha256=records["support_graph"]["sha256"],
        map_location="cpu",
        label="clean support graph",
    )
    support, row_authority = _validate_source_only_graph(
        graph,
        state=state,
        field_checkpoint_sha256=records["field_checkpoint"]["sha256"],
        factorized_radio_cache_sha256=support_bundle.cache.sha256,
    )
    wrapped, checkpoint = SurfaceRegionSummaryReadoutV4.from_accepted_v2_checkpoint(
        records["accepted_v2_checkpoint"]["path"], map_location="cpu"
    )
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("AcceptedV2 checkpoint lacks frozen provenance")
    contract = surface_region_contract_from_metadata(
        {
            **provenance,
            "region_contract_version": provenance.get(
                "region_contract_version",
                provenance.get("region_contract", {}).get("version"),
            ),
        }
    )
    if (
        type(contract) is not SurfaceRegionContractV2
        or contract.digest != ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256
    ):
        raise ValueError("AcceptedV2 canonical V2 region contract differs")
    validate_accepted_v2_graph_contract(graph, contract)
    contract.prepare_graph(support, torch.as_tensor(graph["xyz"]).float().cpu())
    input_authority = accepted_region_input_authority(
        state_file_sha256=records["factorized_state"]["sha256"],
        field_checkpoint_sha256=records["field_checkpoint"]["sha256"],
        factorized_radio_cache_sha256=support_bundle.cache.sha256,
        support_graph_sha256=records["support_graph"]["sha256"],
        row_authority=row_authority,
        geometry_fingerprint=state.metadata["geometry_fingerprint"],
        selection_authority=selection_authority,
    )
    shard.validate_accepted_region_input_authority(
        input_authority,
        geometry_fingerprint=state.metadata["geometry_fingerprint"],
    )
    return _Runtime(
        scene_id=str(args.scene_id),
        field=support_bundle.field.cpu().eval().requires_grad_(False),
        state=state,
        graph_payload=graph,
        support=support,
        contract=contract,
        readout=wrapped.base_readout.cpu().eval().requires_grad_(False),
        input_authority=input_authority,
        input_records=records,
        anchor_visible=anchor_visible,
    )


def preflight_summary(
    runtime: _Runtime, selection: Mapping[str, Any]
) -> dict[str, Any]:
    rows = int(runtime.state.global_rows.numel())
    return {
        "status": "ready",
        "scene_id": runtime.scene_id,
        "physical_space_id": canonical_physical_space_id(runtime.scene_id),
        "active_primitive_rows": rows,
        "region_scales": list(runtime.contract.radii_m),
        "canonical_candidate_region_rows": rows * len(runtime.contract.radii_m),
        "selected_region_rows": int(selection["canonical_region_indices"].numel()),
        "selection_audit": dict(selection["selection_audit"]),
        "input_records": dict(runtime.input_records),
        "source_access": shard._authority_access(source_rgb_used=False),
        "semantic_cache_final_payload_used": False,
        "outputs_written": False,
    }


def _expanded_region_batch(
    runtime: _Runtime,
    *,
    prepared: Any,
    xyz: torch.Tensor,
    centers: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    regions = runtime.contract.expand_batch(
        runtime.support,
        xyz,
        centers.tolist(),
        float(radius),
        prepared_graph=prepared,
    )
    width = int(runtime.contract.maximum_tokens)
    local_rows = torch.zeros(len(regions), width, dtype=torch.long)
    mask = torch.zeros(len(regions), width, dtype=torch.bool)
    core = torch.zeros(len(regions), width, dtype=torch.bool)
    anchor = torch.zeros(len(regions), dtype=torch.long)
    for offset, (selected, selected_core, _distance) in enumerate(regions):
        count = int(selected.numel())
        local_rows[offset, :count] = selected
        mask[offset, :count] = True
        core[offset, :count] = selected_core
        positions = torch.where(selected == centers[offset])[0]
        if positions.numel() != 1:
            raise RuntimeError("AcceptedV2 expansion lost its anchor")
        anchor[offset] = int(positions[0])
    return local_rows, mask, core, anchor


def _select_canonical_regions(
    runtime: _Runtime,
    *,
    batch_size: int,
) -> dict[str, Any]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    graph = runtime.graph_payload
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    prepared = runtime.contract.prepare_graph(runtime.support, xyz)
    scale_count = len(runtime.contract.radii_m)
    total = int(global_rows.numel()) * scale_count
    fingerprints = ["0" * 64] * total
    candidate = torch.zeros(total, dtype=torch.bool)
    visible_centers = runtime.anchor_visible[global_rows]
    for scale_index, radius in enumerate(runtime.contract.radii_m):
        centers_for_scale = torch.where(visible_centers)[0]
        canonical_offset = scale_index * int(global_rows.numel())
        candidate[canonical_offset + centers_for_scale] = True
        for start in range(0, centers_for_scale.numel(), size):
            centers = centers_for_scale[start : start + size]
            local_rows, mask, _core, anchor = _expanded_region_batch(
                runtime,
                prepared=prepared,
                xyz=xyz,
                centers=centers,
                radius=float(radius),
            )
            global_region_rows = global_rows[local_rows]
            for row in range(centers.numel()):
                fingerprints[canonical_offset + int(centers[row])] = (
                    region_fingerprint(
                        scene_id=runtime.scene_id,
                        scale_index=scale_index,
                        anchor_global_row=int(
                            global_region_rows[row, int(anchor[row])]
                        ),
                        active_global_rows=global_region_rows[row][mask[row]].tolist(),
                    )
                )
    scale_indices = torch.arange(scale_count, dtype=torch.long).repeat_interleave(
        global_rows.numel()
    )
    selected, selected_by_scale = select_scale_stratified_indices(
        scale_indices, fingerprints, candidate
    )
    return {
        "canonical_region_indices": selected,
        "region_fingerprints": [fingerprints[int(index)] for index in selected],
        "selection_audit": {
            "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
            "canonical_candidate_region_count": total,
            "exact_overlap_candidate_count": total,
            "teacher_visible_candidate_count": int(candidate.sum()),
            "selected_region_count": int(selected.numel()),
            "selected_count_by_scale": selected_by_scale,
        },
    }


def _compute_region_tensors(
    runtime: _Runtime,
    selection: Mapping[str, Any],
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    size = int(batch_size)
    if size <= 0:
        raise ValueError("batch_size must be positive")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("AcceptedV2 materialization requested unavailable CUDA")
    graph = runtime.graph_payload
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    local_sigma = torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4)
    prepared = runtime.contract.prepare_graph(runtime.support, xyz)
    radio = torch.empty(global_rows.numel(), 1280, dtype=torch.float16)
    field = runtime.field.to(compute_device).eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, global_rows.numel(), size):
            stop = min(start + size, global_rows.numel())
            radio[start:stop] = field.radio_features(
                global_rows[start:stop].to(compute_device)
            ).half().cpu()
    reliability = runtime.state.legacy_geometric_reliability().float().cpu()
    if reliability.shape != (global_rows.numel(),):
        raise ValueError("factorized state reliability/global rows differ")
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        runtime.input_records["official_radio_checkpoint"]["path"],
        expected_sha256=runtime.input_records["official_radio_checkpoint"][
            "sha256"
        ],
    ).to(compute_device).eval().requires_grad_(False)
    readout = runtime.readout.to(compute_device).eval().requires_grad_(False)
    # Region topology remains canonical CPU work.  The immutable field,
    # readout, and official head are pure row-wise tensor programs, so moving
    # only those operations to the requested device changes throughput but not
    # the AcceptedV2 mathematical contract or canonical row order.
    radio_device = radio.to(compute_device)
    xyz_device = xyz.to(compute_device)
    sigma_device = local_sigma.to(compute_device)
    reliability_device = reliability.to(compute_device)
    rows_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor] = []
    anchors_out: list[torch.Tensor] = []
    scales_out: list[torch.Tensor] = []
    descriptors_out: list[torch.Tensor] = []
    canonical = torch.as_tensor(selection["canonical_region_indices"]).long().cpu()
    centers_per_scale = int(global_rows.numel())
    with torch.inference_mode():
        for scale_index, radius in enumerate(runtime.contract.radii_m):
            in_scale = canonical[
                torch.div(canonical, centers_per_scale, rounding_mode="floor")
                == scale_index
            ]
            centers_for_scale = torch.remainder(in_scale, centers_per_scale)
            for start in range(0, centers_for_scale.numel(), size):
                centers = centers_for_scale[start : start + size]
                local_rows, mask, core, anchor = _expanded_region_batch(
                    runtime,
                    prepared=prepared,
                    xyz=xyz,
                    centers=centers,
                    radius=float(radius),
                )
                local_rows_device = local_rows.to(compute_device)
                mask_device = mask.to(compute_device)
                core_device = core.to(compute_device)
                anchor_device = anchor.to(compute_device)
                token_xyz = xyz_device[local_rows_device]
                token_scale = sigma_device[local_rows_device, None].expand(-1, -1, 3)
                token_reliability = reliability_device[local_rows_device, None]
                geometry = surface_region_geometry_v2(
                    token_xyz,
                    token_scale,
                    token_reliability,
                    float(radius),
                    anchor_index=anchor_device,
                    core_mask=core_device,
                    token_mask=mask_device,
                )
                summary = readout(
                    radio_device[local_rows_device],
                    geometry,
                    token_mask=mask_device,
                    reliability=token_reliability,
                    anchor_index=anchor_device,
                )
                descriptor = F.normalize(
                    head(summary[:, None])[:, 0].float(), dim=-1
                )
                global_region_rows = global_rows[local_rows]
                global_region_rows[~mask] = -1
                rows_out.append(global_region_rows.contiguous())
                masks_out.append(mask.contiguous())
                anchors_out.append(anchor.contiguous())
                scales_out.append(
                    torch.full((centers.numel(),), scale_index, dtype=torch.long)
                )
                descriptors_out.append(descriptor.cpu().contiguous())
    return {
        "canonical_region_indices": canonical,
        "region_fingerprints": list(selection["region_fingerprints"]),
        "selection_audit": dict(selection["selection_audit"]),
        "region_rows": torch.cat(rows_out, dim=0),
        "token_mask": torch.cat(masks_out, dim=0),
        "anchor_index": torch.cat(anchors_out, dim=0),
        "scale_indices": torch.cat(scales_out, dim=0),
        "accepted_v2_e0": torch.cat(descriptors_out, dim=0),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if not bool(getattr(args, "preflight_only", False)) and (
        output.exists() or output.is_symlink()
    ):
        raise FileExistsError(f"refuses to clobber AcceptedV2 authority: {output}")
    runtime = preflight(args)
    selection_batch_size = int(
        getattr(args, "selection_batch_size", 8192)
    )
    readout_batch_size = int(
        getattr(args, "readout_batch_size", getattr(args, "batch_size", 128))
    )
    selection = _select_canonical_regions(
        runtime, batch_size=selection_batch_size
    )
    summary = preflight_summary(runtime, selection)
    if bool(getattr(args, "preflight_only", False)):
        return summary
    computed = _compute_region_tensors(
        runtime,
        selection,
        batch_size=readout_batch_size,
        device=str(getattr(args, "device", "cpu")),
    )
    payload = build_authority_payload(
        scene_id=runtime.scene_id,
        geometry_fingerprint=runtime.state.metadata["geometry_fingerprint"],
        accepted_base_valid=runtime.state.valid,
        input_authority=runtime.input_authority,
        **computed,
    )
    write_torch_noclobber(output, payload)
    return {
        **summary,
        "status": "materialized",
        "output": file_record(output),
        "outputs_written": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--factorized-field-checkpoint", required=True)
    parser.add_argument(
        "--expected-factorized-field-checkpoint-sha256", required=True
    )
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument(
        "--expected-factorized-primitive-state-sha256", required=True
    )
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--exact-marginal-responsibility-authority", required=True)
    parser.add_argument(
        "--expected-exact-marginal-responsibility-authority-sha256",
        required=True,
    )
    parser.add_argument("--accepted-v2-checkpoint", required=True)
    parser.add_argument(
        "--expected-accepted-v2-checkpoint-sha256", required=True
    )
    parser.add_argument("--official-radio-checkpoint", required=True)
    parser.add_argument(
        "--expected-official-radio-checkpoint-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
    parser.add_argument("--readout-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
