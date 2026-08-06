#!/usr/bin/env python3
"""Materialize a sparse typed-context overlay for AcceptedV2 canonical rows.

The producer consumes caller-SHA-bound clean source authorities only.  It does
not copy or recompute AcceptedV2 ``e0`` and never invokes the AcceptedV2
readout or RADIO's official summary head.  Candidate-cap-hit rows retain a
sparse topology audit, while their context carrier and statistics remain exact
zero.  Publication is first-writer-wins.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
)
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV4
from radio_gs.interfaces.surface_region_typed_context import (
    CandidateCompleteTypedSelection,
    TYPED_CONTEXT_AUTHORITY_SCHEMA,
    TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
    TYPED_CONTEXT_FEATURE_DIM,
    TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
    TYPED_CONTEXT_STATISTIC_DIM,
    candidate_complete_typed_context_batch,
    pool_typed_context_radio,
    typed_context_channel_sha256,
    typed_context_overlay_contract,
    typed_context_source_access,
    validate_typed_context_authority,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.build_surface_region_semantic_cache import (
    load_surface_factorized_state_bundle,
)
from radio_gs.scripts.materialize_accepted_v2_canonical_region_authority import (
    _validate_source_only_graph,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    expected = _require_sha(expected_sha256, label=label)
    observed = sha256_file(source)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs")
    return {"path": str(source), "sha256": observed}


def validate_local_global_row_mapping(
    local_rows: torch.Tensor,
    global_rows: torch.Tensor,
    graph_global_rows: torch.Tensor,
) -> None:
    """Validate one sparse local/global audit against the graph row authority."""

    local = torch.as_tensor(local_rows).detach().long().cpu().reshape(-1)
    global_value = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    authority = torch.as_tensor(graph_global_rows).detach().long().cpu().reshape(-1)
    if (
        local.shape != global_value.shape
        or bool((local < 0).any())
        or bool((local >= authority.numel()).any())
        or not torch.equal(authority[local], global_value)
    ):
        raise ValueError("typed-context local/global primitive row mapping differs")


def _accepted_anchor_rows(
    accepted: Mapping[str, Any],
    graph_global_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    region_rows = torch.as_tensor(accepted["region_rows"]).long().cpu()
    anchor_index = torch.as_tensor(accepted["anchor_index"]).long().cpu()
    region_count = int(region_rows.shape[0])
    anchor_global = region_rows[torch.arange(region_count), anchor_index]
    accepted_valid = torch.as_tensor(accepted["accepted_base_valid"]).bool().cpu()
    global_to_local = torch.full(
        (accepted_valid.numel(),), -1, dtype=torch.long
    )
    graph_global = torch.as_tensor(graph_global_rows).long().cpu()
    if (
        bool((graph_global < 0).any())
        or bool((graph_global >= accepted_valid.numel()).any())
        or int(torch.unique(graph_global).numel()) != graph_global.numel()
    ):
        raise ValueError("support graph global-row authority differs")
    global_to_local[graph_global] = torch.arange(graph_global.numel())
    anchor_local = global_to_local[anchor_global]
    if bool((anchor_local < 0).any()):
        raise ValueError("AcceptedV2 anchor is absent from typed-context graph")
    validate_local_global_row_mapping(anchor_local, anchor_global, graph_global)
    return anchor_local, anchor_global


def materialize_candidate_selections(
    contract: SurfaceRegionContractV4,
    prepared_graph: Any,
    anchor_local_rows: torch.Tensor,
    scale_indices: torch.Tensor,
    *,
    batch_size: int,
) -> list[CandidateCompleteTypedSelection]:
    """Expand all Accepted rows while preserving their exact canonical order."""

    anchors = torch.as_tensor(anchor_local_rows).long().cpu().reshape(-1)
    scales = torch.as_tensor(scale_indices).long().cpu().reshape(-1)
    size = int(batch_size)
    if anchors.shape != scales.shape or size <= 0:
        raise ValueError("typed-context candidate materialization inputs differ")
    if bool((scales < 0).any()) or bool((scales >= len(contract.radii_m)).any()):
        raise ValueError("typed-context scale index is outside the V4 contract")
    results: list[CandidateCompleteTypedSelection | None] = [None] * anchors.numel()
    for scale_index, radius in enumerate(contract.radii_m):
        region_indices = torch.where(scales == scale_index)[0]
        for start in range(0, region_indices.numel(), size):
            selected_regions = region_indices[start : start + size]
            selected = candidate_complete_typed_context_batch(
                contract,
                prepared_graph,
                anchors[selected_regions],
                float(radius),
            )
            for region, value in zip(selected_regions.tolist(), selected):
                results[region] = value
    if any(value is None for value in results):
        raise RuntimeError("typed-context candidate materialization missed a row")
    return [value for value in results if value is not None]


def _decode_unique_local_rows(
    field: torch.nn.Module,
    graph_global_rows: torch.Tensor,
    required_local_rows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = torch.as_tensor(required_local_rows).long().cpu().reshape(-1)
    if required.numel() == 0:
        return required, torch.empty(0, TYPED_CONTEXT_FEATURE_DIM)
    unique = torch.unique(required, sorted=True)
    graph_global = torch.as_tensor(graph_global_rows).long().cpu()
    if bool((unique < 0).any()) or bool((unique >= graph_global.numel()).any()):
        raise ValueError("typed-context decode row is outside graph support")
    size = int(batch_size)
    if size <= 0:
        raise ValueError("typed-context field batch size must be positive")
    model = field.to(device).eval().requires_grad_(False)
    decoded = torch.empty(unique.numel(), TYPED_CONTEXT_FEATURE_DIM)
    with torch.inference_mode():
        for start in range(0, unique.numel(), size):
            stop = min(start + size, unique.numel())
            values = model.radio_features(
                graph_global[unique[start:stop]].to(device)
            ).float()
            if values.shape != (stop - start, TYPED_CONTEXT_FEATURE_DIM) or not bool(
                torch.isfinite(values).all()
            ):
                raise ValueError("typed-context field decode differs")
            decoded[start:stop] = values.cpu()
    return unique, decoded


def _pool_candidate_rows(
    selections: Sequence[CandidateCompleteTypedSelection],
    *,
    radii_m: Sequence[float],
    context_ratio: float,
    scale_indices: torch.Tensor,
    anchor_local_rows: torch.Tensor,
    graph_global_rows: torch.Tensor,
    reliability: torch.Tensor,
    field: torch.nn.Module,
    field_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    scales = torch.as_tensor(scale_indices).long().cpu()
    anchors = torch.as_tensor(anchor_local_rows).long().cpu()
    graph_global = torch.as_tensor(graph_global_rows).long().cpu()
    if len(selections) != scales.numel() or anchors.shape != scales.shape:
        raise ValueError("typed-context pooling rows are not aligned")
    audit_local: list[torch.Tensor] = []
    audit_global: list[torch.Tensor] = []
    counts: list[int] = []
    required_decode: list[torch.Tensor] = []
    for anchor, selection in zip(anchors.tolist(), selections):
        context_local = selection.rows[selection.context_mask]
        context_global = graph_global[context_local]
        audit_local.append(context_local)
        audit_global.append(context_global)
        counts.append(int(context_local.numel()))
        if selection.search_complete and context_local.numel() > 0:
            required_decode.extend(
                (context_local, torch.tensor([anchor], dtype=torch.long))
            )
    required = (
        torch.cat(required_decode)
        if required_decode
        else torch.empty(0, dtype=torch.long)
    )
    decoded_local, decoded = _decode_unique_local_rows(
        field,
        graph_global,
        required,
        batch_size=field_batch_size,
        device=device,
    )
    local_to_decode = torch.full((graph_global.numel(),), -1, dtype=torch.long)
    if decoded_local.numel():
        local_to_decode[decoded_local] = torch.arange(decoded_local.numel())
    direction_rows: list[torch.Tensor] = []
    statistic_rows: list[torch.Tensor] = []
    present: list[bool] = []
    valid: list[bool] = []
    for region, selection in enumerate(selections):
        context_local = audit_local[region]
        context_distance = selection.semantic_geodesic_distance[
            selection.context_mask
        ]
        radius = float(radii_m[int(scales[region])])
        if selection.search_complete and context_local.numel() > 0:
            context_decode = local_to_decode[context_local]
            anchor_decode = local_to_decode[anchors[region : region + 1]]
            if bool((context_decode < 0).any()) or bool((anchor_decode < 0).any()):
                raise RuntimeError("typed-context unique field decode missed a row")
            pooled = pool_typed_context_radio(
                decoded[context_decode],
                torch.as_tensor(reliability).float().cpu()[context_local],
                context_distance,
                raw_anchor_radio=decoded[int(anchor_decode[0])],
                radius_m=radius,
                selected_semantic_token_count=int(selection.rows.numel()),
                search_complete=True,
                context_ratio=float(context_ratio),
            )
        else:
            pooled = pool_typed_context_radio(
                torch.empty(0, TYPED_CONTEXT_FEATURE_DIM),
                torch.empty(0),
                torch.empty(0),
                raw_anchor_radio=torch.ones(TYPED_CONTEXT_FEATURE_DIM),
                radius_m=radius,
                selected_semantic_token_count=int(selection.rows.numel()),
                search_complete=selection.search_complete,
            )
            if context_local.numel() > 0:
                # The empty call above creates the required exact-zero carrier;
                # preserve the observed context-present audit on cap-hit rows.
                pooled = type(pooled)(
                    pooled.direction,
                    pooled.statistics,
                    True,
                    False,
                )
        direction_rows.append(pooled.direction)
        statistic_rows.append(pooled.statistics)
        present.append(bool(context_local.numel() > 0))
        valid.append(bool(pooled.pool_valid))
    offsets = torch.zeros(len(selections) + 1, dtype=torch.long)
    offsets[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    flat_local = (
        torch.cat(audit_local)
        if audit_local and int(offsets[-1])
        else torch.empty(0, dtype=torch.long)
    )
    flat_global = (
        torch.cat(audit_global)
        if audit_global and int(offsets[-1])
        else torch.empty(0, dtype=torch.long)
    )
    validate_local_global_row_mapping(flat_local, flat_global, graph_global)
    return {
        "pooled_context_radio_direction": torch.stack(direction_rows),
        "typed_context_statistics": torch.stack(statistic_rows),
        "context_present": torch.tensor(present, dtype=torch.bool),
        "typed_context_valid": torch.tensor(valid, dtype=torch.bool),
        "context_token_count": torch.tensor(counts, dtype=torch.long),
        "context_token_row_offsets": offsets,
        "context_token_local_rows": flat_local,
        "context_token_global_rows": flat_global,
    }


def assemble_authority_payload(
    *,
    accepted: Mapping[str, Any],
    accepted_file: Mapping[str, str],
    field_file: Mapping[str, str],
    state_file: Mapping[str, str],
    factorized_radio_cache_sha256: str,
    graph_file: Mapping[str, str],
    primitive_row_authority_sha256: str,
    anchor_local_rows: torch.Tensor,
    anchor_global_rows: torch.Tensor,
    selections: Sequence[CandidateCompleteTypedSelection],
    pooled: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble and independently validate the no-e0 typed-context payload."""

    scene = str(accepted["scene_id"])
    fingerprints = list(accepted["region_fingerprints"])
    regions = len(fingerprints)
    if len(selections) != regions:
        raise ValueError("typed-context selections and Accepted rows differ")
    payload: dict[str, Any] = {
        "schema": TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "schema_version": TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
        "contract": typed_context_overlay_contract(),
        "contract_sha256": TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
        "scene_id": scene,
        "physical_space_id": str(accepted["physical_space_id"]),
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "accepted_v2_canonical_region_authority": dict(accepted_file),
            "accepted_region_channel_sha256": canonical_json_sha256(
                accepted["channel_sha256"]
            ),
            "accepted_region_fingerprints_sha256": canonical_json_sha256(
                fingerprints
            ),
            "factorized_field_checkpoint": dict(field_file),
            "factorized_primitive_state": dict(state_file),
            "factorized_radio_cache_sha256": _require_sha(
                factorized_radio_cache_sha256,
                label="typed-context factorized RADIO cache",
            ),
            "support_graph": dict(graph_file),
            "primitive_row_authority_sha256": _require_sha(
                primitive_row_authority_sha256,
                label="typed-context primitive row authority",
            ),
        },
        "region_row_ids": [
            shard.stable_region_id(scene, fingerprint)
            for fingerprint in fingerprints
        ],
        "canonical_region_indices": torch.as_tensor(
            accepted["canonical_region_indices"]
        ).long().cpu().contiguous(),
        "scale_indices": torch.as_tensor(accepted["scale_indices"])
        .long()
        .cpu()
        .contiguous(),
        "anchor_local_rows": torch.as_tensor(anchor_local_rows)
        .long()
        .cpu()
        .contiguous(),
        "anchor_global_rows": torch.as_tensor(anchor_global_rows)
        .long()
        .cpu()
        .contiguous(),
        "pooled_context_radio_direction": torch.as_tensor(
            pooled["pooled_context_radio_direction"]
        )
        .half()
        .cpu()
        .contiguous(),
        "typed_context_statistics": torch.as_tensor(
            pooled["typed_context_statistics"]
        )
        .float()
        .cpu()
        .contiguous(),
        "context_present": torch.as_tensor(pooled["context_present"])
        .bool()
        .cpu()
        .contiguous(),
        "candidate_search_complete": torch.tensor(
            [selection.search_complete for selection in selections],
            dtype=torch.bool,
        ),
        "typed_context_valid": torch.as_tensor(pooled["typed_context_valid"])
        .bool()
        .cpu()
        .contiguous(),
        "candidate_termination": [
            selection.termination for selection in selections
        ],
        "candidate_probe_count": torch.tensor(
            [selection.candidate_probe_count for selection in selections],
            dtype=torch.long,
        ),
        "context_token_count": torch.as_tensor(pooled["context_token_count"])
        .long()
        .cpu()
        .contiguous(),
        "context_token_row_offsets": torch.as_tensor(
            pooled["context_token_row_offsets"]
        )
        .long()
        .cpu()
        .contiguous(),
        "context_token_local_rows": torch.as_tensor(
            pooled["context_token_local_rows"]
        )
        .long()
        .cpu()
        .contiguous(),
        "context_token_global_rows": torch.as_tensor(
            pooled["context_token_global_rows"]
        )
        .long()
        .cpu()
        .contiguous(),
        "source_access": typed_context_source_access(),
    }
    payload["channel_sha256"] = typed_context_channel_sha256(payload)
    return validate_typed_context_authority(payload)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber typed-context authority: {output}")
    records = {
        "accepted": _require_file(
            args.accepted_v2_authority,
            args.expected_accepted_v2_authority_sha256,
            label="AcceptedV2 canonical region authority",
        ),
        "field": _require_file(
            args.factorized_field_checkpoint,
            args.expected_factorized_field_checkpoint_sha256,
            label="factorized field checkpoint",
        ),
        "state": _require_file(
            args.factorized_primitive_state,
            args.expected_factorized_primitive_state_sha256,
            label="factorized primitive state",
        ),
        "graph": _require_file(
            args.support_graph,
            args.expected_support_graph_sha256,
            label="query-free support graph",
        ),
    }
    accepted_raw, _, _ = load_torch_mapping(
        records["accepted"]["path"],
        expected_sha256=records["accepted"]["sha256"],
        map_location="cpu",
        label="AcceptedV2 canonical region authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    if str(accepted["scene_id"]) != str(args.scene_id):
        raise ValueError("typed-context scene differs from AcceptedV2 authority")
    expected_cache = _require_sha(
        args.expected_factorized_radio_cache_sha256,
        label="factorized RADIO cache",
    )
    accepted_inputs = accepted["input_authority"]
    geometry_input = accepted_inputs["geometry_authority"]
    support_input = accepted_inputs["support_graph_authority"]
    if (
        geometry_input["factorized_primitive_state_file_sha256"]
        != records["state"]["sha256"]
        or geometry_input["factorized_field_checkpoint_file_sha256"]
        != records["field"]["sha256"]
        or geometry_input["factorized_radio_cache_file_sha256"] != expected_cache
        or support_input["support_graph_file_sha256"]
        != records["graph"]["sha256"]
    ):
        raise ValueError("typed-context caller files differ from Accepted lineage")
    support_bundle, state = load_surface_factorized_state_bundle(
        records["field"]["path"],
        expected_field_checkpoint_sha256=records["field"]["sha256"],
        expected_factorized_radio_cache_sha256=expected_cache,
        state_path=records["state"]["path"],
        expected_state_sha256=records["state"]["sha256"],
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
        or state.metadata["geometry_fingerprint"]
        != accepted["geometry_fingerprint"]
    ):
        raise ValueError("typed-context factorized state is not clean aligned v2")
    graph_payload, _, _ = load_torch_mapping(
        records["graph"]["path"],
        expected_sha256=records["graph"]["sha256"],
        map_location="cpu",
        label="typed-context support graph",
    )
    support, row_authority = _validate_source_only_graph(
        graph_payload,
        state=state,
        field_checkpoint_sha256=records["field"]["sha256"],
        factorized_radio_cache_sha256=support_bundle.cache.sha256,
    )
    if (
        row_authority.digest
        != support_input["primitive_row_authority_sha256"]
        or row_authority.digest
        != geometry_input["primitive_row_authority_sha256"]
    ):
        raise ValueError("typed-context primitive row authority differs")
    graph_global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    anchor_local, anchor_global = _accepted_anchor_rows(
        accepted, graph_global_rows
    )
    contract = SurfaceRegionContractV4()
    xyz = torch.as_tensor(graph_payload["xyz"]).float().cpu()
    prepared = contract.prepare_graph(support, xyz)
    selections = materialize_candidate_selections(
        contract,
        prepared,
        anchor_local,
        accepted["scale_indices"],
        batch_size=int(args.candidate_batch_size),
    )
    reliability = state.legacy_geometric_reliability().float().cpu()
    if reliability.shape != (graph_global_rows.numel(),):
        raise ValueError("typed-context reliability and graph rows differ")
    pooled = _pool_candidate_rows(
        selections,
        radii_m=contract.radii_m,
        context_ratio=contract.context_ratio,
        scale_indices=accepted["scale_indices"],
        anchor_local_rows=anchor_local,
        graph_global_rows=graph_global_rows,
        reliability=reliability,
        field=support_bundle.field,
        field_batch_size=int(args.field_batch_size),
        device=torch.device(args.device),
    )
    payload = assemble_authority_payload(
        accepted=accepted,
        accepted_file=records["accepted"],
        field_file=records["field"],
        state_file=records["state"],
        factorized_radio_cache_sha256=expected_cache,
        graph_file=records["graph"],
        primitive_row_authority_sha256=row_authority.digest,
        anchor_local_rows=anchor_local,
        anchor_global_rows=anchor_global,
        selections=selections,
        pooled=pooled,
    )
    write_torch_noclobber(output, payload)
    output_record = file_record(output)
    complete = payload["candidate_search_complete"]
    present = payload["context_present"]
    valid = payload["typed_context_valid"]
    scales = payload["scale_indices"]
    per_scale = {}
    for scale_index, radius in enumerate(contract.radii_m):
        selected = scales == scale_index
        count = int(selected.sum())
        per_scale[str(radius)] = {
            "regions": count,
            "context_present": int((present & selected).sum()),
            "search_complete": int((complete & selected).sum()),
            "typed_context_valid": int((valid & selected).sum()),
        }
    return {
        "status": "materialized",
        "schema": TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "scene_id": str(args.scene_id),
        "regions": int(scales.numel()),
        "context_present": int(present.sum()),
        "search_complete": int(complete.sum()),
        "typed_context_valid": int(valid.sum()),
        "candidate_cap_reached": int((~complete).sum()),
        "context_tokens_audited": int(payload["context_token_row_offsets"][-1]),
        "per_scale": per_scale,
        "output": output_record,
        "accepted_v2_e0_copied": False,
        "source_access": typed_context_source_access(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--accepted-v2-authority", required=True)
    parser.add_argument("--expected-accepted-v2-authority-sha256", required=True)
    parser.add_argument("--factorized-field-checkpoint", required=True)
    parser.add_argument("--expected-factorized-field-checkpoint-sha256", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--expected-factorized-primitive-state-sha256", required=True)
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    parser.add_argument("--field-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
