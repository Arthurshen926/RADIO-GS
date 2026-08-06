#!/usr/bin/env python3
"""Materialize the v2 adaptive typed-budget context authority.

The inputs and AcceptedV2 row alignment are identical to Stage A.  Only the
versioned context selection overlay changes: a 192-core/64-context V4
parameterization adaptively expands its Dijkstra probe until the final bounded
256-token typed selection is proved immutable.  The producer never copies e0,
invokes the Accepted readout, or applies the official summary head.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
)
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_FEATURE_DIM,
    pool_typed_context_radio,
    typed_context_source_access,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_MAX_BATCH_SIZE,
    ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
    ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
    ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
    ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
    AdaptiveTypedBudgetBatch,
    AdaptiveTypedBudgetSelection,
    adaptive_typed_budget_context_batch,
    adaptive_typed_context_channel_sha256,
    adaptive_typed_context_overlay_contract,
    adaptive_typed_context_v4_contract,
    validate_adaptive_typed_context_authority,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.build_surface_region_semantic_cache import (
    load_surface_factorized_state_bundle,
)
from radio_gs.scripts.materialize_accepted_v2_canonical_region_authority import (
    _validate_source_only_graph,
)
from radio_gs.scripts.materialize_accepted_v2_typed_context_authority import (
    _accepted_anchor_rows,
    _decode_unique_local_rows,
    _require_file,
    _require_sha,
    validate_local_global_row_mapping,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_torch_noclobber,
)


def materialize_adaptive_selections(
    prepared_graph: Any,
    anchor_local_rows: torch.Tensor,
    scale_indices: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[list[AdaptiveTypedBudgetSelection], dict[str, int]]:
    contract = adaptive_typed_context_v4_contract()
    anchors = torch.as_tensor(anchor_local_rows).long().cpu().reshape(-1)
    scales = torch.as_tensor(scale_indices).long().cpu().reshape(-1)
    if anchors.shape != scales.shape:
        raise ValueError("adaptive typed-context anchor/scale rows differ")
    results: list[AdaptiveTypedBudgetSelection | None] = [None] * anchors.numel()
    maximum_estimated = 0
    for scale_index, radius in enumerate(contract.radii_m):
        region_indices = torch.where(scales == scale_index)[0]
        if region_indices.numel() == 0:
            continue
        batch = adaptive_typed_budget_context_batch(
            contract,
            prepared_graph,
            anchors[region_indices],
            float(radius),
            batch_size=int(batch_size),
            memory_ceiling_bytes=ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
        )
        maximum_estimated = max(
            maximum_estimated, int(batch.maximum_estimated_working_bytes)
        )
        for region, selection in zip(region_indices.tolist(), batch.selections):
            results[region] = selection
    if any(value is None for value in results):
        raise RuntimeError("adaptive typed-context missed an Accepted row")
    return (
        [value for value in results if value is not None],
        {
            "memory_ceiling_bytes": ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
            "maximum_estimated_working_bytes": maximum_estimated,
            "requested_batch_size": int(batch_size),
        },
    )


def _pool_adaptive_rows(
    selections: Sequence[AdaptiveTypedBudgetSelection],
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
    reliability_value = torch.as_tensor(reliability).float().cpu()
    if (
        len(selections) != scales.numel()
        or anchors.shape != scales.shape
        or reliability_value.shape != (graph_global.numel(),)
    ):
        raise ValueError("adaptive typed-context pooling rows differ")
    local_by_region: list[torch.Tensor] = []
    global_by_region: list[torch.Tensor] = []
    required_decode: list[torch.Tensor] = []
    for anchor, selection in zip(anchors.tolist(), selections):
        context_local = selection.rows[selection.context_mask]
        local_by_region.append(context_local)
        global_by_region.append(graph_global[context_local])
        if context_local.numel():
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
        batch_size=int(field_batch_size),
        device=device,
    )
    local_to_decode = torch.full((graph_global.numel(),), -1, dtype=torch.long)
    if decoded_local.numel():
        local_to_decode[decoded_local] = torch.arange(decoded_local.numel())
    directions: list[torch.Tensor] = []
    statistics: list[torch.Tensor] = []
    valid: list[bool] = []
    counts: list[int] = []
    for region, selection in enumerate(selections):
        context_local = local_by_region[region]
        count = int(context_local.numel())
        counts.append(count)
        radius = float(radii_m[int(scales[region])])
        if count:
            context_decode = local_to_decode[context_local]
            anchor_decode = local_to_decode[anchors[region : region + 1]]
            if bool((context_decode < 0).any()) or bool((anchor_decode < 0).any()):
                raise RuntimeError("adaptive typed-context field decode missed a row")
            pooled = pool_typed_context_radio(
                decoded[context_decode],
                reliability_value[context_local],
                selection.semantic_geodesic_distance[selection.context_mask],
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
                search_complete=True,
                context_ratio=float(context_ratio),
            )
        directions.append(pooled.direction)
        statistics.append(pooled.statistics)
        valid.append(bool(pooled.pool_valid))
    offsets = torch.zeros(len(selections) + 1, dtype=torch.long)
    offsets[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    flat_local = (
        torch.cat(local_by_region)
        if int(offsets[-1])
        else torch.empty(0, dtype=torch.long)
    )
    flat_global = (
        torch.cat(global_by_region)
        if int(offsets[-1])
        else torch.empty(0, dtype=torch.long)
    )
    validate_local_global_row_mapping(flat_local, flat_global, graph_global)
    return {
        "pooled_context_radio_direction": torch.stack(directions),
        "typed_context_statistics": torch.stack(statistics),
        "context_present": torch.tensor(counts, dtype=torch.long) > 0,
        "typed_context_valid": torch.tensor(valid, dtype=torch.bool),
        "context_token_count": torch.tensor(counts, dtype=torch.long),
        "context_token_row_offsets": offsets,
        "context_token_local_rows": flat_local,
        "context_token_global_rows": flat_global,
    }


def assemble_adaptive_authority_payload(
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
    selections: Sequence[AdaptiveTypedBudgetSelection],
    memory_audit: Mapping[str, int],
    pooled: Mapping[str, Any],
) -> dict[str, Any]:
    scene = str(accepted["scene_id"])
    fingerprints = list(accepted["region_fingerprints"])
    if len(selections) != len(fingerprints):
        raise ValueError("adaptive typed-context selection/Accepted rows differ")
    payload: dict[str, Any] = {
        "schema": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "schema_version": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA_VERSION,
        "contract": adaptive_typed_context_overlay_contract(),
        "contract_sha256": ADAPTIVE_TYPED_CONTEXT_OVERLAY_CONTRACT_SHA256,
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
                label="adaptive typed-context factorized RADIO cache",
            ),
            "support_graph": dict(graph_file),
            "primitive_row_authority_sha256": _require_sha(
                primitive_row_authority_sha256,
                label="adaptive typed-context primitive row authority",
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
        .long().cpu().contiguous(),
        "anchor_local_rows": torch.as_tensor(anchor_local_rows)
        .long().cpu().contiguous(),
        "anchor_global_rows": torch.as_tensor(anchor_global_rows)
        .long().cpu().contiguous(),
        "pooled_context_radio_direction": torch.as_tensor(
            pooled["pooled_context_radio_direction"]
        ).half().cpu().contiguous(),
        "typed_context_statistics": torch.as_tensor(
            pooled["typed_context_statistics"]
        ).float().cpu().contiguous(),
        "context_present": torch.as_tensor(pooled["context_present"])
        .bool().cpu().contiguous(),
        "selection_complete": torch.ones(len(selections), dtype=torch.bool),
        "typed_context_valid": torch.as_tensor(pooled["typed_context_valid"])
        .bool().cpu().contiguous(),
        "candidate_termination": [selection.termination for selection in selections],
        "final_probe_width": torch.tensor(
            [selection.final_probe_width for selection in selections], dtype=torch.long
        ),
        "settled_candidate_count": torch.tensor(
            [selection.settled_candidate_count for selection in selections],
            dtype=torch.long,
        ),
        "adaptive_round_count": torch.tensor(
            [selection.adaptive_round_count for selection in selections],
            dtype=torch.long,
        ),
        "context_token_count": torch.as_tensor(pooled["context_token_count"])
        .long().cpu().contiguous(),
        "context_token_row_offsets": torch.as_tensor(
            pooled["context_token_row_offsets"]
        ).long().cpu().contiguous(),
        "context_token_local_rows": torch.as_tensor(
            pooled["context_token_local_rows"]
        ).long().cpu().contiguous(),
        "context_token_global_rows": torch.as_tensor(
            pooled["context_token_global_rows"]
        ).long().cpu().contiguous(),
        "memory_audit": {
            "memory_ceiling_bytes": int(memory_audit["memory_ceiling_bytes"]),
            "maximum_estimated_working_bytes": int(
                memory_audit["maximum_estimated_working_bytes"]
            ),
            "requested_batch_size": int(memory_audit["requested_batch_size"]),
        },
        "source_access": typed_context_source_access(),
    }
    payload["channel_sha256"] = adaptive_typed_context_channel_sha256(payload)
    return validate_adaptive_typed_context_authority(payload)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refuses to clobber adaptive typed-context authority: {output}"
        )
    candidate_batch_size = int(args.candidate_batch_size)
    field_batch_size = int(args.field_batch_size)
    if not 0 < candidate_batch_size <= ADAPTIVE_MAX_BATCH_SIZE:
        raise ValueError("adaptive typed-context candidate batch exceeds strict maximum")
    if field_batch_size <= 0:
        raise ValueError("adaptive typed-context field batch must be positive")
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
        raise ValueError("adaptive typed-context scene differs from AcceptedV2")
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
        raise ValueError("adaptive typed-context caller lineage differs")
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
        raise ValueError("adaptive typed-context state is not clean aligned v2")
    graph_payload, _, _ = load_torch_mapping(
        records["graph"]["path"],
        expected_sha256=records["graph"]["sha256"],
        map_location="cpu",
        label="adaptive typed-context support graph",
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
        raise ValueError("adaptive typed-context row authority differs")
    graph_global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    anchor_local, anchor_global = _accepted_anchor_rows(
        accepted, graph_global_rows
    )
    contract = adaptive_typed_context_v4_contract()
    prepared = contract.prepare_graph(
        support, torch.as_tensor(graph_payload["xyz"]).float().cpu()
    )
    selections, memory_audit = materialize_adaptive_selections(
        prepared,
        anchor_local,
        accepted["scale_indices"],
        batch_size=candidate_batch_size,
    )
    pooled = _pool_adaptive_rows(
        selections,
        radii_m=contract.radii_m,
        context_ratio=contract.context_ratio,
        scale_indices=accepted["scale_indices"],
        anchor_local_rows=anchor_local,
        graph_global_rows=graph_global_rows,
        reliability=state.legacy_geometric_reliability().float().cpu(),
        field=support_bundle.field,
        field_batch_size=field_batch_size,
        device=torch.device(args.device),
    )
    payload = assemble_adaptive_authority_payload(
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
        memory_audit=memory_audit,
        pooled=pooled,
    )
    write_torch_noclobber(output, payload)
    output_record = file_record(output)
    scales = payload["scale_indices"]
    present = payload["context_present"]
    valid = payload["typed_context_valid"]
    per_scale = {}
    for scale_index, radius in enumerate(contract.radii_m):
        selected = scales == scale_index
        per_scale[str(radius)] = {
            "regions": int(selected.sum()),
            "context_present": int((present & selected).sum()),
            "typed_context_valid": int((valid & selected).sum()),
        }
    return {
        "status": "materialized",
        "schema": ADAPTIVE_TYPED_CONTEXT_AUTHORITY_SCHEMA,
        "scene_id": str(args.scene_id),
        "regions": int(scales.numel()),
        "selection_complete": int(scales.numel()),
        "context_present": int(present.sum()),
        "typed_context_valid": int(valid.sum()),
        "termination_counts": {
            name: payload["candidate_termination"].count(name)
            for name in sorted(set(payload["candidate_termination"]))
        },
        "maximum_probe_width": int(payload["final_probe_width"].max()),
        "maximum_settled_candidates": int(
            payload["settled_candidate_count"].max()
        ),
        "maximum_adaptive_rounds": int(payload["adaptive_round_count"].max()),
        "context_tokens_audited": int(payload["context_token_row_offsets"][-1]),
        "memory_audit": dict(payload["memory_audit"]),
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
    parser.add_argument(
        "--candidate-batch-size", type=int, default=ADAPTIVE_MAX_BATCH_SIZE
    )
    parser.add_argument("--field-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
