#!/usr/bin/env python3
"""Materialize adaptive typed context over target AcceptedV2 canonical rows."""

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
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.interfaces.surface_region_target_adaptive_typed_context import (
    TARGET_ADAPTIVE_TYPED_CONTEXT_CONTRACT_SHA256,
    TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA,
    TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION,
    target_adaptive_access_audit,
    target_adaptive_typed_context_contract,
    validate_target_adaptive_typed_context_authority,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_MAX_BATCH_SIZE,
    adaptive_typed_context_channel_sha256,
    adaptive_typed_context_v4_contract,
)
from radio_gs.scripts import (
    materialize_accepted_v2_typed_context_adaptive_authority as source,
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
    _require_file,
    _require_sha,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    write_torch_noclobber,
)


def assemble_target_adaptive_authority_payload(
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
    selections: Sequence[Any],
    memory_audit: Mapping[str, int],
    pooled: Mapping[str, Any],
    producer: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    scene = str(accepted["scene_id"])
    fingerprints = list(accepted["region_fingerprints"])
    if len(selections) != len(fingerprints):
        raise ValueError("target adaptive selection/Accepted rows differ")
    payload: dict[str, Any] = {
        "schema": TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA,
        "schema_version": TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA_VERSION,
        "contract": target_adaptive_typed_context_contract(),
        "contract_sha256": TARGET_ADAPTIVE_TYPED_CONTEXT_CONTRACT_SHA256,
        "scene_id": scene,
        "physical_space_id": str(accepted["physical_space_id"]),
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": dict(
            producer if producer is not None else file_record(Path(__file__).resolve())
        ),
        "input_authority": {
            "accepted_v2_canonical_region_authority": dict(accepted_file),
            "accepted_region_channel_sha256": canonical_json_sha256(
                accepted["channel_sha256"]
            ),
            "accepted_region_fingerprints_sha256": canonical_json_sha256(fingerprints),
            "factorized_field_checkpoint": dict(field_file),
            "factorized_primitive_state": dict(state_file),
            "factorized_radio_cache_sha256": _require_sha(
                factorized_radio_cache_sha256,
                label="target adaptive factorized RADIO cache",
            ),
            "support_graph": dict(graph_file),
            "primitive_row_authority_sha256": _require_sha(
                primitive_row_authority_sha256,
                label="target adaptive primitive row authority",
            ),
        },
        "region_row_ids": [
            shard.stable_region_id(scene, fingerprint) for fingerprint in fingerprints
        ],
        "canonical_region_indices": torch.as_tensor(
            accepted["canonical_region_indices"]
        )
        .long()
        .cpu()
        .contiguous(),
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
        "typed_context_statistics": torch.as_tensor(pooled["typed_context_statistics"])
        .float()
        .cpu()
        .contiguous(),
        "context_present": torch.as_tensor(pooled["context_present"])
        .bool()
        .cpu()
        .contiguous(),
        "selection_complete": torch.ones(len(selections), dtype=torch.bool),
        "typed_context_valid": torch.as_tensor(pooled["typed_context_valid"])
        .bool()
        .cpu()
        .contiguous(),
        "candidate_termination": [item.termination for item in selections],
        "final_probe_width": torch.tensor(
            [item.final_probe_width for item in selections], dtype=torch.long
        ),
        "settled_candidate_count": torch.tensor(
            [item.settled_candidate_count for item in selections], dtype=torch.long
        ),
        "adaptive_round_count": torch.tensor(
            [item.adaptive_round_count for item in selections], dtype=torch.long
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
        "context_token_local_rows": torch.as_tensor(pooled["context_token_local_rows"])
        .long()
        .cpu()
        .contiguous(),
        "context_token_global_rows": torch.as_tensor(
            pooled["context_token_global_rows"]
        )
        .long()
        .cpu()
        .contiguous(),
        "memory_audit": {
            "memory_ceiling_bytes": int(memory_audit["memory_ceiling_bytes"]),
            "maximum_estimated_working_bytes": int(
                memory_audit["maximum_estimated_working_bytes"]
            ),
            "requested_batch_size": int(memory_audit["requested_batch_size"]),
        },
        "access_audit": target_adaptive_access_audit(),
    }
    payload["channel_sha256"] = adaptive_typed_context_channel_sha256(payload)
    return validate_target_adaptive_typed_context_authority(payload)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refuses to clobber target adaptive typed-context authority: {output}"
        )
    candidate_batch_size = int(args.candidate_batch_size)
    field_batch_size = int(args.field_batch_size)
    if not 0 < candidate_batch_size <= ADAPTIVE_MAX_BATCH_SIZE:
        raise ValueError("target adaptive candidate batch exceeds strict maximum")
    if field_batch_size <= 0:
        raise ValueError("target adaptive field batch must be positive")
    records = {
        "accepted": _require_file(
            args.target_accepted_v2_authority,
            args.expected_target_accepted_v2_authority_sha256,
            label="target AcceptedV2 canonical region authority",
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
        label="target AcceptedV2 canonical region authority",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    expected_cache = _require_sha(
        args.expected_factorized_radio_cache_sha256,
        label="factorized RADIO cache",
    )
    geometry_input = accepted["input_authority"]["geometry_authority"]
    support_input = accepted["input_authority"]["support_graph_authority"]
    if (
        geometry_input["factorized_primitive_state_file_sha256"]
        != records["state"]["sha256"]
        or geometry_input["factorized_field_checkpoint_file_sha256"]
        != records["field"]["sha256"]
        or geometry_input["factorized_radio_cache_file_sha256"] != expected_cache
        or support_input["support_graph_file_sha256"] != records["graph"]["sha256"]
    ):
        raise ValueError("target adaptive caller lineage differs")
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
        or state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
    ):
        raise ValueError("target adaptive factorized state differs")
    graph_payload, _, _ = load_torch_mapping(
        records["graph"]["path"],
        expected_sha256=records["graph"]["sha256"],
        map_location="cpu",
        label="target adaptive support graph",
    )
    support, row_authority = _validate_source_only_graph(
        graph_payload,
        state=state,
        field_checkpoint_sha256=records["field"]["sha256"],
        factorized_radio_cache_sha256=support_bundle.cache.sha256,
    )
    if (
        row_authority.digest != support_input["primitive_row_authority_sha256"]
        or row_authority.digest != geometry_input["primitive_row_authority_sha256"]
    ):
        raise ValueError("target adaptive primitive row authority differs")
    graph_global_rows = torch.as_tensor(graph_payload["global_rows"]).long().cpu()
    anchor_local, anchor_global = _accepted_anchor_rows(accepted, graph_global_rows)
    contract = adaptive_typed_context_v4_contract()
    prepared = contract.prepare_graph(
        support, torch.as_tensor(graph_payload["xyz"]).float().cpu()
    )
    selections, memory_audit = source.materialize_adaptive_selections(
        prepared,
        anchor_local,
        accepted["scale_indices"],
        batch_size=candidate_batch_size,
    )
    pooled = source._pool_adaptive_rows(
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
    payload = assemble_target_adaptive_authority_payload(
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
    return {
        "status": "materialized",
        "schema": TARGET_ADAPTIVE_TYPED_CONTEXT_SCHEMA,
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "regions": int(payload["canonical_region_indices"].numel()),
        "typed_context_valid": int(payload["typed_context_valid"].sum()),
        "output": file_record(output),
        "access_audit": target_adaptive_access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accepted-v2-authority", required=True)
    parser.add_argument("--expected-target-accepted-v2-authority-sha256", required=True)
    parser.add_argument("--factorized-field-checkpoint", required=True)
    parser.add_argument("--expected-factorized-field-checkpoint-sha256", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--expected-factorized-primitive-state-sha256", required=True)
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--field-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
