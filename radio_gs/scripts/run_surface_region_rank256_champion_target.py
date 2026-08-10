#!/usr/bin/env python3
"""Frozen source-gated rank-256 V2.1B/V2.1C target materialization."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.interfaces.surface_region_target_adaptive_typed_context import (
    validate_target_adaptive_typed_context_authority,
)
from radio_gs.interfaces import surface_region_rank256_champion as formal
from radio_gs.scripts import materialize_surface_region_v21_target_descriptor as v21_target
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as routing,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
)
TARGET_STATUS = "authorized_after_rank256_source_pass_for_query_free_target"


def _existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be an absolute canonical path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _source(args: argparse.Namespace) -> dict[str, Any]:
    return formal.validate_champion_source(
        args.source_variant,
        args.source_result,
        expected_sha256=args.expected_source_result_sha256,
    )


def build_target_authority(args: argparse.Namespace) -> dict[str, Any]:
    gate = _source(args)  # Must remain the first artifact access.
    authority_output = _new(args.output_authority, label="target authority output")
    descriptor_output = _new(
        args.target_descriptor_output, label="target descriptor output"
    )
    accepted = _existing(args.target_accepted_v2, label="target AcceptedV2")
    adaptive = _existing(
        args.target_adaptive_typed_context, label="target adaptive context"
    )
    state = _existing(
        args.factorized_primitive_state, label="factorized primitive state"
    )
    physical = formal.physical_authority(
        args.dataset_id, args.scene_id, args.geometry_checkpoint_sha256
    )
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": TARGET_STATUS,
        "source_variant": args.source_variant,
        "source_result": dict(gate["source_result"]),
        "scene_id": physical["scene_id"],
        "physical_space_id": physical["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "preregistration": file_record(PREREGISTRATION),
        "target_inputs": {
            "target_accepted_v2": file_record(accepted),
            "target_adaptive_typed_context": file_record(adaptive),
            "factorized_primitive_state": file_record(state),
            "champion_checkpoint": dict(gate["checkpoint"]),
            "champion_normalization": dict(gate["normalization_authority"]),
        },
        "target_descriptor_output": str(descriptor_output),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {
        "status": "rank256_target_authority_built",
        "authority": file_record(written),
    }


def validate_target_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="rank-256 target authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "source_variant",
        "source_result",
        "scene_id",
        "physical_space_id",
        "implementation",
        "preregistration",
        "target_inputs",
        "target_descriptor_output",
        "materialization_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if required != set(authority):
        raise ValueError("rank-256 target authority fields differ")
    source_record = formal._record(
        authority["source_result"], label="target source result"
    )
    gate = formal.validate_champion_source(
        authority["source_variant"],
        source_record["path"],
        expected_sha256=source_record["sha256"],
    )
    if (
        authority["schema"] != formal.TARGET_EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != TARGET_STATUS
        or authority["source_variant"] not in formal.SOURCE_VARIANTS
        or authority["materialization_authorized"] is not True
        or authority["query_execution_authorized"] is not False
        or authority["metric_execution_authorized"] is not False
        or authority["access_audit"] != formal.target_access_audit()
        or source_record != gate["source_result"]
    ):
        raise ValueError("rank-256 target authority header differs")
    if (
        validate_file_record(
            authority["implementation"], label="target implementation"
        )
        != IMPLEMENTATION
        or validate_file_record(
            authority["preregistration"], label="target preregistration"
        )
        != PREREGISTRATION
    ):
        raise ValueError("rank-256 target implementation/preregistration differs")
    inputs = authority["target_inputs"]
    names = {
        "target_accepted_v2",
        "target_adaptive_typed_context",
        "factorized_primitive_state",
        "champion_checkpoint",
        "champion_normalization",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("rank-256 target inputs differ")
    authority["target_inputs"] = {
        name: formal._record(inputs[name], label=f"target {name}")
        for name in sorted(names)
    }
    if (
        authority["target_inputs"]["champion_checkpoint"] != gate["checkpoint"]
        or authority["target_inputs"]["champion_normalization"]
        != gate["normalization_authority"]
    ):
        raise ValueError("rank-256 target source model binding differs")
    output = formal._output(
        authority["target_descriptor_output"], label="target descriptor output"
    )
    if expected_output is not None and output != str(
        Path(expected_output).expanduser().resolve()
    ):
        raise ValueError("rank-256 target output differs")
    authority.update(
        {
            "source_result": source_record,
            "target_descriptor_output": output,
            "verified_source_gate": gate,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def _target_inputs(execution: Mapping[str, Any]) -> dict[str, Any]:
    records = execution["target_inputs"]
    accepted_raw, _, _ = load_torch_mapping(
        records["target_accepted_v2"]["path"],
        expected_sha256=records["target_accepted_v2"]["sha256"],
        map_location="cpu",
        label="rank-256 target AcceptedV2",
    )
    adaptive_raw, _, _ = load_torch_mapping(
        records["target_adaptive_typed_context"]["path"],
        expected_sha256=records["target_adaptive_typed_context"]["sha256"],
        map_location="cpu",
        label="rank-256 target adaptive context",
    )
    return {
        "records": records,
        "accepted": validate_target_accepted_v2_authority(accepted_raw),
        "adaptive": validate_target_adaptive_typed_context_authority(adaptive_raw),
        "state": load_factorized_primitive_state(
            records["factorized_primitive_state"]["path"],
            expected_sha256=records["factorized_primitive_state"]["sha256"],
        ),
    }


def materialize_target(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="rank-256 target descriptor")
    execution = validate_target_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = _target_inputs(execution)
    v21_target._validate_alignment(inputs)
    accepted = inputs["accepted"]
    adaptive = inputs["adaptive"]
    state = inputs["state"]
    summary = aggregate_surface_region_full_scalars(
        state,
        accepted["accepted_base_valid"],
        accepted["region_rows"],
        accepted["token_mask"],
        accepted["anchor_index"],
    )
    model, normalization, _checkpoint = formal.load_champion_model(
        execution["verified_source_gate"], execution["source_variant"]
    )
    scene = {
        "raw_full_scalar_summary": summary.summary,
        "typed_context_statistics": adaptive["typed_context_statistics"],
        "eligible": summary.use_full_scalar_mask,
        "typed_context_valid": adaptive["typed_context_valid"],
    }
    declared, effective_ood, active = routing._pilot_routing(scene, normalization)
    regions = int(accepted["accepted_v2_e0"].shape[0])
    descriptor = torch.empty_like(accepted["accepted_v2_e0"].float().cpu())
    reliability = torch.empty(regions, dtype=torch.float32)
    budget = torch.empty(regions, dtype=torch.float32)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("rank-256 target batch size must be positive")
    with torch.inference_mode():
        for start in range(0, regions, batch_size):
            stop = min(start + batch_size, regions)
            diagnostics = model.forward_with_diagnostics(
                accepted["accepted_v2_e0"][start:stop].float().cpu(),
                adaptive["pooled_context_radio_direction"][start:stop]
                .float()
                .cpu(),
                summary.summary[start:stop].float().cpu(),
                adaptive["typed_context_statistics"][start:stop].float().cpu(),
                active_mask=declared[start:stop],
                ood_mask=effective_ood[start:stop],
            )
            descriptor[start:stop] = diagnostics.semantic_descriptor
            reliability[start:stop] = diagnostics.reliability_score
            budget[start:stop] = diagnostics.angular_budget_radians
    fallback = ~active
    base = accepted["accepted_v2_e0"].float().cpu()
    fallback_equal = torch.equal(descriptor[fallback], base[fallback])
    if not fallback_equal:
        raise RuntimeError("rank-256 target immutable fallback changed")
    masks = {
        "full_scalar_eligible_mask": summary.use_full_scalar_mask.bool()
        .cpu()
        .contiguous(),
        "typed_context_valid_mask": adaptive["typed_context_valid"]
        .bool()
        .cpu()
        .contiguous(),
        "normalization_ood_mask": (
            effective_ood & summary.use_full_scalar_mask
        )
        .bool()
        .cpu()
        .contiguous(),
        "effective_ood_mask": effective_ood.bool().cpu().contiguous(),
        "active_update_mask": active.bool().cpu().contiguous(),
        "immutable_fallback_mask": fallback.bool().cpu().contiguous(),
        "descriptor_changed_mask": (descriptor != base).any(dim=-1).contiguous(),
    }
    payload = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_contract(),
        "contract_sha256": formal.TARGET_CONTRACT_SHA256,
        "source_variant": execution["source_variant"],
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": file_record(IMPLEMENTATION),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(execution["target_inputs"]),
        "region_row_ids": list(adaptive["region_row_ids"]),
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": descriptor.float().cpu().contiguous(),
        "reliability_score": reliability.contiguous(),
        "angular_budget_radians": budget.contiguous(),
        **masks,
        "fallback_bitwise_equal": fallback_equal,
        "routing_audit": {
            "regions": regions,
            "active_update": int(active.sum()),
            "immutable_fallback": int(fallback.sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": formal.target_access_audit(),
    }
    payload["channel_sha256"] = formal.target_channel_sha256(payload)
    payload = formal.validate_target_descriptor(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "rank256_target_descriptor_complete",
        "output": file_record(output),
        "routing_audit": payload["routing_audit"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-authority")
    build.add_argument("--source-variant", choices=formal.SOURCE_VARIANTS, required=True)
    build.add_argument("--source-result", required=True)
    build.add_argument("--expected-source-result-sha256", required=True)
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--geometry-checkpoint-sha256", required=True)
    build.add_argument("--target-accepted-v2", required=True)
    build.add_argument("--target-adaptive-typed-context", required=True)
    build.add_argument("--factorized-primitive-state", required=True)
    build.add_argument("--target-descriptor-output", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(func=build_target_authority)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--execution-authority", required=True)
    materialize.add_argument("--expected-execution-authority-sha256", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--batch-size", type=int, default=256)
    materialize.set_defaults(func=materialize_target)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.func(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_target_authority",
    "materialize_target",
    "validate_target_authority",
]
