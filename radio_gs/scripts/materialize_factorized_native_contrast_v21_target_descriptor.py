#!/usr/bin/env python3
"""Materialize the independent contrast V2.1 query-free target descriptor."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as formal
from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.materialize_factorized_native_target_descriptor import (
    apply_factorized_native_canonical_forward,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_target_inputs(execution: Mapping[str, Any]) -> dict[str, Any]:
    records = execution["target_inputs"]
    accepted_raw, _, _ = load_torch_mapping(
        records["target_accepted_v2"]["path"],
        expected_sha256=records["target_accepted_v2"]["sha256"],
        map_location="cpu",
        label="contrast V2.1 target AcceptedV2",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    state = load_factorized_primitive_state(
        records["factorized_primitive_state"]["path"],
        expected_sha256=records["factorized_primitive_state"]["sha256"],
    )
    geometry = accepted["input_authority"]["geometry_authority"]
    if (
        geometry["factorized_primitive_state_file_sha256"] != state.sha256
        or state.schema != FACTORIZED_PRIMITIVE_STATE_SCHEMA
        or state.schema_version != FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
        or state.metadata.get("geometry_fingerprint")
        != accepted["geometry_fingerprint"]
        or state.valid.shape != accepted["accepted_base_valid"].shape
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
        raise ValueError("contrast V2.1 target Accepted/state alignment differs")
    return {"records": records, "accepted": accepted, "state": state}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber target descriptor: {output}")
    execution = formal.validate_target_execution_authority(
        args.target_execution_authority,
        expected_sha256=args.expected_target_execution_authority_sha256,
        expected_output=output,
    )
    target = _load_target_inputs(execution)
    source = execution["verified_source_gate"]
    accepted, state = target["accepted"], target["state"]
    model = readout.build_model(DIRECTION_ONLY, source["normalization"])
    model.load_state_dict(source["checkpoint"]["model_state_dict"], strict=True)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        source["official_radio_checkpoint"]["path"],
        expected_sha256=source["official_radio_checkpoint"]["sha256"],
    )
    forward = apply_factorized_native_canonical_forward(
        accepted_v2_e0=accepted["accepted_v2_e0"],
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        anchor_index=accepted["anchor_index"],
        state=state,
        model=model,
        head=head,
        device=args.device,
        batch_size=int(args.batch_size),
    )
    region_ids = [
        shard.stable_region_id(accepted["scene_id"], fingerprint)
        for fingerprint in accepted["region_fingerprints"]
    ]
    masks = {
        name: forward[name]
        for name in (
            "exact_state_anchor_mask",
            "active_update_mask",
            "immutable_fallback_mask",
            "descriptor_changed_mask",
        )
    }
    result = source["result"]
    payload: dict[str, Any] = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_descriptor_contract(),
        "contract_sha256": formal.TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": accepted["scene_id"],
        "physical_space_id": accepted["physical_space_id"],
        "physical_space_authority": dict(accepted["physical_space_authority"]),
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_contrast_v21_result": dict(
                execution["source_contrast_v21_result"]
            ),
            "source_contrast_v21_checkpoint": dict(result["checkpoint"]),
            "source_normalization": dict(result["normalization"]),
            "source_contrast_reference": dict(result["contrast_reference"]),
            "official_radio_checkpoint": dict(
                source["official_radio_checkpoint"]
            ),
            **dict(target["records"]),
        },
        "source_arm": DIRECTION_ONLY,
        "source_selected_step": source["selected_step"],
        "source_gate_audit": {
            "result_schema": result["schema"],
            "checkpoint_schema": source["checkpoint"]["schema"],
            "schema_version": 21,
            "status": result["status"],
            "source_only_passed": source["source_only_passed"],
        },
        "region_row_ids": region_ids,
        "canonical_region_indices": accepted["canonical_region_indices"].clone(),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "semantic_descriptor": forward["semantic_descriptor"],
        **masks,
        "fallback_bitwise_equal": forward["fallback_bitwise_equal"],
        "routing_audit": {
            "regions": len(region_ids),
            "exact_state_anchor": int(masks["exact_state_anchor_mask"].sum()),
            "active_update": int(masks["active_update_mask"].sum()),
            "immutable_fallback": int(masks["immutable_fallback_mask"].sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": formal.target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    payload = formal.validate_target_descriptor_authority(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": (
            "materialized_query_free_factorized_native_contrast_v21_"
            "target_descriptor"
        ),
        "scene_id": payload["scene_id"],
        "source_arm": payload["source_arm"],
        "source_selected_step": payload["source_selected_step"],
        "regions": len(region_ids),
        "routing_audit": payload["routing_audit"],
        "output": file_record(output),
        "exact_query_descriptor_view_schema": (
            formal.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA
        ),
        "access_audit": formal.target_descriptor_access_audit(),
    }


def _bound_record(path: str, digest: str, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": str(digest)}
    validate_file_record(record, label=label)
    return record


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.authority_output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber target authority: {output}")
    source_record = _bound_record(
        args.source_contrast_v21_result,
        args.source_contrast_v21_result_sha256,
        label="contrast V2.1 source result",
    )
    # Complete source-only gate before binding/opening either target artifact.
    source = formal.validate_source_contrast_v21_result(source_record)
    target_inputs = {
        "target_accepted_v2": _bound_record(
            args.target_accepted_v2,
            args.target_accepted_v2_sha256,
            label="target AcceptedV2",
        ),
        "factorized_primitive_state": _bound_record(
            args.factorized_primitive_state,
            args.factorized_primitive_state_sha256,
            label="target factorized primitive state",
        ),
    }
    descriptor_output = str(Path(args.descriptor_output).expanduser().resolve())
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": (
            "authorized_after_contrast_v21_source_promotion_for_query_free_target"
        ),
        "source_contrast_v21_result": source_record,
        "implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.TARGET_IMPLEMENTATION_DEPENDENCIES.items()
        },
        "target_inputs": target_inputs,
        "target_descriptor_output": descriptor_output,
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_descriptor_access_audit(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "factorized_native_contrast_v21_target_authority_built",
        "source_selected_step": source["selected_step"],
        "authority": file_record(output),
        "descriptor_output": descriptor_output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--source-contrast-v21-result", required=True)
    build.add_argument("--source-contrast-v21-result-sha256", required=True)
    build.add_argument("--target-accepted-v2", required=True)
    build.add_argument("--target-accepted-v2-sha256", required=True)
    build.add_argument("--factorized-primitive-state", required=True)
    build.add_argument("--factorized-primitive-state-sha256", required=True)
    build.add_argument("--descriptor-output", required=True)
    build.add_argument("--authority-output", required=True)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--target-execution-authority", required=True)
    validate.add_argument("--expected-target-execution-authority-sha256", required=True)
    run = commands.add_parser("materialize")
    run.add_argument("--target-execution-authority", required=True)
    run.add_argument("--expected-target-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-authority":
        result = build_authority(args)
    elif args.command == "validate-authority":
        authority = formal.validate_target_execution_authority(
            args.target_execution_authority,
            expected_sha256=args.expected_target_execution_authority_sha256,
        )
        result = {
            "status": "factorized_native_contrast_v21_target_authority_validated",
            "source_selected_step": authority["verified_source_gate"]["selected_step"],
            "target_files_opened_after_source_gate": True,
        }
    else:
        result = materialize(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["build_authority", "build_parser", "materialize"]
