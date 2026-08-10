#!/usr/bin/env python3
"""Materialize an independent primitive-domain frozen-relative unary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_contrast_v21_primitive_domain_frozen_relative as formal
from radio_gs.interfaces.factorized_primitive_state import load_factorized_primitive_state
from radio_gs.scripts import materialize_factorized_native_contrast_v21_frozen_relative_readout as region_script
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
TEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests/test_factorized_native_contrast_v21_primitive_domain_frozen_relative.py"
)
EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_primitive_domain_"
    "frozen_relative_execution.v1"
)
EXECUTION_STATUS = "authorized_independent_query_opaque_primitive_domain_unary"
DEPENDENCIES = {
    "primitive_domain_formal": Path(formal.__file__).resolve(),
    "primitive_domain_tests": TEST_PATH,
    "strict_region_input_loader": Path(region_script.__file__).resolve(),
    "factorized_primitive_state_loader": Path(load_factorized_primitive_state.__code__.co_filename).resolve(),
    "frozen_protocol": Path(formal.region_formal.FROZEN_PROTOCOL_RECORD["path"]).resolve(),
}


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical absolute path")
    return path


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if str(validate_file_record(result, label=label)) != result["path"]:
        raise ValueError(f"{label} path differs")
    return result


def _load_inputs(
    *, exact_record: Mapping[str, str], renderer_record: Mapping[str, str]
) -> dict[str, Any]:
    inputs = region_script._load_inputs(
        exact_relevance_record=exact_record,
        renderer_geometry_record=renderer_record,
    )
    descriptor = inputs["descriptor"]
    accepted = inputs["accepted"]
    state_record = _record(
        descriptor["input_authority"]["factorized_primitive_state"],
        label="primitive-domain factorized primitive state",
    )
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    if (
        state.xyz.shape != inputs["accepted"]["accepted_base_valid"].shape + (3,)
        or not torch.equal(state.valid, accepted["accepted_base_valid"])
        or state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
        or accepted["input_authority"]["geometry_authority"][
            "factorized_primitive_state_file_sha256"
        ] != state_record["sha256"]
    ):
        raise ValueError("primitive-domain state/AcceptedV2 geometry differs")
    return {
        **inputs,
        "state": state,
        "state_record": state_record,
        "records": {
            "exact_relevance": dict(inputs["relevance_record"]),
            "query_execution": dict(inputs["query_execution_record"]),
            "target_descriptor": dict(inputs["descriptor_record"]),
            "target_accepted_v2": dict(inputs["accepted_record"]),
            "renderer_geometry_checkpoint": dict(inputs["renderer_record"]),
            "factorized_primitive_state": state_record,
        },
    }


def _dependency_records() -> dict[str, dict[str, str]]:
    return {name: file_record(path) for name, path in sorted(DEPENDENCIES.items())}


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.output_authority, label="primitive-domain authority")
    output = _new(args.output, label="primitive-domain readout output")
    exact_record = _record(
        {
            "path": str(Path(args.exact_relevance).expanduser().resolve()),
            "sha256": args.expected_exact_relevance_sha256,
        }, label="primitive-domain exact relevance",
    )
    renderer_record = _record(
        {
            "path": str(Path(args.renderer_geometry_checkpoint).expanduser().resolve()),
            "sha256": args.expected_renderer_geometry_checkpoint_sha256,
        }, label="primitive-domain renderer geometry",
    )
    inputs = _load_inputs(exact_record=exact_record, renderer_record=renderer_record)
    authority = {
        "schema": EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": EXECUTION_STATUS,
        "scene_id": inputs["relevance"]["scene_id"],
        "physical_space_id": inputs["relevance"]["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": _dependency_records(),
        "readout_contract": formal.readout_contract(),
        "readout_contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "projection_rule_audit": formal.projection_rule_audit(),
        "input_authority": inputs["records"],
        "configuration": {
            "projection_rule": formal.PROJECTION_RULE,
            "semantic_levels": formal.SEMANTIC_LEVELS,
            "knn_neighbors": formal.KNN_NEIGHBORS,
            "knn_chunk_size": formal.KNN_CHUNK_SIZE,
            "mask_threshold": formal.MASK_THRESHOLD,
            "covered_valid_primitive_domain_only": True,
            "graph_or_relation": "none",
        },
        "output": str(output),
        "materialization_authorized": True,
        "metric_execution_authorized": False,
        "existing_candidate_mutation_authorized": False,
        "access_audit": formal.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {
        "status": "primitive_domain_frozen_relative_authority_built",
        "authority": file_record(authority_output),
        "output": str(output),
    }


def validate_authority(
    path: str | Path, *, expected_sha256: str, expected_output: str | Path | None = None
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256,
        label="primitive-domain frozen-relative execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "physical_space_id",
        "implementation", "implementation_dependencies", "readout_contract",
        "readout_contract_sha256", "projection_rule_audit", "input_authority",
        "configuration", "output", "materialization_authorized",
        "metric_execution_authorized", "existing_candidate_mutation_authorized",
        "access_audit",
    }
    authority = dict(raw)
    expected_configuration = {
        "projection_rule": formal.PROJECTION_RULE,
        "semantic_levels": formal.SEMANTIC_LEVELS,
        "knn_neighbors": formal.KNN_NEIGHBORS,
        "knn_chunk_size": formal.KNN_CHUNK_SIZE,
        "mask_threshold": formal.MASK_THRESHOLD,
        "covered_valid_primitive_domain_only": True,
        "graph_or_relation": "none",
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("readout_contract") != formal.readout_contract()
        or authority.get("readout_contract_sha256") != formal.READOUT_CONTRACT_SHA256
        or authority.get("projection_rule_audit") != formal.projection_rule_audit()
        or authority.get("configuration") != expected_configuration
        or authority.get("materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("existing_candidate_mutation_authorized") is not False
        or authority.get("access_audit") != formal.access_audit()
        or _record(authority["implementation"], label="primitive-domain implementation")
        != file_record(IMPLEMENTATION)
    ):
        raise ValueError("primitive-domain execution authority differs")
    dependencies = authority["implementation_dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("primitive-domain dependency fields differ")
    dependencies = {
        name: _record(dependencies[name], label=f"primitive-domain dependency {name}")
        for name in sorted(DEPENDENCIES)
    }
    if any(dependencies[name] != file_record(path) for name, path in DEPENDENCIES.items()):
        raise ValueError("primitive-domain dependency records differ")
    records = authority["input_authority"]
    expected_names = {
        "exact_relevance", "query_execution", "target_descriptor",
        "target_accepted_v2", "renderer_geometry_checkpoint",
        "factorized_primitive_state",
    }
    if not isinstance(records, Mapping) or set(records) != expected_names:
        raise ValueError("primitive-domain input authority fields differ")
    records = {
        name: _record(records[name], label=f"primitive-domain input {name}")
        for name in sorted(expected_names)
    }
    inputs = _load_inputs(
        exact_record=records["exact_relevance"],
        renderer_record=records["renderer_geometry_checkpoint"],
    )
    if records != inputs["records"]:
        raise ValueError("primitive-domain nested input lineage differs")
    output_raw = str(authority["output"])
    output = str(Path(output_raw).expanduser().resolve())
    if output != output_raw or (
        expected_output is not None
        and output != str(Path(expected_output).expanduser().resolve())
    ):
        raise ValueError("primitive-domain output differs")
    if (
        authority["scene_id"] != inputs["relevance"]["scene_id"]
        or authority["physical_space_id"] != inputs["relevance"]["physical_space_id"]
    ):
        raise ValueError("primitive-domain target identity differs")
    authority.update(
        {
            "implementation_dependencies": dependencies,
            "input_authority": records,
            "verified_inputs": inputs,
            "output": output,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="primitive-domain readout output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = execution["verified_inputs"]
    relevance = inputs["relevance"]
    accepted = inputs["accepted"]
    state = inputs["state"]
    readout = formal.primitive_domain_frozen_relative_readout(
        region_raw_relevance=relevance["region_absolute_relevance"],
        scale_indices=accepted["scale_indices"],
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        primitive_xyz=state.xyz,
        primitive_valid=state.valid,
        chunk_size=execution["configuration"]["knn_chunk_size"],
    )
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": relevance["scene_id"],
        "physical_space_id": relevance["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(execution["input_authority"]),
        "query_axis_count": int(relevance["region_absolute_relevance"].shape[1]),
        "region_raw_relevance": relevance["region_absolute_relevance"].clone(),
        "scale_indices": accepted["scale_indices"].clone(),
        "region_rows": accepted["region_rows"].clone(),
        "token_mask": accepted["token_mask"].clone(),
        "primitive_xyz": state.xyz.clone(),
        "primitive_valid": state.valid.clone(),
        "projection_coverage_count": readout.projection_coverage_count,
        "projection_coverage": readout.projection_coverage,
        "projected_raw_relevance": readout.projected_raw_relevance,
        "smoothed_relevance": readout.smoothed_relevance,
        "remapped_relevance": readout.remapped_relevance,
        "raw_smoothed_peaks": readout.raw_smoothed_peaks,
        "selected_scale_indices": readout.selected_scale_indices,
        "selected_scale_eligibility": readout.selected_scale_eligibility,
        "relative_relevance": readout.relative_relevance,
        "query_gate": readout.query_gate,
        "unary_candidate_mask": readout.unary_candidate_mask,
        "audit": formal.expected_audit(readout),
        "channel_sha256": {},
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    payload = formal.validate_readout_authority(payload, replay=readout)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "primitive_domain_frozen_relative_unary_complete",
        "candidate_status": "query_opaque_structural_audit_only_no_metric_authority",
        "scene_id": payload["scene_id"],
        "audit": payload["audit"],
        "output": file_record(written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--exact-relevance", required=True)
    build.add_argument("--expected-exact-relevance-sha256", required=True)
    build.add_argument("--renderer-geometry-checkpoint", required=True)
    build.add_argument("--expected-renderer-geometry-checkpoint-sha256", required=True)
    build.add_argument("--output-authority", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=build_authority)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    validate.add_argument("--expected-output")
    validate.set_defaults(handler=lambda args: {
        "status": "primitive_domain_frozen_relative_authority_valid",
        "authority": validate_authority(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
            expected_output=args.expected_output,
        )["verified_record"],
    })
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEPENDENCIES", "EXECUTION_SCHEMA", "EXECUTION_STATUS", "IMPLEMENTATION",
    "build_authority", "build_parser", "materialize", "validate_authority",
]
