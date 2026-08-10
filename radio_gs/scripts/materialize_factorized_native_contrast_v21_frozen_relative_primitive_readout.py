#!/usr/bin/env python3
"""Materialize a no-graph primitive union from frozen-relative region unary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_primitive_readout as formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as relative_formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_target_descriptor as target_formal,
)
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_frozen_relative_readout
    as relative_script,
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
EXECUTION_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_"
    "frozen_relative_primitive_execution.v1"
)
EXECUTION_STATUS = "authorized_query_opaque_selected_scale_unary_union_only"
DEPENDENCIES = {
    "primitive_readout_formal": Path(formal.__file__).resolve(),
    "relative_readout_formal": Path(relative_formal.__file__).resolve(),
    "relative_readout_materializer": Path(relative_script.__file__).resolve(),
    "target_descriptor_formal": Path(target_formal.__file__).resolve(),
    "target_accepted_v2_formal": Path(
        validate_target_accepted_v2_authority.__code__.co_filename
    ).resolve(),
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
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if str(validate_file_record(record, label=label)) != record["path"]:
        raise ValueError(f"{label} record path differs")
    return record


def _load_inputs(relative_record: Mapping[str, str]) -> dict[str, Any]:
    raw, digest, source = load_torch_mapping(
        relative_record["path"],
        expected_sha256=relative_record["sha256"],
        map_location="cpu",
        label="frozen-relative region readout",
    )
    if {"path": str(source), "sha256": digest} != dict(relative_record):
        raise ValueError("frozen-relative region readout record differs")
    relative = relative_formal.validate_readout_authority(raw)
    execution_record = _record(
        relative["execution_authority"], label="relative execution authority"
    )
    relative_execution = relative_script.validate_authority(
        execution_record["path"],
        expected_sha256=execution_record["sha256"],
        expected_output=relative_record["path"],
    )
    inputs = relative["input_authority"]
    accepted_record = _record(inputs["target_accepted_v2"], label="target AcceptedV2")
    descriptor_record = _record(
        inputs["target_descriptor"], label="contrast target descriptor"
    )
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        accepted_record["path"],
        expected_sha256=accepted_record["sha256"],
        map_location="cpu",
        label="primitive union target AcceptedV2",
    )
    descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="primitive union target descriptor",
    )
    accepted = validate_target_accepted_v2_authority(accepted_raw)
    descriptor = target_formal.validate_target_descriptor_authority(descriptor_raw)
    if (
        accepted_record != {"path": str(accepted_path), "sha256": accepted_sha}
        or descriptor_record
        != {"path": str(descriptor_path), "sha256": descriptor_sha}
        or descriptor["input_authority"]["target_accepted_v2"] != accepted_record
    ):
        raise ValueError("primitive union descriptor/AcceptedV2 lineage differs")
    state_record = _record(
        descriptor["input_authority"]["factorized_primitive_state"],
        label="factorized primitive state",
    )
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    if (
        relative["scene_id"] != accepted["scene_id"]
        or relative["physical_space_id"] != accepted["physical_space_id"]
        or relative["region_fingerprints_sha256"]
        != accepted["channel_sha256"]["region_fingerprints"]
        or not torch.equal(
            relative["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
        or not torch.equal(relative["scale_indices"], accepted["scale_indices"])
        or state.valid.shape != accepted["accepted_base_valid"].shape
        or state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
        or accepted["input_authority"]["geometry_authority"][
            "factorized_primitive_state_file_sha256"
        ]
        != state_record["sha256"]
    ):
        raise ValueError("primitive union relative/region/state axes differ")
    return {
        "relative": relative,
        "relative_execution": relative_execution,
        "accepted": accepted,
        "descriptor": descriptor,
        "state": state,
        "records": {
            "frozen_relative_readout": dict(relative_record),
            "frozen_relative_execution": execution_record,
            "target_accepted_v2": accepted_record,
            "target_descriptor": descriptor_record,
            "factorized_primitive_state": state_record,
        },
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.output_authority, label="primitive union authority")
    output = _new(args.output, label="primitive union output")
    relative_record = _record(
        {
            "path": str(Path(args.frozen_relative_readout).expanduser().resolve()),
            "sha256": args.expected_frozen_relative_readout_sha256,
        },
        label="frozen-relative readout",
    )
    inputs = _load_inputs(relative_record)
    authority = {
        "schema": EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": EXECUTION_STATUS,
        "scene_id": inputs["relative"]["scene_id"],
        "physical_space_id": inputs["relative"]["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": {
            name: file_record(path) for name, path in sorted(DEPENDENCIES.items())
        },
        "readout_contract": formal.readout_contract(),
        "readout_contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "input_authority": inputs["records"],
        "configuration": {
            "score_threshold": formal.SCORE_THRESHOLD,
            "maximum_regions": formal.MAXIMUM_REGIONS,
            "candidate_chunk_rows": formal.CANDIDATE_CHUNK_ROWS,
            "graph_or_relation": "none",
            "selected_scale_only": True,
        },
        "output": str(output),
        "materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": formal.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {
        "status": "frozen_relative_primitive_execution_authority_built",
        "authority": file_record(authority_output),
        "output": str(output),
    }


def validate_authority(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_output: str | Path | None = None,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="frozen-relative primitive execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "implementation_dependencies",
        "readout_contract",
        "readout_contract_sha256",
        "input_authority",
        "configuration",
        "output",
        "materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    expected_configuration = {
        "score_threshold": formal.SCORE_THRESHOLD,
        "maximum_regions": formal.MAXIMUM_REGIONS,
        "candidate_chunk_rows": formal.CANDIDATE_CHUNK_ROWS,
        "graph_or_relation": "none",
        "selected_scale_only": True,
    }
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("readout_contract") != formal.readout_contract()
        or authority.get("readout_contract_sha256")
        != formal.READOUT_CONTRACT_SHA256
        or authority.get("configuration") != expected_configuration
        or authority.get("materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != formal.access_audit()
        or _record(authority["implementation"], label="primitive implementation")
        != file_record(IMPLEMENTATION)
    ):
        raise ValueError("frozen-relative primitive execution authority differs")
    dependencies = authority.get("implementation_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("primitive execution dependencies differ")
    for name, expected in DEPENDENCIES.items():
        if _record(dependencies[name], label=f"primitive {name}") != file_record(expected):
            raise ValueError(f"primitive execution dependency differs: {name}")
    records = authority.get("input_authority")
    expected_names = {
        "frozen_relative_readout",
        "frozen_relative_execution",
        "target_accepted_v2",
        "target_descriptor",
        "factorized_primitive_state",
    }
    if not isinstance(records, Mapping) or set(records) != expected_names:
        raise ValueError("primitive execution inputs differ")
    records = {
        name: _record(records[name], label=f"primitive input {name}")
        for name in sorted(expected_names)
    }
    inputs = _load_inputs(records["frozen_relative_readout"])
    if records != inputs["records"]:
        raise ValueError("primitive execution nested inputs differ")
    output_raw = str(authority["output"])
    output = str(Path(output_raw).expanduser().resolve())
    if output_raw != output or (
        expected_output is not None
        and output != str(Path(expected_output).expanduser().resolve())
    ):
        raise ValueError("primitive execution output differs")
    if (
        authority["scene_id"] != inputs["relative"]["scene_id"]
        or authority["physical_space_id"] != inputs["relative"]["physical_space_id"]
    ):
        raise ValueError("primitive execution target identity differs")
    authority.update(
        {
            "input_authority": records,
            "verified_inputs": inputs,
            "output": output,
            "verified_record": {"path": str(source), "sha256": digest},
        }
    )
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="primitive union output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    inputs = execution["verified_inputs"]
    relative = inputs["relative"]
    accepted = inputs["accepted"]
    state = inputs["state"]
    readout = formal.frozen_relative_primitive_readout(
        relative_relevance=relative["relative_relevance"],
        selected_scale_eligibility=relative["selected_scale_eligibility"],
        unary_candidate_mask=relative["unary_candidate_mask"],
        region_rows=accepted["region_rows"],
        token_mask=accepted["token_mask"],
        primitive_valid=state.valid,
    )
    maximum = max(len(value) for value in readout.selected_region_indices)
    selected_total = sum(len(value) for value in readout.selected_region_indices)
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": relative["scene_id"],
        "physical_space_id": relative["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": dict(execution["input_authority"]),
        "region_fingerprints_sha256": relative["region_fingerprints_sha256"],
        "query_axis_count": relative["query_axis_count"],
        "canonical_region_indices": relative["canonical_region_indices"].clone(),
        "selected_scale_indices": relative["selected_scale_indices"].clone(),
        "selected_scale_eligibility": relative[
            "selected_scale_eligibility"
        ].clone(),
        "relative_relevance": relative["relative_relevance"].clone(),
        "unary_candidate_mask": relative["unary_candidate_mask"].clone(),
        "candidate_probability": readout.candidate_probability,
        "region_rows": accepted["region_rows"].clone(),
        "token_mask": accepted["token_mask"].clone(),
        "primitive_valid": readout.primitive_valid,
        "primitive_membership": readout.primitive_membership,
        "selected_region_indices": readout.selected_region_indices,
        "selected_region_scores": readout.selected_region_scores,
        "selected_marginal_core_rows": readout.selected_marginal_core_rows,
        "audit": {
            "opaque_query_axes": relative["query_axis_count"],
            "query_gate_passed": int(relative["unary_candidate_mask"].any(dim=0).sum()),
            "maximum_union_regions": maximum,
            "selected_region_total": selected_total,
            "selected_cross_scale_regions": 0,
            "selected_non_candidate_regions": 0,
            "primitive_memberships": int(readout.primitive_membership.sum()),
            "invalid_primitive_memberships_removed": (
                readout.invalid_primitive_memberships_removed
            ),
            "graph_or_relation_applied": False,
            "query_identifiers_consumed": False,
            "target_metric_computed": False,
        },
        "channel_sha256": {},
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    payload = formal.validate_readout_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "frozen_relative_primitive_union_complete",
        "audit": payload["audit"],
        "output": file_record(written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-authority")
    build.add_argument("--frozen-relative-readout", required=True)
    build.add_argument("--expected-frozen-relative-readout-sha256", required=True)
    build.add_argument("--output-authority", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=build_authority)
    run = subparsers.add_parser("materialize")
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
    "DEPENDENCIES",
    "EXECUTION_SCHEMA",
    "EXECUTION_STATUS",
    "IMPLEMENTATION",
    "build_authority",
    "build_parser",
    "materialize",
    "validate_authority",
]
