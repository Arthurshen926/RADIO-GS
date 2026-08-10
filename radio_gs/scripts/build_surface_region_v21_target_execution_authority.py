#!/usr/bin/env python3
"""Build a source-gated V2.1 target-descriptor execution authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.interfaces.surface_region_v21_source_gate import (
    validate_source_pilot_chain,
)
from radio_gs.interfaces.surface_region_v21_target import (
    TARGET_EXECUTION_SCHEMA,
    TARGET_IMPLEMENTATION_DEPENDENCIES,
    TARGET_IMPLEMENTATION_PATH,
    TARGET_PREREGISTRATION_PATH,
    target_descriptor_access_audit,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_existing(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an existing canonical regular file")
    return path


def _canonical_new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def build(args: argparse.Namespace) -> dict[str, Any]:
    # Source PASS is deliberately the first filesystem action after argument
    # parsing.  Output preflight and every target stat/hash happen afterwards.
    source_gate = validate_source_pilot_chain(
        args.source_pilot_result,
        expected_sha256=args.expected_source_pilot_result_sha256,
        require_promotion=True,
    )
    if source_gate.get("source_promotion_authorized") is not True:
        raise ValueError("V2.1 source promotion is not authorized")

    authority_output = _canonical_new(
        args.output_authority, label="target execution authority output"
    )
    descriptor_output = _canonical_new(
        args.target_descriptor_output, label="target descriptor output"
    )
    accepted = _canonical_existing(
        args.target_accepted_v2, label="target AcceptedV2 authority"
    )
    adaptive = _canonical_existing(
        args.target_adaptive_typed_context,
        label="target adaptive typed-context authority",
    )
    state = _canonical_existing(
        args.factorized_primitive_state, label="factorized primitive state"
    )
    geometry_sha = str(args.geometry_checkpoint_sha256)
    if _SHA256.fullmatch(geometry_sha) is None:
        raise ValueError("geometry checkpoint SHA-256 differs")
    physical = target_physical_space_authority(
        dataset_id=args.dataset_id,
        scene_id=args.scene_id,
        geometry_checkpoint_sha256=geometry_sha,
    )
    authority = {
        "schema": TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_v21_source_promotion_for_query_free_descriptor_only",
        "scene_id": physical["scene_id"],
        "physical_space_id": physical["physical_space_id"],
        "source_pilot_result": dict(source_gate["source_result"]),
        "implementation": file_record(TARGET_IMPLEMENTATION_PATH),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in TARGET_IMPLEMENTATION_DEPENDENCIES.items()
        },
        "preregistration": file_record(TARGET_PREREGISTRATION_PATH),
        "target_inputs": {
            "target_accepted_v2": file_record(accepted),
            "target_adaptive_typed_context": file_record(adaptive),
            "factorized_primitive_state": file_record(state),
            "v21_checkpoint": dict(source_gate["checkpoint"]),
            "v21_normalization": dict(source_gate["normalization_authority"]),
        },
        "target_descriptor_output": str(descriptor_output),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": target_descriptor_access_audit(),
    }
    written = write_frozen_json(authority_output, authority)
    return {
        "status": "v21_target_execution_authority_built_after_source_pass",
        "authority": file_record(written),
        "target_descriptor_output": str(descriptor_output),
        "scene_id": physical["scene_id"],
        "physical_space_id": physical["physical_space_id"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pilot-result", required=True)
    parser.add_argument("--expected-source-pilot-result-sha256", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--geometry-checkpoint-sha256", required=True)
    parser.add_argument("--target-accepted-v2", required=True)
    parser.add_argument("--target-adaptive-typed-context", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--target-descriptor-output", required=True)
    parser.add_argument("--output-authority", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
