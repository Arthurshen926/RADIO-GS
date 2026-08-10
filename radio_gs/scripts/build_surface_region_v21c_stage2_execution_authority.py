#!/usr/bin/env python3
"""Build V2.1C Stage-II authority only from a positive pair-conflict audit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces import surface_region_v21c_stage2_pair_trigger as trigger
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as frozen_stage_i,
)
from radio_gs.scripts import (
    train_surface_region_v21c_stage2_pair_constrained_adamw as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


BUILD_SPEC_SCHEMA = (
    "radio_gs.surface_region_v21c_stage2_pair_constrained_"
    "execution_build_spec.v1"
)


def _record(value: object, *, label: str) -> dict[str, str]:
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def validate_build_spec(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "parent_v21b_execution_authority",
        "stage_i_execution_authority",
        "stage_i_audit_result",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C Stage-II build spec fields differ")
    spec = dict(value)
    if spec["schema"] != BUILD_SPEC_SCHEMA or spec["schema_version"] != 1:
        raise ValueError("V2.1C Stage-II build spec header differs")
    for name in (
        "parent_v21b_execution_authority",
        "stage_i_execution_authority",
        "stage_i_audit_result",
    ):
        spec[name] = _record(spec[name], label=f"V2.1C Stage-II {name}")
    return spec


def _load_audit(record: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, _sha, _path = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="V2.1C Stage-I audit result",
    )
    audit = frozen_stage_i.validate_stage_i_audit_result(raw)
    evidence = trigger.require_authorized(audit)
    return audit, evidence


def build(spec: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_build_spec(spec)
    audit, evidence = _load_audit(frozen["stage_i_audit_result"])
    if (
        audit["execution_authority"] != frozen["stage_i_execution_authority"]
        or audit["parent_v21b_execution_authority"]
        != frozen["parent_v21b_execution_authority"]
    ):
        raise ValueError("V2.1C Stage-II build lineage differs")
    code = {
        name: file_record(path)
        for name, path in trainer._expected_code_paths().items()
    }
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_stage_ii_after_pair_conflict_majority",
        **code,
        "parent_v21b_execution_authority": dict(
            frozen["parent_v21b_execution_authority"]
        ),
        "stage_i_execution_authority": dict(
            frozen["stage_i_execution_authority"]
        ),
        "stage_i_audit_result": dict(frozen["stage_i_audit_result"]),
        "pair_trigger_evidence": evidence,
        "training_authorized": True,
        "projection_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    authority = trainer.validate_execution_authority(authority)
    for name in (
        *trainer._CODE_RECORD_FIELDS,
        "parent_v21b_execution_authority",
        "stage_i_execution_authority",
        "stage_i_audit_result",
    ):
        validate_file_record(authority[name], label=f"V2.1C Stage-II {name}")
    return authority


def write_authority(spec: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"V2.1C Stage-II builder refuses to clobber: {destination}"
        )
    return write_frozen_json(destination, build(spec))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--expected-input-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw, _sha, _path = load_json_object(
        args.input_spec,
        expected_sha256=args.expected_input_spec_sha256,
        label="V2.1C Stage-II build spec",
    )
    written = write_authority(raw, args.output)
    print(json.dumps({"status": "V2.1C Stage-II authority built", "output": file_record(written)}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["BUILD_SPEC_SCHEMA", "build", "build_parser", "validate_build_spec", "write_authority"]
