#!/usr/bin/env python3
"""Build fail-closed V2.1C Stage-I or triggered Stage-II authority."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


BUILD_SPEC_SCHEMA = (
    "radio_gs.surface_region_v21c_two_stage_execution_build_spec.v1"
)
SCHEMA_VERSION = 1


def validate_build_spec(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "stage",
        "parent_v21b_execution_authority",
        "stage_i_audit_result",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("V2.1C execution build spec fields differ")
    spec = dict(value)
    if (
        spec["schema"] != BUILD_SPEC_SCHEMA
        or spec["schema_version"] != SCHEMA_VERSION
        or spec["stage"] not in (trainer.STAGE_I, trainer.STAGE_II)
    ):
        raise ValueError("V2.1C execution build spec header differs")
    validate_file_record(
        spec["parent_v21b_execution_authority"],
        label="V2.1C parent V2.1B execution authority",
    )
    if spec["stage"] == trainer.STAGE_I:
        if spec["stage_i_audit_result"] is not None:
            raise ValueError("V2.1C Stage-I build spec cannot bind an audit")
    else:
        validate_file_record(
            spec["stage_i_audit_result"], label="V2.1C Stage-I audit result"
        )
    return spec


def _load_positive_audit(record: Mapping[str, str]) -> dict[str, Any]:
    raw, _digest, _path = load_json_object(
        validate_file_record(record, label="V2.1C Stage-I audit result"),
        expected_sha256=record["sha256"],
        label="V2.1C Stage-I audit result",
    )
    audit = trainer.validate_stage_i_audit_result(raw)
    if audit["trigger"]["stage_ii_authorized"] is not True:
        raise ValueError(
            "V2.1C Stage-II builder refuses a non-triggering Stage-I audit"
        )
    return audit


def build(spec: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_build_spec(spec)
    stage = frozen["stage"]
    if stage == trainer.STAGE_II:
        audit = _load_positive_audit(frozen["stage_i_audit_result"])
        if audit["parent_v21b_execution_authority"] != frozen[
            "parent_v21b_execution_authority"
        ]:
            raise ValueError("V2.1C Stage-I audit binds another V2.1B parent")
    code = {
        name: file_record(path)
        for name, path in trainer._expected_code_paths().items()
    }
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": (
            "authorized_source_only_stage_i_audit"
            if stage == trainer.STAGE_I
            else "authorized_source_only_stage_ii_after_positive_audit"
        ),
        "stage": stage,
        **code,
        "parent_v21b_execution_authority": dict(
            frozen["parent_v21b_execution_authority"]
        ),
        "stage_i_audit_result": (
            None
            if stage == trainer.STAGE_I
            else dict(frozen["stage_i_audit_result"])
        ),
        "training_authorized": True,
        "projection_authorized": stage == trainer.STAGE_II,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    authority = trainer.validate_execution_authority(authority)
    for name in (*trainer._CODE_RECORD_FIELDS, "parent_v21b_execution_authority"):
        validate_file_record(authority[name], label=f"V2.1C authority {name}")
    if stage == trainer.STAGE_II:
        validate_file_record(
            authority["stage_i_audit_result"],
            label="V2.1C authority Stage-I result",
        )
    return authority


def write_authority(spec: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"V2.1C authority builder refuses to clobber: {destination}"
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
    raw, _digest, _path = load_json_object(
        args.input_spec,
        expected_sha256=args.expected_input_spec_sha256,
        label="V2.1C execution build spec",
    )
    written = write_authority(raw, args.output)
    print(
        json.dumps(
            {
                "status": "V2.1C source execution authority built",
                "output": file_record(written),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BUILD_SPEC_SCHEMA",
    "SCHEMA_VERSION",
    "build",
    "build_parser",
    "validate_build_spec",
    "write_authority",
]
