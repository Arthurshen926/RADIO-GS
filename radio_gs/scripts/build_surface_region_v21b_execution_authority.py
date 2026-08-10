#!/usr/bin/env python3
"""Build the no-clobber source-only V2.1B exact-4+2 execution authority."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


# Reuse the already frozen exact-4+2 asset specification byte-for-byte.  This
# builder changes only the execution/code authority, never the data cohort.
BUILD_SPEC_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_pilot_"
    "execution_build_spec.v1"
)
SCHEMA_VERSION = 1


def validate_build_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1B execution build spec must be a mapping")
    spec = dict(value)
    required = {
        "schema",
        "schema_version",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
        "compositional_banks",
        "typed_relation_authority",
        "source_train",
        "source_validation",
    }
    if (
        set(spec) != required
        or spec.get("schema") != BUILD_SPEC_SCHEMA
        or spec.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("V2.1B execution build spec fields differ")
    return spec


def _validate_all_file_records(authority: Mapping[str, Any]) -> None:
    for name in (
        *trainer._CODE_RECORD_FIELDS,
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    ):
        validate_file_record(authority[name], label=f"V2.1B execution {name}")
    for name in trainer.COMPONENT_WEIGHTS:
        record = authority["compositional_banks"][name]
        validate_file_record(
            {"path": record["path"], "sha256": record["sha256"]},
            label=f"V2.1B execution {name}",
        )
    relation = authority["typed_relation_authority"]
    validate_file_record(
        {"path": relation["path"], "sha256": relation["sha256"]},
        label="V2.1B execution typed relation authority",
    )
    for split in ("source_train", "source_validation"):
        for row in authority[split]:
            for name in (
                "training_shard",
                "adaptive_context",
                "hard_negative_authority",
            ):
                validate_file_record(
                    row[name],
                    label=f"V2.1B execution {split} {row['scene_id']} {name}",
                )


def build(spec: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_build_spec(spec)
    code = {
        name: file_record(path)
        for name, path in trainer._resolved_expected_code_paths().items()
    }
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_v21b_exact4train_2validation",
        **code,
        "cohort_authority": frozen["cohort_authority"],
        "pilot_cohort_region_view_registry": frozen[
            "pilot_cohort_region_view_registry"
        ],
        "benchmark_exclusion_manifest": frozen["benchmark_exclusion_manifest"],
        "fit_text_bank": frozen["fit_text_bank"],
        "canonical_negative_bank": frozen["canonical_negative_bank"],
        "compositional_banks": frozen["compositional_banks"],
        "typed_relation_authority": frozen["typed_relation_authority"],
        "source_train": frozen["source_train"],
        "source_validation": frozen["source_validation"],
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    authority = trainer.validate_execution_authority(authority)
    _validate_all_file_records(authority)
    return authority


def write_authority(spec: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "V2.1B execution-authority builder refuses to clobber: "
            f"{destination}"
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
    raw, _digest, _source = load_json_object(
        args.input_spec,
        expected_sha256=args.expected_input_spec_sha256,
        label="V2.1B execution build spec",
    )
    written = write_authority(raw, args.output)
    print(
        json.dumps(
            {
                "status": "V2.1B source execution authority built",
                "output": file_record(written),
                "complete_content_validation_command": (
                    "train_surface_region_v21b_conditioned_rank256_exact4x2.py "
                    "validate-authority"
                ),
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
