#!/usr/bin/env python3
"""Build the fixed source-only V2.1 4+2 execution authority without clobbering.

The caller-SHA-bound build spec supplies only immutable file records and the
six fixed scene records.  Protocol identity, implementation, addendum, source
access and authorization flags are supplied here.  The resulting authority is
structurally validated and every file record is SHA-checked.  The trainer's
``validate-authority`` command remains the complete content/replay validator.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
    write_frozen_json,
)


BUILD_SPEC_SCHEMA = (
    "radio_gs.surface_region_typed_context_response_listwise_v21_pilot_"
    "execution_build_spec.v1"
)
SCHEMA_VERSION = 1


def validate_build_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1 pilot execution build spec must be a mapping")
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
        raise ValueError("V2.1 pilot execution build spec fields differ")
    return spec


def _validate_all_file_records(authority: Mapping[str, Any]) -> None:
    for name in (
        "implementation",
        "active_pair_addendum",
        "cohort_authority",
        "pilot_cohort_region_view_registry",
        "benchmark_exclusion_manifest",
        "fit_text_bank",
        "canonical_negative_bank",
    ):
        validate_file_record(authority[name], label=f"V2.1 execution {name}")
    for name in pilot.COMPONENT_WEIGHTS:
        record = authority["compositional_banks"][name]
        validate_file_record(
            {"path": record["path"], "sha256": record["sha256"]},
            label=f"V2.1 execution {name}",
        )
    relation = authority["typed_relation_authority"]
    validate_file_record(
        {"path": relation["path"], "sha256": relation["sha256"]},
        label="V2.1 execution typed relation authority",
    )
    for split in ("source_train", "source_validation"):
        for row in authority[split]:
            scene = row["scene_id"]
            for name in (
                "training_shard",
                "adaptive_context",
                "hard_negative_authority",
            ):
                validate_file_record(
                    row[name], label=f"V2.1 execution {split} {scene} {name}"
                )


def build(spec: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_build_spec(spec)
    authority = {
        "schema": pilot.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_4train_2validation_v21_pilot",
        "implementation": file_record(Path(pilot.__file__).resolve()),
        "active_pair_addendum": file_record(
            Path(pilot.__file__).resolve().parents[2] / pilot.ACTIVE_PAIR_ADDENDUM
        ),
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
        "source_access": pilot.source_access(),
    }
    authority = pilot.validate_execution_authority(authority)
    _validate_all_file_records(authority)
    return authority


def write_authority(spec: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "V2.1 pilot execution-authority builder refuses to clobber: "
            f"{destination}"
        )
    return write_frozen_json(destination, build(spec))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--expected-input-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw, _digest, _source = load_json_object(
        args.input_spec,
        expected_sha256=args.expected_input_spec_sha256,
        label="V2.1 pilot execution build spec",
    )
    written = write_authority(raw, args.output)
    print(
        json.dumps(
            {
                "status": "V2.1 pilot execution authority built",
                "output": file_record(written),
                "complete_content_validation_command": (
                    "train_surface_region_typed_context_response_listwise_v21_"
                    "pilot.py validate-authority"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
