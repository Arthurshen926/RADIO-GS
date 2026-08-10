#!/usr/bin/env python3
"""Build the independent no-clobber source-only V2.1A execution authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21_pilot as pilot,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a_rescue as rescue,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    write_frozen_json,
)


def build(
    base_v21_execution_authority: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        base_v21_execution_authority,
        expected_sha256=expected_sha256,
        label="V2.1 parent asset execution authority",
    )
    parent = pilot.validate_execution_authority(raw)
    implementations = rescue._implementation_records()
    authority = {
        "schema": rescue.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_v21a_anchor_triplet_full30_rescue",
        **implementations,
        "base_v21_asset_execution_authority": {
            "path": str(source),
            "sha256": digest,
        },
        "parent_implementation": dict(parent["implementation"]),
        "active_pair_addendum": dict(parent["active_pair_addendum"]),
        "cohort_authority": dict(parent["cohort_authority"]),
        "pilot_cohort_region_view_registry": dict(
            parent["pilot_cohort_region_view_registry"]
        ),
        "benchmark_exclusion_manifest": dict(
            parent["benchmark_exclusion_manifest"]
        ),
        "fit_text_bank": dict(parent["fit_text_bank"]),
        "canonical_negative_bank": dict(parent["canonical_negative_bank"]),
        "compositional_banks": {
            name: dict(row) for name, row in parent["compositional_banks"].items()
        },
        "typed_relation_authority": dict(parent["typed_relation_authority"]),
        "source_train": [dict(row) for row in parent["source_train"]],
        "source_validation": [dict(row) for row in parent["source_validation"]],
        "training_contract_sha256": rescue.TRAINING_CONTRACT_SHA256,
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": rescue.source_access(),
    }
    return rescue.validate_execution_authority(authority)


def write_authority(
    base_v21_execution_authority: str | Path,
    *,
    expected_sha256: str,
    output: str | Path,
) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"V2.1A execution-authority builder refuses to clobber: {destination}"
        )
    return write_frozen_json(
        destination,
        build(
            base_v21_execution_authority,
            expected_sha256=expected_sha256,
        ),
    )


def validate(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    inputs = rescue.prepare_inputs(path, expected_sha256=expected_sha256)
    return {
        "schema": "radio_gs.surface_region_v21a_execution_validation.v1",
        "status": "source_only_v21a_execution_authority_validated",
        "execution_authority": {
            "path": inputs.execution["verified_path"],
            "sha256": inputs.execution["verified_sha256"],
        },
        "source_train": [item.scene_id for item in inputs.train],
        "source_validation": [item.scene_id for item in inputs.validation],
        "independent_from_parent_training_implementation": (
            inputs.execution["implementation"]
            != inputs.execution["parent_implementation"]
        ),
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--base-v21-execution-authority", required=True)
    create.add_argument("--expected-base-v21-execution-authority-sha256", required=True)
    create.add_argument("--output", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--execution-authority", required=True)
    check.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        path = write_authority(
            args.base_v21_execution_authority,
            expected_sha256=args.expected_base_v21_execution_authority_sha256,
            output=args.output,
        )
        result = {
            "status": "source_only_v21a_execution_authority_built",
            "path": str(path),
        }
    else:
        result = validate(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["build", "validate", "write_authority"]
