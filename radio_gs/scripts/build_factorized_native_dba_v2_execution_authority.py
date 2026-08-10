#!/usr/bin/env python3
"""Build the immutable source-only precision-ranking DBA-v2 authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v2 as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    validate_file_record,
    write_frozen_json,
)


def _bound(path: str, sha256: str, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": str(sha256)}
    observed = validate_file_record(record, label=label)
    if str(observed) != record["path"]:
        raise ValueError(f"{label} canonical path differs")
    return record


def build(args: argparse.Namespace) -> dict:
    output = str(Path(args.training_output).expanduser().resolve())
    if output != str(args.training_output):
        raise ValueError("DBA-v2 training output must be canonical absolute")
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": trainer.SCHEMA_VERSION,
        "status": (
            "authorized_source_only_precision_ranking_dba_v2_"
            "exact4train_2validation"
        ),
        "implementation": file_record(Path(trainer.__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in sorted(trainer._DEPENDENCY_PATHS.items())
        },
        "design_preregistration": file_record(trainer.DESIGN_PATH),
        "base_dba_v1_execution_authority": _bound(
            args.base_dba_v1_execution_authority,
            args.base_dba_v1_execution_authority_sha256,
            label="DBA-v2 base DBA-v1 execution authority",
        ),
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "training_output": output,
        "training_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    trainer.validate_execution_authority_header(authority)
    return authority


def write_authority(args: argparse.Namespace) -> Path:
    destination = Path(args.authority_output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"DBA-v2 authority already exists: {destination}")
    authority = build(args)
    # Recursively validate the source/text/exclusion chain before training is
    # authorized; no target or runtime query is reachable through this path.
    trainer.dba_v1.prepare_inputs(
        authority["base_dba_v1_execution_authority"]["path"],
        expected_sha256=authority["base_dba_v1_execution_authority"]["sha256"],
    )
    return write_frozen_json(destination, authority)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dba-v1-execution-authority", required=True)
    parser.add_argument("--base-dba-v1-execution-authority-sha256", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--authority-output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = write_authority(args)
    print(
        json.dumps(
            {
                "status": "source-only precision-ranking DBA-v2 authority frozen",
                "authority": file_record(written),
                "training_output": str(Path(args.training_output).resolve()),
                "target_query_or_metric_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["build", "build_parser", "write_authority"]
