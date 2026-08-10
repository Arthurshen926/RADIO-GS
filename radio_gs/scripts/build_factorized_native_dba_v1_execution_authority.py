#!/usr/bin/env python3
"""Build the immutable source-only DBA-v1 execution authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_dba_v1 as trainer,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    validate_file_record,
    write_frozen_json,
)


def _bound_record(path: str, sha256: str, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": str(sha256)}
    observed = validate_file_record(record, label=label)
    if str(observed) != record["path"]:
        raise ValueError(f"{label} canonical path differs")
    return record


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = str(Path(args.training_output).expanduser().resolve())
    if str(args.training_output) != output:
        raise ValueError("DBA-v1 training output must be canonical absolute")
    authority = {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": trainer.SCHEMA_VERSION,
        "status": "authorized_source_only_dba_v1_exact4train_2validation",
        "implementation": file_record(Path(trainer.__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in sorted(trainer._DEPENDENCY_PATHS.items())
        },
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "source_contrast_v21_result": _bound_record(
            args.source_contrast_v21_result,
            args.source_contrast_v21_result_sha256,
            label="DBA-v1 source contrast V2.1 result",
        ),
        "fit_text_bank": _bound_record(
            args.fit_text_bank,
            args.fit_text_bank_sha256,
            label="DBA-v1 fit text bank",
        ),
        "canonical_negative_bank": _bound_record(
            args.canonical_negative_bank,
            args.canonical_negative_bank_sha256,
            label="DBA-v1 canonical negative bank",
        ),
        "benchmark_exclusion_manifest": _bound_record(
            args.benchmark_exclusion_manifest,
            args.benchmark_exclusion_manifest_sha256,
            label="DBA-v1 benchmark exclusion manifest",
        ),
        "training_output": output,
        "training_authorized": True,
        "target_execution_authorized": False,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "source_access": trainer.source_access(),
    }
    # Full source promotion is checked before this returns.  Text assets are
    # subsequently checked by prepare_inputs in the same fail-closed order.
    trainer.validate_execution_authority_header(authority)
    return authority


def write_authority(args: argparse.Namespace) -> Path:
    destination = Path(args.authority_output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"DBA-v1 authority already exists: {destination}")
    authority = build(args)
    # Verify the promoted warm-start chain before freezing a training permit.
    trainer.source_formal.validate_source_contrast_v21_result(
        authority["source_contrast_v21_result"]
    )
    return write_frozen_json(destination, authority)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contrast-v21-result", required=True)
    parser.add_argument("--source-contrast-v21-result-sha256", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-sha256", required=True)
    parser.add_argument("--canonical-negative-bank", required=True)
    parser.add_argument("--canonical-negative-bank-sha256", required=True)
    parser.add_argument("--benchmark-exclusion-manifest", required=True)
    parser.add_argument("--benchmark-exclusion-manifest-sha256", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--authority-output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = write_authority(args)
    print(
        json.dumps(
            {
                "status": "source-only DBA-v1 execution authority frozen",
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
