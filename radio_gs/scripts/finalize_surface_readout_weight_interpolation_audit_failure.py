#!/usr/bin/env python3
"""Close a consumed interpolation audit family after a terminal executor failure.

This finalizer is deliberately incapable of resolving, hashing, or loading the
audit bank or its manifest.  It reads only the committed opening receipt,
checks the absent intended confirmation path, and publishes a no-retry terminal
record on CPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

from radio_gs.utils.immutable_artifacts import (
    canonical_json_bytes,
    file_record,
    load_json_object,
    write_bytes_noclobber,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_readout_weight_interpolation_audit_failure_terminal"
OPENING_ARTIFACT_TYPE = (
    "surface_readout_weight_interpolation_audit90_opening_receipt"
)
FAILURE_STAGE = "post_audit_preterminal_receipt_binding"
DECISION = "confirmation_failed_family_closed_no_retry"
FORMAL_POOL_ROOT = Path("/mnt/pool")
_OPENING_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "opening_count",
    "audit_bank_loads_authorized",
    "selection_validation_completed",
    "query_free_recomputation_completed",
    "selection",
    "diagnostic",
    "declared_audit_bank",
    "intended_confirmation_output",
    "implementation",
    "implementation_closure",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _cpu_only_preflight() -> None:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"},
        "set CUDA_VISIBLE_DEVICES='' (or -1) for this CPU-only failure finalizer",
    )


def _within_formal_pool(path: Path) -> bool:
    root = FORMAL_POOL_ROOT.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _existing_receipt(raw: str | Path) -> tuple[Path, Path]:
    lexical = Path(raw)
    _require(lexical.is_absolute(), "opening receipt must be an absolute path")
    _require(not lexical.is_symlink(), "opening receipt final component cannot be a symlink")
    resolved = lexical.resolve(strict=True)
    _require(resolved.is_file(), "opening receipt must resolve to a regular file")
    _require(_within_formal_pool(resolved), "opening receipt must resolve below /mnt/pool")
    return lexical, resolved


def _absent_canonical_child(raw: str | Path, *, label: str) -> tuple[Path, Path]:
    lexical = Path(raw)
    _require(lexical.is_absolute(), f"{label} must be an absolute path")
    _require(not lexical.exists() and not lexical.is_symlink(), f"{label} already exists")
    parent = lexical.parent.resolve(strict=True)
    _require(parent.is_dir(), f"{label} parent must resolve to a directory")
    resolved = parent / lexical.name
    _require(
        not resolved.exists() and not resolved.is_symlink(),
        f"{label} canonical target already exists",
    )
    _require(_within_formal_pool(resolved), f"{label} must resolve below /mnt/pool")
    return lexical, resolved


def _validate_opening_receipt(
    value: object,
    *,
    executor_sha256: str,
    expected_confirmation_lexical: Path,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "opening receipt must contain a JSON object")
    receipt = dict(value)
    _require(set(receipt) == _OPENING_KEYS, "opening receipt fields differ")
    _require(receipt.get("schema_version") == 1, "opening receipt schema differs")
    _require(
        receipt.get("artifact_type") == OPENING_ARTIFACT_TYPE,
        "opening receipt artifact type differs",
    )
    _require(
        receipt.get("status") == "one_shot_opening_authorization_committed"
        and receipt.get("opening_count") == 1
        and receipt.get("audit_bank_loads_authorized") == 1
        and receipt.get("selection_validation_completed") is True
        and receipt.get("query_free_recomputation_completed") is True,
        "opening receipt is not a committed one-shot authorization",
    )
    _require(
        receipt.get("intended_confirmation_output")
        == str(expected_confirmation_lexical),
        "expected confirmation path differs from the committed receipt",
    )
    implementation = receipt.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation) == {"path", "sha256"}
        and implementation.get("sha256") == executor_sha256,
        "executor SHA differs from the committed receipt",
    )
    closure = receipt.get("implementation_closure")
    _require(
        isinstance(closure, list)
        and closure
        and closure[0]
        == {
            "role": "audit_confirmation_executor",
            "path": implementation["path"],
            "sha256": executor_sha256,
        },
        "opening receipt executor closure differs",
    )
    # Intentionally do not inspect the declared_audit_bank value.  Its mere
    # schema presence is enough to identify the committed receipt, while any
    # path operation on its contents is forbidden in this finalizer.
    return receipt


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    _cpu_only_preflight()
    receipt_sha256 = str(args.opening_receipt_sha256)
    executor_sha256 = str(args.executor_sha256)
    _require(_is_sha256(receipt_sha256), "opening receipt SHA256 is invalid")
    _require(_is_sha256(executor_sha256), "executor SHA256 is invalid")
    receipt_lexical, receipt_resolved = _existing_receipt(args.opening_receipt)
    confirmation_lexical, confirmation_resolved = _absent_canonical_child(
        args.expected_confirmation,
        label="expected confirmation",
    )
    output_lexical, output_resolved = _absent_canonical_child(
        args.output,
        label="failure terminal output",
    )
    _require(
        output_resolved != confirmation_resolved,
        "failure terminal and expected confirmation paths must differ",
    )
    receipt, observed_receipt_sha, receipt_source = load_json_object(
        receipt_lexical,
        expected_sha256=receipt_sha256,
        label="committed audit opening receipt",
    )
    _require(
        observed_receipt_sha == receipt_sha256 and receipt_source == receipt_resolved,
        "opening receipt identity differs",
    )
    receipt = _validate_opening_receipt(
        receipt,
        executor_sha256=executor_sha256,
        expected_confirmation_lexical=confirmation_lexical,
    )
    finalizer_source = Path(__file__).resolve(strict=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "complete_immutable_failure_terminal",
        "decision": DECISION,
        "main_result_eligible": False,
        "family_closed": True,
        "retry_forbidden": True,
        "post_audit_retuning_forbidden": True,
        "failure_stage": FAILURE_STAGE,
        "audit": {
            "opening_receipt_committed": True,
            "consumed_by_failed_executor": True,
            "artifact_accessed_by_finalizer": False,
            "manifest_accessed_by_finalizer": False,
        },
        "opening_receipt": {
            "lexical_path": str(receipt_lexical),
            "resolved_path": str(receipt_resolved),
            "sha256": receipt_sha256,
            "status": receipt["status"],
            "opening_count": receipt["opening_count"],
        },
        "failed_executor": {
            "path": receipt["implementation"]["path"],
            "sha256": executor_sha256,
            "authority": "committed_pre_audit_opening_receipt",
        },
        "confirmation": {
            "expected_lexical_path": str(confirmation_lexical),
            "resolved_path": str(confirmation_resolved),
            "exists": False,
        },
        "terminal_output": {
            "requested_lexical_path": str(output_lexical),
            "resolved_path": str(output_resolved),
        },
        "provenance": {
            "finalizer_implementation": file_record(finalizer_source),
            "receipt_verified_without_audit_bank_resolution": True,
            "only_receipt_executor_and_path_metadata_read": True,
        },
    }
    write_bytes_noclobber(output_resolved, canonical_json_bytes(payload) + b"\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-receipt", required=True, type=Path)
    parser.add_argument("--opening-receipt-sha256", required=True)
    parser.add_argument("--executor-sha256", required=True)
    parser.add_argument("--expected-confirmation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = finalize(args)
    print(
        {
            "output": payload["terminal_output"]["resolved_path"],
            "decision": payload["decision"],
            "failure_stage": payload["failure_stage"],
        }
    )


if __name__ == "__main__":
    main()
