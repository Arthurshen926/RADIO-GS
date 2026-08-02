#!/usr/bin/env python3
"""Create a fail-closed provenance binding to the evaluation protocol freeze.

The binding has two deliberately separate scopes.  An external benchmark run
must name one exact canonical task and its frozen registry row.  Internal
fit/dev work must declare that external benchmarks remain unopened and cannot
name a task.  In both cases the checked-in freeze is fully validated, including
authoritative artifact hashes, before a receipt can be published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.scripts.validate_evaluation_protocol_freeze import (
    FreezeError,
    load_and_validate,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "evaluation_protocol_freeze_binding"
EXTERNAL_SCOPE = "external_benchmark"
UNOPENED_SCOPE = "external_benchmarks_unopened"
DEFAULT_FREEZE = Path(
    "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
)


class BindingError(ValueError):
    """Raised when a requested binding is incomplete or disagrees with the freeze."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _load_validated_freeze(
    freeze: Path, *, repo_root: Path | None
) -> tuple[Path, bytes, Mapping[str, Any]]:
    try:
        canonical = freeze.resolve(strict=True)
    except FileNotFoundError as error:
        raise BindingError(f"freeze does not exist: {freeze}") from error
    if not canonical.is_file():
        raise BindingError(f"freeze is not a regular file: {canonical}")

    identity_before = _stable_file_identity(canonical)
    try:
        payload = load_and_validate(
            canonical,
            root=repo_root.resolve() if repo_root is not None else None,
            verify_hashes=True,
        )
    except FreezeError as error:
        raise BindingError(f"freeze validation failed: {error}") from error
    encoded = canonical.read_bytes()
    if _stable_file_identity(canonical) != identity_before:
        raise BindingError("freeze changed while it was being validated")
    return canonical, encoded, payload


def build_binding(
    freeze: str | Path = DEFAULT_FREEZE,
    *,
    scope: str,
    canonical_task_id: str | None = None,
    registry_row: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the freeze and construct an exact provenance fragment."""

    canonical, encoded, payload = _load_validated_freeze(
        Path(freeze),
        repo_root=Path(repo_root) if repo_root is not None else None,
    )
    freeze_id = payload.get("freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id:
        raise BindingError("validated freeze has no non-empty freeze_id")

    tasks = payload.get("canonical_tasks")
    if not isinstance(tasks, Mapping):
        raise BindingError("validated freeze has no canonical_tasks mapping")

    task: dict[str, str] | None
    if scope == EXTERNAL_SCOPE:
        if not canonical_task_id or not registry_row:
            raise BindingError(
                "external benchmark scope requires canonical_task_id and registry_row"
            )
        frozen_task = tasks.get(canonical_task_id)
        if not isinstance(frozen_task, Mapping):
            raise BindingError(
                f"canonical task is not selected by the freeze: {canonical_task_id}"
            )
        frozen_row = frozen_task.get("registry_row")
        if registry_row != frozen_row:
            raise BindingError(
                "registry row does not match the frozen canonical task: "
                f"expected {frozen_row!r}, got {registry_row!r}"
            )
        task = {
            "canonical_task_id": canonical_task_id,
            "registry_row": registry_row,
        }
    elif scope == UNOPENED_SCOPE:
        if canonical_task_id is not None or registry_row is not None:
            raise BindingError(
                "external_benchmarks_unopened scope requires task to be null"
            )
        task = None
    else:
        raise BindingError(
            f"scope must be {EXTERNAL_SCOPE!r} or {UNOPENED_SCOPE!r}, got {scope!r}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "validated_and_bound",
        "scope": scope,
        "freeze": {
            "path": str(canonical),
            "sha256": _sha256_bytes(encoded),
            "freeze_id": freeze_id,
        },
        "task": task,
        "validation": {
            "validator": (
                "radio_gs.scripts.validate_evaluation_protocol_freeze."
                "load_and_validate"
            ),
            "authoritative_artifact_hashes_verified": True,
        },
    }


def write_binding_receipt(output: str | Path, binding: Mapping[str, Any]) -> Path:
    """Atomically publish one immutable JSON receipt without replacing a file."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise BindingError(
                f"binding receipt already exists: {destination}"
            ) from error
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_binding_receipt(
    receipt: str | Path, expected_binding: Mapping[str, Any]
) -> Path:
    """Verify an existing receipt without following links or accepting drift."""

    path = Path(receipt)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("binding receipt verification requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError as error:
        raise BindingError(f"binding receipt does not exist: {path}") from error
    except OSError as error:
        raise BindingError(f"binding receipt cannot be opened safely: {path}") from error
    try:
        info_before = os.fstat(descriptor)
        if not stat.S_ISREG(info_before.st_mode) or info_before.st_nlink != 1:
            raise BindingError("binding receipt must be a singly linked regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read()
        info_after = os.fstat(descriptor)
        if _stable_file_identity_from_stat(info_after) != _stable_file_identity_from_stat(
            info_before
        ):
            raise BindingError("binding receipt changed while it was being verified")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BindingError("binding receipt is not valid JSON") from error
    if payload != dict(expected_binding):
        raise BindingError("binding receipt differs from the validated freeze binding")
    return path


def _stable_file_identity_from_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--scope",
        required=True,
        choices=(EXTERNAL_SCOPE, UNOPENED_SCOPE),
    )
    parser.add_argument("--canonical-task-id")
    parser.add_argument("--registry-row")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an existing exact receipt instead of publishing a new one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    binding = build_binding(
        args.freeze,
        scope=args.scope,
        canonical_task_id=args.canonical_task_id,
        registry_row=args.registry_row,
        repo_root=args.repo_root,
    )
    output = (
        verify_binding_receipt(args.output, binding)
        if args.verify_existing
        else write_binding_receipt(args.output, binding)
    )
    print(
        f"bound {binding['freeze']['freeze_id']} "
        f"({binding['scope']}) at {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
