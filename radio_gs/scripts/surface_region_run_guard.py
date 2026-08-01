#!/usr/bin/env python3
"""Fail-closed runtime-closure and thermal-canary audits for SurfaceRegion.

This module is deliberately independent of the checkpoint loading helpers used
by the training code.  A concurrent edit to one of those helpers must be
detected by the closure audit before another GPU stage starts, rather than
being imported by the auditor itself.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping

import torch


REPO_PYTHON_ENTRYPOINTS = (
    "radio_gs/scripts/surface_region_run_guard.py",
    "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
    "radio_gs/scripts/build_scannet_surface_region_cache.py",
    "radio_gs/scripts/train_surface_region_summary_readout.py",
)
REPO_SHELL_SOURCES = (
    "radio_gs/scripts/run_surface_region_context_recovery_screen.sh",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    "radio_gs/scripts/run_repo_python.sh",
)
RADIO_ROOT_FILES = ("hubconf.py", "hf_hub.py")
RUNTIME_PACKAGES = (
    "numpy",
    "Pillow",
    "scipy",
    "timm",
    "torch",
    "torchvision",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "RADIO_GS_REPO_ROOT",
    "RADIO_GS_DRIVER_LIBRARY",
    "RADIO_GS_LD_LIBRARY_PATH",
    "RADIO_GS_SAM3_SOURCE",
    "RADIO_GS_SITE_PACKAGES",
)
RUNTIME_IMPORT_MODULES = (
    "radio_gs",
    "radio_gs.interfaces.frozen_radio_views",
    "radio_gs.interfaces.surface_region_contract",
    "radio_gs.interfaces.surface_region_summary",
    "radio_gs.models.radio_adaptors",
    "radio_gs.models.siglip_projection",
    "radio_gs.scripts.build_scannet_surface_region_cache",
    "radio_gs.scripts.surface_gpu1_lock_supervisor",
    "radio_gs.scripts.surface_region_run_guard",
    "radio_gs.scripts.train_surface_region_summary_readout",
    "radio_gs.utils.checkpoint_io",
    "radio_gs.utils.immutable_artifacts",
)
TELEMETRY_COLUMNS = (
    "timestamp",
    "gpu",
    "bus_id",
    "temp_c",
    "power_w",
    "power_limit_w",
    "util_pct",
    "memory_mib",
    "pstate",
    "event",
)
TELEMETRY_FAULT_MARKERS = (
    "thermal_abort",
    "pcie_unresponsive",
    "telemetry_failed",
    "peer_telemetry_failed",
    "cuda_release_verification_failed",
)
KERNEL_FAULT_PATTERN = re.compile(
    r"(?:\bNVRM\b.*\bXid\b|fallen off|PCIe.*(?:error|fatal)|"
    r"GPU.*lost PCIe)",
    re.IGNORECASE,
)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    target = Path(os.path.abspath(os.fspath(path)))
    info = os.stat(target, follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a non-symlink regular file: {target}")
    return target


def file_identity(path: str | Path) -> dict[str, int | str]:
    source = _regular_file(Path(path), label="closure file")
    info = os.stat(source, follow_symlinks=False)
    return {
        "path": str(source),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "links": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _module_file(repo_root: Path, module_name: str) -> Path | None:
    if not module_name.startswith("radio_gs"):
        return None
    stem = repo_root.joinpath(*module_name.split("."))
    source = stem.with_suffix(".py")
    if source.is_file():
        return source
    package = stem / "__init__.py"
    return package if package.is_file() else None


def _module_name(repo_root: Path, source: Path) -> str:
    relative = source.resolve().relative_to(repo_root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_initializers(repo_root: Path, source: Path) -> Iterable[Path]:
    relative = source.resolve().relative_to(repo_root.resolve())
    parent = relative.parent
    while parent.parts:
        initializer = repo_root / parent / "__init__.py"
        if initializer.is_file():
            yield initializer
        parent = parent.parent


def discover_repo_python_closure(
    repo_root: str | Path,
    entrypoints: Iterable[str] = REPO_PYTHON_ENTRYPOINTS,
) -> tuple[str, ...]:
    """Discover the static RADIO-GS import closure for the GPU entrypoints."""

    root = Path(repo_root).resolve()
    queue = [root / relative for relative in entrypoints]
    discovered: set[Path] = set()
    while queue:
        source = queue.pop()
        source = _regular_file(source, label="Python closure source").resolve()
        if source in discovered:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot parse closure source: {source}") from error
        discovered.add(source)
        # Package initializers execute before a submodule import.  They must
        # therefore be parsed as closure roots rather than merely added to the
        # file inventory; otherwise imports re-exported by ``__init__.py`` are
        # available at runtime but absent from the frozen source closure.
        for initializer in _package_initializers(root, source):
            resolved_initializer = initializer.resolve()
            if resolved_initializer not in discovered:
                queue.append(resolved_initializer)
        module_name = _module_name(root, source)
        package_name = (
            module_name
            if source.name == "__init__.py"
            else module_name.rpartition(".")[0]
        )
        for node in ast.walk(tree):
            module_candidates: list[str] = []
            if isinstance(node, ast.Import):
                module_candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                raw_module = node.module or ""
                if node.level:
                    try:
                        resolved = importlib.util.resolve_name(
                            "." * node.level + raw_module,
                            package_name,
                        )
                    except (ImportError, ValueError) as error:
                        raise ValueError(
                            f"cannot resolve import in closure source: {source}"
                        ) from error
                else:
                    resolved = raw_module
                if resolved:
                    module_candidates.append(resolved)
                    module_candidates.extend(
                        f"{resolved}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
            for candidate in module_candidates:
                dependency = _module_file(root, candidate)
                if dependency is not None and dependency.resolve() not in discovered:
                    queue.append(dependency)
    return tuple(
        sorted(str(path.relative_to(root)) for path in discovered)
    )


def repo_source_closure(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    runtime_entrypoints: list[str] = []
    for module_name in RUNTIME_IMPORT_MODULES:
        source = _module_file(root, module_name)
        if source is None:
            raise ValueError(
                f"runtime import has no in-repository source: {module_name}"
            )
        runtime_entrypoints.append(str(source.relative_to(root)))
    closure_entrypoints = tuple(
        dict.fromkeys((*REPO_PYTHON_ENTRYPOINTS, *runtime_entrypoints))
    )
    python_sources = discover_repo_python_closure(
        root,
        entrypoints=closure_entrypoints,
    )
    relative_paths = sorted(set(python_sources) | set(REPO_SHELL_SOURCES))
    files = {
        relative: sha256_file(
            _regular_file(root / relative, label="repository closure source")
        )
        for relative in relative_paths
    }
    payload: dict[str, object] = {
        "python_entrypoints": list(closure_entrypoints),
        "runtime_import_modules": list(RUNTIME_IMPORT_MODULES),
        "shell_sources": list(REPO_SHELL_SOURCES),
        "files": files,
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def radio_source_tree(radio_repo: str | Path) -> dict[str, object]:
    root = Path(radio_repo).resolve()
    if not root.is_dir():
        raise ValueError(f"RADIO source root is missing: {root}")
    sources: set[Path] = set()
    for relative in RADIO_ROOT_FILES:
        sources.add(_regular_file(root / relative, label="RADIO source"))
    radio_package = root / "radio"
    for source in radio_package.rglob("*.py"):
        sources.add(_regular_file(source, label="RADIO source"))
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(sources)
    }
    payload: dict[str, object] = {
        "root": str(root),
        "selection": ["hubconf.py", "hf_hub.py", "radio/**/*.py"],
        "files": files,
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def runtime_fingerprint(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    packages: dict[str, str] = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    executable = _regular_file(Path(sys.executable).resolve(), label="Python executable")
    cudnn_version = torch.backends.cudnn.version()
    imported_modules: dict[str, dict[str, str]] = {}
    for module_name in RUNTIME_IMPORT_MODULES:
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"runtime module has no source path: {module_name}")
        source = _regular_file(Path(raw_path).resolve(), label="runtime import")
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"runtime import escaped source snapshot: {module_name} -> {source}"
            ) from error
        imported_modules[module_name] = {
            "path": str(source),
            "relative_path": str(relative),
            "sha256": sha256_file(source),
        }
    return {
        "repository_import_root": str(root),
        "imported_modules": imported_modules,
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_version": sys.version,
        "python_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": (
            int(cudnn_version) if cudnn_version is not None else None
        ),
        "environment": {
            key: os.environ.get(key)
            for key in RUNTIME_ENVIRONMENT_KEYS
        },
    }


def build_runtime_closure(
    *,
    repo_root: str | Path,
    radio_repo: str | Path,
    radio_checkpoint: str | Path,
    checkpoint_sha256: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    payload: dict[str, object] = {
        "schema_version": 1,
        "repository_sources": repo_source_closure(repo_root),
        "radio_source_tree": radio_source_tree(radio_repo),
        "runtime_fingerprint": runtime_fingerprint(repo_root),
        "radio_checkpoint": {
            **file_identity(radio_checkpoint),
            "sha256": checkpoint_sha256,
        },
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def _stable_artifact_bytes(path: str | Path, *, label: str) -> tuple[bytes, str, Path]:
    source = Path(os.path.abspath(os.fspath(path)))
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("SurfaceRegion audits require O_NOFOLLOW")
    descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        before = os.fstat(handle.fileno())
        path_before = os.stat(source, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise ValueError(f"{label} is not a stable regular file")
        value = handle.read()
        after = os.fstat(handle.fileno())
        path_after = os.stat(source, follow_symlinks=False)
        fingerprint = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if not (
            fingerprint(before)
            == fingerprint(after)
            == fingerprint(path_before)
            == fingerprint(path_after)
        ):
            raise ValueError(f"{label} changed while being read")
    return value, hashlib.sha256(value).hexdigest(), source.parent.resolve() / source.name


def _artifact_file_record(path: str | Path, *, label: str) -> dict[str, str]:
    _, digest, source = _stable_artifact_bytes(path, label=label)
    return {"path": str(source), "sha256": digest}


def _validate_artifact_file_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    observed = _artifact_file_record(str(record["path"]), label=label)
    if observed != dict(record):
        raise ValueError(f"{label} SHA/path differs")
    return Path(observed["path"])


def _json_object(path: str | Path) -> dict:
    encoded, _, _ = _stable_artifact_bytes(path, label="JSON artifact")
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_publish_json(path: str | Path, payload: Mapping) -> None:
    output = Path(path)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _atomic_publish_bytes(output, encoded)


def _atomic_publish_bytes(path: str | Path, encoded: bytes) -> None:
    """Publish immutable bytes without following or replacing an old name."""

    output = Path(path)
    if os.path.lexists(output):
        observed, _, _ = _stable_artifact_bytes(
            output,
            label="immutable SurfaceRegion report",
        )
        if observed != encoded:
            raise ValueError(f"existing immutable report differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            observed, _, _ = _stable_artifact_bytes(
                output,
                label="concurrent immutable SurfaceRegion report",
            )
            if observed != encoded:
                raise ValueError(f"concurrent immutable report differs: {output}")
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def verify_runtime_closure(
    manifest_path: str | Path,
    *,
    full_checkpoint: bool = False,
) -> tuple[dict, dict]:
    manifest_source = Path(manifest_path).resolve()
    manifest = _json_object(manifest_source)
    expected = manifest.get("runtime_closure")
    if not isinstance(expected, dict):
        raise ValueError("SurfaceRegion manifest lacks runtime_closure")
    checkpoint_record = expected.get("radio_checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ValueError("SurfaceRegion runtime closure lacks checkpoint record")
    current = build_runtime_closure(
        repo_root=Path(__file__).resolve().parents[2],
        radio_repo=str(manifest["radio_repo"]),
        radio_checkpoint=str(manifest["radio_checkpoint"]),
        checkpoint_sha256=str(checkpoint_record.get("sha256", "")),
    )
    if current != expected:
        expected_sources = expected.get("repository_sources", {}).get("files", {})
        current_sources = current.get("repository_sources", {}).get("files", {})
        changed = sorted(
            key
            for key in set(expected_sources) | set(current_sources)
            if expected_sources.get(key) != current_sources.get(key)
        )
        suffix = f"; changed repository sources: {changed}" if changed else ""
        raise ValueError(f"SurfaceRegion runtime closure changed{suffix}")
    if (
        manifest.get("source_snapshot_root")
        != current["runtime_fingerprint"]["repository_import_root"]
        or manifest.get("source_snapshot_import_root")
        != current["runtime_fingerprint"]["repository_import_root"]
        or manifest.get("source_snapshot_tree_sha256")
        != current["repository_sources"]["digest"]
    ):
        raise ValueError("SurfaceRegion source-snapshot binding differs")
    if manifest.get("runner_sha256") != current["repository_sources"]["files"].get(
        "radio_gs/scripts/run_surface_region_context_recovery_screen.sh"
    ):
        raise ValueError("SurfaceRegion runner hash differs from runtime closure")
    if full_checkpoint:
        current_digest = sha256_file(str(manifest["radio_checkpoint"]))
        if current_digest != checkpoint_record["sha256"]:
            raise ValueError("SurfaceRegion RADIO checkpoint digest changed")
    return manifest, current


def write_closure_report(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    phase: str,
    full_checkpoint: bool,
    attempt_root: str | Path | None = None,
    log_root: str | Path | None = None,
) -> dict:
    manifest, closure = verify_runtime_closure(
        manifest_path,
        full_checkpoint=full_checkpoint,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "surface_region_runtime_closure_audit",
        "status": "runtime_closure_verified",
        "phase": str(phase),
        "run_manifest": str(Path(manifest_path).resolve()),
        "run_manifest_sha256": sha256_file(manifest_path),
        "runtime_closure_digest": closure["digest"],
        "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
        "full_checkpoint_rehashed": bool(full_checkpoint),
    }
    if (attempt_root is None) != (log_root is None):
        raise ValueError("attempt_root and log_root must be provided together")
    if attempt_root is not None and log_root is not None:
        report["attempt_inventory"] = audit_attempt_inventory(
            manifest_path=manifest_path,
            attempt_root=attempt_root,
            log_root=log_root,
        )
    _atomic_publish_json(output_path, report)
    return report


def _telemetry_interval_bytes(
    telemetry_path: str | Path,
    *,
    start_line: int,
    end_line: int,
) -> tuple[bytes, Path]:
    if start_line < 0 or end_line < start_line:
        raise ValueError("invalid canary telemetry line interval")
    encoded, _, source = _stable_artifact_bytes(
        telemetry_path,
        label="SurfaceRegion telemetry",
    )
    if encoded and not encoded.endswith(b"\n"):
        raise ValueError("SurfaceRegion telemetry has an unterminated final row")
    lines = encoded.splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError("canary telemetry interval exceeds the telemetry file")
    return b"".join(lines[start_line:end_line]), source


def telemetry_interval_record(
    telemetry_path: str | Path,
    *,
    start_line: int,
    end_line: int,
) -> dict[str, int | str]:
    encoded, source = _telemetry_interval_bytes(
        telemetry_path,
        start_line=start_line,
        end_line=end_line,
    )
    return {
        "path": str(source),
        "start_line": int(start_line),
        "end_line": int(end_line),
        "line_count": int(end_line - start_line),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _telemetry_rows(
    telemetry_path: str | Path,
    *,
    start_line: int,
    end_line: int,
) -> list[dict[str, str]]:
    encoded, _ = _telemetry_interval_bytes(
        telemetry_path,
        start_line=start_line,
        end_line=end_line,
    )
    try:
        selected_lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("SurfaceRegion telemetry is not UTF-8") from error
    rows: list[dict[str, str]] = []
    for offset, values in enumerate(csv.reader(selected_lines), start=1):
        line_number = start_line + offset
        if values == list(TELEMETRY_COLUMNS):
            continue
        if len(values) != len(TELEMETRY_COLUMNS):
            raise ValueError(f"invalid telemetry row {line_number}")
        rows.append(dict(zip(TELEMETRY_COLUMNS, values)))
    return rows


def summarize_canary_telemetry(
    rows: Iterable[Mapping[str, str]],
    *,
    expected_gpu: int = 1,
    maximum_temperature_c: int = 71,
    peer_gpu: int | None = 0,
) -> dict[str, int | float]:
    temperatures: list[int] = []
    powers: list[float] = []
    utilizations: list[int] = []
    pause_count = 0
    resume_count = 0
    peer_interrupt_count = 0
    paused = False
    sample_count = 0
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        if row.get("gpu") != str(expected_gpu):
            raise ValueError(f"canary telemetry row {index} belongs to another GPU")
        event = str(row.get("event", ""))
        if any(marker in event for marker in TELEMETRY_FAULT_MARKERS):
            raise ValueError(f"canary telemetry contains fault event: {event}")
        if event.startswith("peer_activity_interrupt_release_cuda_"):
            if peer_gpu is None:
                raise ValueError(
                    "GPU1-only canary contains a forbidden peer interruption"
                )
            peer_interrupt_count += 1
        if event.startswith("soft_pause_"):
            if peer_gpu is None:
                if not event.startswith(f"soft_pause_gpu{expected_gpu}_t"):
                    raise ValueError(
                        "GPU1-only canary paused for an unexpected reason"
                    )
                if "peer" in event:
                    raise ValueError(
                        "GPU1-only canary contains a peer pause marker"
                    )
            elif f"peer{peer_gpu}_activity" not in event:
                raise ValueError("canary paused for an unexpected non-peer reason")
            if paused:
                raise ValueError("canary telemetry contains nested soft pauses")
            paused = True
            pause_count += 1
        elif event.startswith("soft_resume"):
            if not paused:
                raise ValueError("canary telemetry resumes without a peer pause")
            paused = False
            resume_count += 1
        temp = str(row.get("temp_c", "")).strip()
        if temp:
            if re.fullmatch(r"[0-9]+", temp) is None:
                raise ValueError("canary telemetry has an invalid temperature")
            temperatures.append(int(temp))
            sample_count += 1
        power = str(row.get("power_w", "")).strip()
        if power:
            try:
                powers.append(float(power))
            except ValueError as error:
                raise ValueError("canary telemetry has invalid power") from error
        utilization = str(row.get("util_pct", "")).strip()
        if utilization:
            if re.fullmatch(r"[0-9]+", utilization) is None:
                raise ValueError("canary telemetry has invalid utilization")
            utilizations.append(int(utilization))
    if not temperatures:
        raise ValueError("canary telemetry contains no GPU samples")
    if max(temperatures) > int(maximum_temperature_c):
        raise ValueError(
            f"canary reached {max(temperatures)}C, above {maximum_temperature_c}C"
        )
    if paused or pause_count != resume_count:
        raise ValueError("canary soft pause/resume events are not paired")
    return {
        "samples": sample_count,
        "minimum_temperature_c": min(temperatures),
        "maximum_temperature_c": max(temperatures),
        "maximum_power_w": max(powers) if powers else 0.0,
        "maximum_utilization_pct": max(utilizations) if utilizations else 0,
        "soft_pause_count": pause_count if peer_gpu is None else 0,
        "soft_resume_count": resume_count if peer_gpu is None else 0,
        "peer_pause_count": pause_count if peer_gpu is not None else 0,
        "peer_resume_count": resume_count if peer_gpu is not None else 0,
        "peer_interrupt_count": peer_interrupt_count,
        "fault_event_count": 0,
    }


def validate_attempt_receipt(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path,
    expected_stage: str,
    expected_index: int,
    expected_log: str | Path,
    expected_command: list[str] | None = None,
) -> dict:
    manifest_record = _artifact_file_record(
        manifest_path,
        label="SurfaceRegion run manifest",
    )
    receipt = _json_object(receipt_path)
    manifest = _json_object(manifest_path)
    attempt_contract = manifest.get("attempt_receipt_contract", {})
    owner_audit_required = bool(
        isinstance(attempt_contract, Mapping)
        and attempt_contract.get("owner_audit_required") is True
        and attempt_contract.get("owner_audit_location") == "beside_receipt"
    )
    expected_fields = {
        "artifact_type",
        "schema_version",
        "run_manifest",
        "stage",
        "attempt_index",
        "command",
        "command_status",
        "result",
        "log",
        "telemetry_interval",
        "kernel_journal",
        "gpu_release_postflight",
        "terminal",
        "sidecar",
        "peer_activity_action",
        "peer_activity_interrupt_exit_code",
    }
    if owner_audit_required:
        expected_fields.add("owner_audit")
    if set(receipt) != expected_fields:
        raise ValueError("SurfaceRegion attempt receipt fields differ")
    safety = manifest["thermal_safety_contract"]
    if (
        receipt["artifact_type"] != "surface-region-stage-attempt-v1"
        or receipt["schema_version"] != 1
        or receipt["run_manifest"] != manifest_record
        or receipt["stage"] != expected_stage
        or receipt["attempt_index"] != int(expected_index)
        or receipt["peer_activity_action"]
        != safety["peer_activity_action"]
        or receipt["peer_activity_interrupt_exit_code"]
        != safety["peer_activity_interrupt_exit_code"]
        or not isinstance(receipt["command"], list)
        or not all(isinstance(value, str) for value in receipt["command"])
    ):
        raise ValueError("SurfaceRegion attempt receipt identity differs")
    if expected_command is not None and receipt["command"] != expected_command:
        raise ValueError("SurfaceRegion attempt command differs")
    expected_log_record = _artifact_file_record(
        expected_log,
        label="SurfaceRegion attempt log",
    )
    if receipt["log"] != expected_log_record:
        raise ValueError("SurfaceRegion attempt log record differs")
    _validate_artifact_file_record(receipt["log"], label="SurfaceRegion attempt log")
    owner_audit_events: list[str] = []
    if owner_audit_required:
        expected_owner_audit_path = Path(receipt_path).with_suffix(
            ".owner_audit.csv"
        )
        expected_owner_audit_record = _artifact_file_record(
            expected_owner_audit_path,
            label="SurfaceRegion attempt owner audit",
        )
        if receipt["owner_audit"] != expected_owner_audit_record:
            raise ValueError("SurfaceRegion attempt owner audit record differs")
        owner_encoded, _, _ = _stable_artifact_bytes(
            expected_owner_audit_path,
            label="SurfaceRegion attempt owner audit",
        )
        try:
            owner_rows = list(
                csv.reader(owner_encoded.decode("utf-8").splitlines())
            )
        except UnicodeDecodeError as error:
            raise ValueError("SurfaceRegion owner audit is not UTF-8") from error
        owner_header = [
            "timestamp",
            "gpu_uuid",
            "child_pgid",
            "owner_pids",
            "child_owner_pids",
            "foreign_owner_pids",
            "event",
        ]
        if (
            len(owner_rows) < 2
            or owner_rows[0] != owner_header
            or any(len(row) != len(owner_header) for row in owner_rows[1:])
            or any(row[1] != safety["gpu_uuid"] for row in owner_rows[1:])
            or any(row[5] for row in owner_rows[1:])
            or owner_rows[1][6] != "prelaunch_owner_clear"
        ):
            raise ValueError("SurfaceRegion owner audit identity differs")
        owner_audit_events = [row[6] for row in owner_rows[1:]]
        allowed_owner_events = {
            "prelaunch_owner_clear",
            "runtime_owner_audit",
            "runtime_owner_audit_host_pid_singleton",
            "postexit_owner_clear",
        }
        if not set(owner_audit_events) <= allowed_owner_events:
            raise ValueError("SurfaceRegion owner audit contains an invalid event")
        singleton_pids = {
            row[3]
            for row in owner_rows[1:]
            if row[6] == "runtime_owner_audit_host_pid_singleton"
        }
        if (
            any(
                row[3] != row[4]
                for row in owner_rows[1:]
                if row[6].startswith("runtime_owner_audit")
            )
            or any(
                not pid.isdigit() or ";" in pid
                for pid in singleton_pids
            )
            or len(singleton_pids) > 1
            or (
                safety.get("owner_pid_namespace_mode")
                == "exclusive-singleton-after-clear-v1"
                and "runtime_owner_audit_host_pid_singleton"
                in owner_audit_events
                and len(singleton_pids) != 1
            )
        ):
            raise ValueError("SurfaceRegion owner audit PID binding differs")
    for key in ("terminal", "sidecar"):
        if receipt[key] is not None:
            _validate_artifact_file_record(
                receipt[key],
                label=f"SurfaceRegion attempt {key}",
            )
    telemetry = receipt["telemetry_interval"]
    journal = receipt["kernel_journal"]
    if (
        not isinstance(telemetry, Mapping)
        or set(telemetry)
        != {"path", "start_line", "end_line", "line_count", "sha256"}
        or not isinstance(journal, Mapping)
        or set(journal)
        != {
            "start_epoch",
            "end_epoch",
            "capture_status",
            "fault_count",
            "file",
        }
        or not isinstance(telemetry["start_line"], int)
        or not isinstance(telemetry["end_line"], int)
        or telemetry["start_line"] < 0
        or telemetry["end_line"] <= telemetry["start_line"]
        or telemetry["line_count"]
        != telemetry["end_line"] - telemetry["start_line"]
        or not isinstance(journal["start_epoch"], int)
        or not isinstance(journal["end_epoch"], int)
        or journal["start_epoch"] <= 0
        or journal["end_epoch"] < journal["start_epoch"]
        or not isinstance(journal["capture_status"], int)
        or isinstance(journal["capture_status"], bool)
        or not isinstance(journal["fault_count"], int)
        or isinstance(journal["fault_count"], bool)
        or journal["fault_count"] < 0
    ):
        raise ValueError("SurfaceRegion attempt intervals differ")
    expected_telemetry = telemetry_interval_record(
        telemetry["path"],
        start_line=int(telemetry["start_line"]),
        end_line=int(telemetry["end_line"]),
    )
    if dict(telemetry) != expected_telemetry:
        raise ValueError("SurfaceRegion attempt telemetry digest differs")
    attempt_contract = manifest.get("attempt_receipt_contract")
    if (
        not isinstance(attempt_contract, Mapping)
        or Path(str(telemetry["path"])).resolve()
        != Path(str(attempt_contract.get("telemetry_path", ""))).resolve()
    ):
        raise ValueError("SurfaceRegion attempt telemetry path differs")
    expected_kernel_path = Path(expected_log).with_name(
        Path(expected_log).name.removesuffix(".log") + ".kernel.log"
    )
    expected_kernel_record = _artifact_file_record(
        expected_kernel_path,
        label="SurfaceRegion attempt kernel journal",
    )
    if journal["file"] != expected_kernel_record:
        raise ValueError("SurfaceRegion attempt kernel journal record differs")
    kernel_encoded, _, _ = _stable_artifact_bytes(
        expected_kernel_path,
        label="SurfaceRegion attempt kernel journal",
    )
    observed_kernel_faults = kernel_fault_lines(kernel_encoded)
    if len(observed_kernel_faults) != journal["fault_count"]:
        raise ValueError("SurfaceRegion attempt kernel fault count differs")
    journal_clear = (
        journal["capture_status"] == 0 and journal["fault_count"] == 0
    )
    rows = _telemetry_rows(
        telemetry["path"],
        start_line=int(telemetry["start_line"]),
        end_line=int(telemetry["end_line"]),
    )
    peer_events = sum(
        str(row.get("event", "")).startswith(
            "peer_activity_interrupt_release_cuda_"
        )
        for row in rows
    )
    status = receipt["command_status"]
    result = receipt["result"]
    gpu1_only = safety.get("peer_gpu") is None
    interrupt_code = int(safety["peer_activity_interrupt_exit_code"])
    postflight = receipt["gpu_release_postflight"]
    postflight_clear = False
    if postflight is not None:
        if (
            not isinstance(postflight, Mapping)
            or set(postflight) != {"capture_status", "report"}
            or not isinstance(postflight["capture_status"], int)
            or isinstance(postflight["capture_status"], bool)
            or postflight["report"] is None
        ):
            raise ValueError("SurfaceRegion GPU release postflight differs")
        expected_postflight_path = Path(expected_log).with_name(
            Path(expected_log).name.removesuffix(".log")
            + ".gpu_release_postflight.json"
        )
        expected_postflight_record = _artifact_file_record(
            expected_postflight_path,
            label="SurfaceRegion GPU release postflight",
        )
        if postflight["report"] != expected_postflight_record:
            raise ValueError("SurfaceRegion GPU release postflight record differs")
        postflight_payload = _json_object(expected_postflight_path)
        required_postflight = {
            "artifact_type",
            "schema_version",
            "status",
            "physical_gpu",
            "expected_uuid",
            "observed_uuid",
            "bus_id",
            "pcie_config_prefix",
            "pcie_responsive",
            "compute_query_succeeded",
            "compute_owner_pids",
        }
        if (
            set(postflight_payload) != required_postflight
            or postflight_payload["artifact_type"]
            != "surface-region-gpu-release-postflight-v1"
            or postflight_payload["schema_version"] != 1
            or postflight_payload["physical_gpu"]
            != int(safety["physical_gpu"])
            or postflight_payload["expected_uuid"] != safety["gpu_uuid"]
        ):
            raise ValueError("SurfaceRegion GPU release postflight identity differs")
        postflight_clear = (
            postflight["capture_status"] == 0
            and postflight_payload["status"] == "gpu_release_verified_clear"
            and postflight_payload["observed_uuid"] == safety["gpu_uuid"]
            and postflight_payload["pcie_responsive"] is True
            and postflight_payload["compute_query_succeeded"] is True
            and postflight_payload["compute_owner_pids"] == []
        )
    if result == "completed":
        valid_result = (
            status == 0
            and peer_events == 0
            and postflight is None
            and journal_clear
        )
    elif result == "peer_activity_interrupted_cuda_released_retry_authorized":
        valid_result = (
            status == interrupt_code
            and peer_events == 1
            and postflight_clear
            and journal_clear
        )
    elif result == "peer_activity_interrupted_cuda_release_unverified_no_retry":
        valid_result = (
            status == interrupt_code
            and peer_events == 1
            and postflight is not None
            and not postflight_clear
            and journal_clear
        )
    elif result == "failed_no_retry":
        valid_result = (
            isinstance(status, int)
            and not isinstance(status, bool)
            and status != 0
            and peer_events == 0
            and postflight is None
            and journal_clear
        )
    elif result == "attempt_evidence_failed_no_retry":
        valid_result = not journal_clear
    else:
        valid_result = False
    if gpu1_only and (
        peer_events != 0
        or str(result).startswith("peer_activity_interrupted_")
        or status == interrupt_code
        or postflight is not None
    ):
        valid_result = False
    if owner_audit_required and result == "completed" and (
        not owner_audit_events
        or owner_audit_events[-1] != "postexit_owner_clear"
    ):
        valid_result = False
    if not valid_result:
        raise ValueError("SurfaceRegion attempt result/telemetry differs")
    return receipt


def audit_attempt_inventory(
    *,
    manifest_path: str | Path,
    attempt_root: str | Path,
    log_root: str | Path,
) -> dict:
    root = Path(attempt_root)
    logs = Path(log_root)
    if root.is_symlink() or logs.is_symlink() or not root.is_dir() or not logs.is_dir():
        raise ValueError("SurfaceRegion attempt/log roots differ")
    manifest = _json_object(manifest_path)
    contract = manifest.get("attempt_receipt_contract")
    if (
        not isinstance(contract, Mapping)
        or Path(str(contract.get("root", ""))).resolve() != root.resolve()
        or Path(str(contract.get("log_root", ""))).resolve() != logs.resolve()
    ):
        raise ValueError("SurfaceRegion attempt inventory contract differs")
    canonical_telemetry = Path(str(contract.get("telemetry_path", ""))).resolve()
    rows: list[dict] = []
    expected_log_names: set[str] = set()
    for stage_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if stage_dir.is_symlink() or not stage_dir.is_dir():
            raise ValueError("SurfaceRegion attempt root contains a non-directory")
        entries = sorted(stage_dir.iterdir(), key=lambda path: path.name)
        owner_audit_required = contract.get("owner_audit_required") is True
        receipts = [
            path
            for path in entries
            if re.fullmatch(r"attempt_[0-9]{6}\.json", path.name)
        ]
        if not receipts:
            raise ValueError("SurfaceRegion stage attempt directory is empty")
        expected_entry_names = {path.name for path in receipts}
        if owner_audit_required:
            expected_entry_names.update(
                path.with_suffix(".owner_audit.csv").name
                for path in receipts
            )
        if {path.name for path in entries} != expected_entry_names:
            raise ValueError("SurfaceRegion stage attempt evidence inventory differs")
        stage_rows: list[dict] = []
        for expected_index, receipt_path in enumerate(receipts, start=1):
            expected_name = f"attempt_{expected_index:06d}.json"
            if receipt_path.name != expected_name:
                raise ValueError("SurfaceRegion attempt receipt sequence has a gap")
            log_name = f"{stage_dir.name}.attempt_{expected_index:06d}.log"
            log_path = logs / log_name
            receipt = validate_attempt_receipt(
                manifest_path=manifest_path,
                receipt_path=receipt_path,
                expected_stage=stage_dir.name,
                expected_index=expected_index,
                expected_log=log_path,
            )
            expected_log_names.add(log_name)
            kernel_name = (
                f"{stage_dir.name}.attempt_{expected_index:06d}.kernel.log"
            )
            expected_log_names.add(kernel_name)
            if receipt["gpu_release_postflight"] is not None:
                expected_log_names.add(
                    f"{stage_dir.name}.attempt_{expected_index:06d}."
                    "gpu_release_postflight.json"
                )
            row = {
                "stage": stage_dir.name,
                "attempt_index": expected_index,
                "result": receipt["result"],
                "command_status": receipt["command_status"],
                "command": receipt["command"],
                "telemetry_interval": receipt["telemetry_interval"],
                "kernel_journal": receipt["kernel_journal"],
                "gpu_release_postflight": receipt["gpu_release_postflight"],
                "owner_audit": receipt.get("owner_audit"),
                "receipt": _artifact_file_record(
                    receipt_path,
                    label="SurfaceRegion attempt receipt",
                ),
                "log": receipt["log"],
            }
            stage_rows.append(row)
            rows.append(row)
        if any(
            row["result"]
            != "peer_activity_interrupted_cuda_released_retry_authorized"
            for row in stage_rows[:-1]
        ) or stage_rows[-1]["result"] != "completed":
            raise ValueError("SurfaceRegion stage attempt sequence is not terminal")
    intervals = [
        [
            int(row["telemetry_interval"]["start_line"]),
            int(row["telemetry_interval"]["end_line"]),
            row,
        ]
        for row in rows
    ]
    intervals.sort(
        key=lambda value: (
            value[0],
            value[1],
            value[2]["stage"],
            value[2]["attempt_index"],
        )
    )
    previous_start = -1
    previous_end = -1
    for start, end, row in intervals:
        if Path(row["telemetry_interval"]["path"]).resolve() != canonical_telemetry:
            raise ValueError("SurfaceRegion attempt telemetry inventory path differs")
        if start <= previous_start or start < previous_end:
            raise ValueError("SurfaceRegion attempt telemetry intervals overlap or repeat")
        previous_start, previous_end = start, end
    log_entries = list(logs.iterdir())
    if any(not path.is_file() and not path.is_symlink() for path in log_entries):
        raise ValueError("SurfaceRegion log root contains a non-file")
    actual_log_names = {path.name for path in log_entries}
    if actual_log_names != expected_log_names:
        raise ValueError("SurfaceRegion attempt log inventory differs")
    payload = {
        "artifact_type": "surface-region-stage-attempt-inventory-v1",
        "schema_version": 1,
        "run_manifest": _artifact_file_record(
            manifest_path,
            label="SurfaceRegion run manifest",
        ),
        "attempt_root": str(root.resolve()),
        "log_root": str(logs.resolve()),
        "attempts": rows,
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def kernel_fault_lines(encoded: bytes | str) -> list[str]:
    if isinstance(encoded, bytes):
        text = encoded.decode("utf-8", errors="replace")
    else:
        text = encoded
    return [line for line in text.splitlines() if KERNEL_FAULT_PATTERN.search(line)]


def capture_kernel_journal(
    *,
    start_epoch: int,
    end_epoch: int,
    output_path: str | Path,
    gpu_bus_id: str | None = None,
) -> dict:
    if start_epoch <= 0 or end_epoch < start_epoch:
        raise ValueError("invalid SurfaceRegion kernel-journal interval")
    try:
        completed = subprocess.run(
            [
                "journalctl",
                "-k",
                "--since",
                f"@{start_epoch}",
                "--until",
                f"@{end_epoch}",
                "--no-pager",
                "-o",
                "short-iso",
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
        status = int(completed.returncode)
        encoded = bytes(completed.stdout)
    except (OSError, subprocess.TimeoutExpired) as error:
        status = 86
        encoded = (f"kernel journal capture failed: {error}\n").encode("utf-8")
    if gpu_bus_id and status == 0:
        normalized = str(gpu_bus_id).strip().lower()
        if re.fullmatch(
            r"(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}(?:\.[0-7])?",
            normalized,
        ) is None:
            raise ValueError("invalid target GPU PCI bus ID")
        without_domain = (
            normalized.split(":", 1)[1]
            if normalized.count(":") == 2
            else normalized
        )
        tokens = {
            normalized,
            normalized.rsplit(".", 1)[0],
            without_domain,
            without_domain.rsplit(".", 1)[0],
        }
        encoded = b"".join(
            line
            for line in encoded.splitlines(keepends=True)
            if any(
                token.encode("ascii") in line.lower()
                for token in tokens
            )
        )
    _atomic_publish_bytes(output_path, encoded)
    faults = kernel_fault_lines(encoded)
    return {
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "capture_status": status,
        "fault_count": len(faults),
        "file": _artifact_file_record(
            output_path,
            label="SurfaceRegion attempt kernel journal",
        ),
    }


def _nvidia_smi(arguments: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 86, ""
    return int(completed.returncode), completed.stdout


def capture_gpu_release_postflight(
    *,
    physical_gpu: int,
    expected_uuid: str,
    output_path: str | Path,
) -> dict:
    """Freeze an independent PCI/UUID/compute-owner release observation."""

    identity_status, identity_output = _nvidia_smi(
        [
            "-i",
            str(physical_gpu),
            "--query-gpu=pci.bus_id,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    identity_lines = [
        line.strip() for line in identity_output.splitlines() if line.strip()
    ]
    bus_id = ""
    observed_uuid = ""
    if identity_status == 0 and len(identity_lines) == 1:
        identity_fields = [
            value.strip() for value in identity_lines[0].split(",")
        ]
        if len(identity_fields) == 2:
            bus_id = re.sub(r"^00000000:", "0000:", identity_fields[0])
            observed_uuid = identity_fields[1]
    pcie_prefix = ""
    if bus_id:
        try:
            with (Path("/sys/bus/pci/devices") / bus_id / "config").open("rb") as handle:
                pcie_prefix = handle.read(16).hex()
        except OSError:
            pcie_prefix = ""
    pcie_responsive = bool(pcie_prefix and set(pcie_prefix.lower()) != {"f"})

    owner_status, owner_output = _nvidia_smi(
        [
            "-i",
            str(physical_gpu),
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    owners: list[int] = []
    owner_parse_ok = owner_status == 0
    if owner_parse_ok:
        for line in owner_output.splitlines():
            if not line.strip():
                continue
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 2 or not fields[1].isdigit():
                owner_parse_ok = False
                owners = []
                break
            if fields[0] == expected_uuid:
                owners.append(int(fields[1]))
    clear = (
        pcie_responsive
        and observed_uuid == expected_uuid
        and owner_parse_ok
        and not owners
    )
    payload = {
        "artifact_type": "surface-region-gpu-release-postflight-v1",
        "schema_version": 1,
        "status": (
            "gpu_release_verified_clear" if clear else "gpu_release_not_clear"
        ),
        "physical_gpu": int(physical_gpu),
        "expected_uuid": str(expected_uuid),
        "observed_uuid": observed_uuid,
        "bus_id": bus_id,
        "pcie_config_prefix": pcie_prefix,
        "pcie_responsive": pcie_responsive,
        "compute_query_succeeded": owner_parse_ok,
        "compute_owner_pids": sorted(owners),
    }
    _atomic_publish_json(output_path, payload)
    return payload


def _kernel_faults(start_epoch: int, end_epoch: int) -> list[str]:
    if start_epoch <= 0 or end_epoch < start_epoch:
        raise ValueError("invalid canary kernel-journal interval")
    completed = subprocess.run(
        [
            "journalctl",
            "-k",
            "--since",
            f"@{start_epoch}",
            "--until",
            f"@{end_epoch}",
            "--no-pager",
            "-o",
            "short-iso",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValueError("cannot audit the kernel journal for canary Xid/PCIe faults")
    return kernel_fault_lines(completed.stdout)


def _stable_torch_load(path: Path) -> tuple[dict, str]:
    source = _regular_file(path, label="canary cache")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        first_digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            first_digest.update(block)
        handle.seek(0)
        try:
            value = torch.load(handle, map_location="cpu", weights_only=True)
        except TypeError as error:
            raise RuntimeError("canary audit requires weights_only=True") from error
        handle.seek(0)
        second_digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            second_digest.update(block)
        after = os.fstat(handle.fileno())
    path_after = os.stat(source, follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    identity_path = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        identity_before != identity_after
        or identity_before != identity_path
        or first_digest.hexdigest() != second_digest.hexdigest()
    ):
        raise ValueError("canary cache changed while being audited")
    if not isinstance(value, dict):
        raise ValueError("canary cache is not a mapping")
    return value, first_digest.hexdigest()


def _audit_canary_cache(cache_path: Path, manifest: Mapping) -> dict:
    payload, cache_digest = _stable_torch_load(cache_path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("canary cache lacks metadata")
    cache_contract = manifest["cache_contract"]
    split_lines = [
        line.strip()
        for line in Path(str(manifest["train_split"])).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    expected_scenes = split_lines[0 :: int(cache_contract["train_shards"])][
        :100
    ]
    records = metadata.get("region_records")
    if not isinstance(records, list):
        raise ValueError("canary cache lacks region records")
    scene_names = metadata.get("scene_names")
    scene_counts = metadata.get("scene_region_counts")
    regions_per_scene = int(cache_contract["regions_per_scene"])
    row_count = len(expected_scenes) * regions_per_scene
    expected_shapes = {
        "radio_features": (row_count, 256, 1280),
        "geometry": (row_count, 256, 14),
        "token_mask": (row_count, 256),
        "reliability": (row_count, 256, 1),
        "official_summary_tokens": (row_count, 3, 1280),
        "official_crop_summaries": (row_count, 3, 1536),
        "teacher_mask": (row_count, 3),
        "anchor_index": (row_count,),
    }
    if (
        metadata.get("schema_version") != 3
        or metadata.get("split_role") != "train"
        or metadata.get("split_file_sha256") != manifest["train_split_sha256"]
        or metadata.get("builder_script_sha256")
        != manifest["runtime_closure"]["repository_sources"]["files"][
            "radio_gs/scripts/build_scannet_surface_region_cache.py"
        ]
        or metadata.get("radio_checkpoint_sha256")
        != manifest["radio_checkpoint_sha256"]
        or scene_names != expected_scenes
        or scene_counts != {scene: regions_per_scene for scene in expected_scenes}
        or metadata.get("failed_scenes") != {}
        or metadata.get("complete_scene_regions") is not True
        or metadata.get("physical_space_disjoint") is not True
        or metadata.get("uses_benchmark_scenes") is not False
        or metadata.get("teacher_target_source") != "fresh_official_runtime"
        or metadata.get("teacher_replay_cache") != {}
        or int(metadata.get("teacher_regions_saturated", -1)) != 0
        or float(metadata.get("execution_radio_thermal_pacing_seconds_per_image", -1))
        != float(cache_contract["radio_thermal_pacing_seconds_per_image"])
        or len(records) != row_count
        or len({str(record.get("region_id", "")) for record in records})
        != row_count
    ):
        raise ValueError("canary cache semantic contract differs")
    for key, shape in expected_shapes.items():
        value = payload.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"canary cache tensor shape differs: {key}")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"canary cache contains non-finite values: {key}")
    if not bool(payload["token_mask"].any(dim=1).all()):
        raise ValueError("canary cache has an empty token row")
    if not bool(payload["teacher_mask"].any(dim=1).all()):
        raise ValueError("canary cache has an empty teacher row")
    anchors = payload["anchor_index"].long()
    if not bool(((anchors >= 0) & (anchors < 256)).all()):
        raise ValueError("canary cache has an invalid anchor")
    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
    sidecar = _json_object(sidecar_path)
    expected_sidecar = {
        "output": str(cache_path.resolve()),
        "regions": row_count,
        "scenes": len(expected_scenes),
        "failed_scenes": {},
        "split_role": "train",
        "split_file_sha256": manifest["train_split_sha256"],
        "teacher_target_source": "fresh_official_runtime",
        "teacher_replay_cache": {},
    }
    if sidecar != expected_sidecar:
        raise ValueError("canary cache sidecar differs")
    return {
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_digest,
        "cache_size": cache_path.stat().st_size,
        "sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": sha256_file(sidecar_path),
        "sidecar_size": sidecar_path.stat().st_size,
        "scenes": len(expected_scenes),
        "regions": row_count,
        "failed_scenes": {},
    }


def audit_canary(
    *,
    manifest_path: str | Path,
    telemetry_path: str | Path,
    terminal_path: str | Path,
    report_path: str | Path,
    start_line: int | None,
    end_line: int | None,
    start_epoch: int | None,
    end_epoch: int | None,
) -> dict:
    report_source = Path(report_path)
    if report_source.is_file():
        existing = _json_object(report_source)
        interval = existing.get("telemetry_interval", {})
        journal = existing.get("kernel_journal_interval", {})
        start_line = int(interval.get("start_line", -1))
        end_line = int(interval.get("end_line", -1))
        start_epoch = int(journal.get("start_epoch", -1))
        end_epoch = int(journal.get("end_epoch", -1))
    requested_start_line = start_line
    requested_end_line = end_line
    manifest, closure = verify_runtime_closure(
        manifest_path,
        full_checkpoint=True,
    )
    attempt_evidence = None
    if "attempt_receipt_contract" in manifest:
        attempt_contract = manifest["attempt_receipt_contract"]
        inventory = audit_attempt_inventory(
            manifest_path=manifest_path,
            attempt_root=attempt_contract["root"],
            log_root=attempt_contract["log_root"],
        )
        expected_stage = manifest["canary_contract"].get(
            "stage", "cache_control_c256_geometric_train_0"
        )
        relevant = [
            row
            for row in inventory["attempts"]
            if row["stage"] == expected_stage
        ]
        if not relevant or relevant[-1]["result"] != "completed":
            raise ValueError("canary lacks a completed stage attempt receipt")
        if any(
            Path(row["telemetry_interval"]["path"]).resolve()
            != Path(telemetry_path).resolve()
            for row in relevant
        ):
            raise ValueError("canary attempt telemetry path differs")
        start_line = min(
            int(row["telemetry_interval"]["start_line"]) for row in relevant
        )
        end_line = max(
            int(row["telemetry_interval"]["end_line"]) for row in relevant
        )
        if requested_start_line is not None and int(requested_start_line) >= 0:
            start_line = min(int(start_line), int(requested_start_line))
        if requested_end_line is not None and int(requested_end_line) >= 0:
            end_line = max(int(end_line), int(requested_end_line))
        start_epoch = min(
            int(row["kernel_journal"]["start_epoch"]) for row in relevant
        )
        end_epoch = max(
            int(row["kernel_journal"]["end_epoch"]) for row in relevant
        )
        rows = _telemetry_rows(
            telemetry_path,
            start_line=int(start_line),
            end_line=int(end_line),
        )
        authorized = [
            row
            for row in relevant
            if row["result"]
            == "peer_activity_interrupted_cuda_released_retry_authorized"
        ]
        peer_interrupt_count = sum(
            str(row.get("event", "")).startswith(
                "peer_activity_interrupt_release_cuda_"
            )
            for row in rows
        )
        if len(authorized) != peer_interrupt_count:
            raise ValueError(
                "canary peer-release telemetry/attempt receipts are not paired"
            )
        attempt_evidence = {
            "stage": expected_stage,
            "attempts": relevant,
            "authorized_peer_interrupts": len(authorized),
            "all_restart_attempts_audited": True,
        }
        attempt_evidence["digest"] = canonical_json_sha256(
            attempt_evidence
        )
        kernel_faults: list[str] = []
    else:
        if None in (start_line, end_line, start_epoch, end_epoch):
            raise ValueError("new canary audit requires telemetry and journal bounds")
        rows = _telemetry_rows(
            telemetry_path,
            start_line=int(start_line),
            end_line=int(end_line),
        )
        kernel_faults = _kernel_faults(int(start_epoch), int(end_epoch))
    thermal = summarize_canary_telemetry(
        rows,
        expected_gpu=int(manifest["thermal_safety_contract"]["physical_gpu"]),
        maximum_temperature_c=int(
            manifest["canary_contract"]["maximum_temperature_c"]
        ),
        peer_gpu=manifest["thermal_safety_contract"]["peer_gpu"],
    )
    if kernel_faults:
        raise ValueError(f"canary kernel journal contains GPU faults: {kernel_faults}")
    cache = _audit_canary_cache(Path(terminal_path).resolve(), manifest)
    report = {
        "schema_version": 1,
        "artifact_type": "surface_region_full_shard_thermal_canary",
        "status": "canary_passed_resume_authorized",
        "run_manifest": str(Path(manifest_path).resolve()),
        "run_manifest_sha256": sha256_file(manifest_path),
        "runtime_closure_digest": closure["digest"],
        "telemetry": str(Path(telemetry_path).resolve()),
        "telemetry_interval": {
            "start_line": int(start_line),
            "end_line": int(end_line),
        },
        "kernel_journal_interval": {
            "start_epoch": int(start_epoch),
            "end_epoch": int(end_epoch),
            "xid_or_pcie_fault_count": 0,
        },
        "thermal_summary": thermal,
        "cache_terminal": cache,
    }
    if attempt_evidence is not None:
        report["attempt_receipts"] = attempt_evidence
    _atomic_publish_json(report_source, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-closure")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--phase", required=True)
    verify.add_argument("--full-checkpoint", action="store_true")
    verify.add_argument("--report", type=Path)
    verify.add_argument("--attempt-root", type=Path)
    verify.add_argument("--log-root", type=Path)
    attempt = subparsers.add_parser("verify-attempt")
    attempt.add_argument("--manifest", type=Path, required=True)
    attempt.add_argument("--receipt", type=Path, required=True)
    attempt.add_argument("--stage", required=True)
    attempt.add_argument("--index", type=int, required=True)
    attempt.add_argument("--log", type=Path, required=True)
    attempt.add_argument("--command-arg", action="append", default=[])
    attempt.add_argument("--allowed-result", action="append", default=[])
    journal = subparsers.add_parser("capture-kernel-journal")
    journal.add_argument("--start-epoch", type=int, required=True)
    journal.add_argument("--end-epoch", type=int, required=True)
    journal.add_argument("--output", type=Path, required=True)
    journal.add_argument("--gpu-bus-id")
    postflight = subparsers.add_parser("capture-gpu-release-postflight")
    postflight.add_argument("--gpu", type=int, required=True)
    postflight.add_argument("--expected-uuid", required=True)
    postflight.add_argument("--output", type=Path, required=True)
    canary = subparsers.add_parser("audit-canary")
    canary.add_argument("--manifest", type=Path, required=True)
    canary.add_argument("--telemetry", type=Path, required=True)
    canary.add_argument("--terminal", type=Path, required=True)
    canary.add_argument("--report", type=Path, required=True)
    canary.add_argument("--start-line", type=int)
    canary.add_argument("--end-line", type=int)
    canary.add_argument("--start-epoch", type=int)
    canary.add_argument("--end-epoch", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-closure":
        if args.report is None:
            _, closure = verify_runtime_closure(
                args.manifest,
                full_checkpoint=bool(args.full_checkpoint),
            )
            payload = {
                "status": "runtime_closure_verified",
                "phase": args.phase,
                "runtime_closure_digest": closure["digest"],
            }
        else:
            payload = write_closure_report(
                args.manifest,
                args.report,
                phase=args.phase,
                full_checkpoint=bool(args.full_checkpoint),
                attempt_root=args.attempt_root,
                log_root=args.log_root,
            )
    elif args.command == "verify-attempt":
        payload = validate_attempt_receipt(
            manifest_path=args.manifest,
            receipt_path=args.receipt,
            expected_stage=args.stage,
            expected_index=args.index,
            expected_log=args.log,
            expected_command=args.command_arg,
        )
        if args.allowed_result and payload["result"] not in args.allowed_result:
            raise ValueError("existing SurfaceRegion attempt is not retryable")
    elif args.command == "capture-kernel-journal":
        payload = capture_kernel_journal(
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            output_path=args.output,
            gpu_bus_id=args.gpu_bus_id,
        )
        if payload["capture_status"] != 0 or payload["fault_count"] != 0:
            print(json.dumps(payload, indent=2, allow_nan=False))
            raise SystemExit(86)
    elif args.command == "capture-gpu-release-postflight":
        payload = capture_gpu_release_postflight(
            physical_gpu=args.gpu,
            expected_uuid=args.expected_uuid,
            output_path=args.output,
        )
        if payload["status"] != "gpu_release_verified_clear":
            print(json.dumps(payload, indent=2, allow_nan=False))
            raise SystemExit(86)
    else:
        payload = audit_canary(
            manifest_path=args.manifest,
            telemetry_path=args.telemetry,
            terminal_path=args.terminal,
            report_path=args.report,
            start_line=args.start_line,
            end_line=args.end_line,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
        )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
