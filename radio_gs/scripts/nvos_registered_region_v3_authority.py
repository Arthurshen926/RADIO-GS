#!/usr/bin/env python3
"""CPU-only source-closure authority for the frozen NVOS-v3 runner.

The GPU runner is intentionally allowed to execute from a read-only source
snapshot without a ``.git`` directory.  This helper binds that exact source
tree into the run manifest and detects any source or import-root drift before
and after every GPU stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
SOURCE_GLOBS = ("*.py", "*.sh")
RUNTIME_PACKAGES = (
    "gsplat",
    "huggingface-hub",
    "numba",
    "numpy",
    "opencv-contrib-python",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "plyfile",
    "PyYAML",
    "safetensors",
    "scipy",
    "timm",
    "tqdm",
    "torch",
    "torchvision",
)
RUNTIME_MODULE_DISTRIBUTIONS = {
    "cv2": "opencv-python",
    "gsplat": "gsplat",
    "huggingface_hub": "huggingface-hub",
    "numba": "numba",
    "numpy": "numpy",
    "PIL": "Pillow",
    "plyfile": "plyfile",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
}
RUNTIME_ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "GPU_OWNER_PID_NAMESPACE_MODE",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "MAX_JOBS",
    "NO_FAST_MATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NUMBA_CACHE_DIR",
    "PYTHONDONTWRITEBYTECODE",
    "RADIO_GS_DRIVER_LIBRARY",
    "RADIO_GS_LD_LIBRARY_PATH",
    "RADIO_GS_PYTHON",
    "RADIO_GS_REPO_ROOT",
    "RADIO_GS_SAM3_SOURCE",
    "RADIO_GS_SITE_PACKAGES",
    "TORCH_EXTENSIONS_DIR",
)
REQUIRED_NON_PACKAGE_SOURCES = (
    "paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml",
)
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
GPU_OWNER_PID_NAMESPACE_MODE = "exclusive-singleton-after-clear-v1"
OWNER_AUDIT_COLUMNS = (
    "timestamp",
    "gpu_uuid",
    "child_pgid",
    "owner_pids",
    "child_owner_pids",
    "foreign_owner_pids",
    "event",
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
CUDA_ATTESTATION_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "scene",
    "observed_epoch",
    "hostname",
    "environment",
    "expected_gpu",
    "torch_cuda",
    "process_namespace_pids",
    "nvidia_inventory_row",
    "nvidia_preallocation_owner_rows",
    "nvidia_compute_owner_rows",
    "owner_pid_binding",
    "attestation_mechanism",
}
CUDA_ATTESTATION_MECHANISM = (
    "torch_cuda0_live_allocation_plus_nvidia_smi_exclusive_owner_"
    "with_container_host_pid_namespace_binding_plus_uuid_pci_v2"
)
POSTCHECK_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "scene",
    "observed_epoch",
    "run_manifest",
    "result",
    "runtime_closure_sha256",
    "gpu_identity",
    "nvidia_inventory_row",
    "proc_driver_identity",
    "pcie_config_prefix_hex",
    "compute_owners",
    "global_lock",
    "kernel_singleton",
}
RECEIPT_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "scene",
    "run_manifest",
    "result",
    "command",
    "telemetry",
    "owner_audit",
    "cuda_attestation",
    "postcheck",
    "output_identity",
    "artifact_bindings",
    "thermal_safety_contract",
}
OUTPUT_IDENTITY_FIELDS = {
    "logical_main_root",
    "resolved_main_root",
    "logical_output_root",
    "resolved_output_root",
    "main_target_device",
    "main_target_inode",
    "output_device",
    "output_inode",
}
ARTIFACT_BINDING_FIELDS = {
    "relative_path",
    "logical_path",
    "resolved_path",
    "parent_device",
    "parent_inode",
}
SCENE_ARTIFACT_NAMES = {
    "result",
    "telemetry",
    "owner_audit",
    "cuda_attestation",
    "command",
    "postcheck",
    "receipt",
    "evaluator_log",
}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_nested_symlinks(path: Path, *, anchor: Path, label: str) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    root = Path(os.path.abspath(os.fspath(anchor)))
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escaped its authority root: {target}") from error
    current = root
    for component in relative.parts:
        current = current / component
        try:
            info = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} has a nested symlink component: {current}")


def _reject_tree_symlinks(root: Path, *, label: str) -> None:
    """Reject symlink entries before source selection can silently omit them."""

    root_info = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"{label} root is not a real directory: {root}")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        for name in sorted((*directories, *files)):
            candidate = Path(current) / name
            info = os.stat(candidate, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"{label} contains a symlink entry: {candidate}")


def _stable_file_record(path: str | Path, *, root: Path) -> dict[str, object]:
    source = Path(os.path.abspath(os.fspath(path)))
    _reject_nested_symlinks(source, anchor=root, label="NVOS closure source")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("NVOS source closure requires O_NOFOLLOW")
    descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(source, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise ValueError(f"NVOS closure source is not a stable regular file: {source}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
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
            raise ValueError(f"NVOS closure source changed while hashing: {source}")
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise ValueError(f"NVOS closure source escaped snapshot root: {source}") from error
        return {
            "path": str(source),
            "relative_path": str(relative),
            "sha256": digest.hexdigest(),
            "bytes": size,
        }
    finally:
        os.close(descriptor)


def _source_paths(repo_root: Path) -> tuple[Path, ...]:
    package = repo_root / "radio_gs"
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"NVOS source snapshot lacks a real radio_gs directory: {package}")
    _reject_nested_symlinks(package, anchor=repo_root, label="NVOS package root")
    _reject_tree_symlinks(package, label="NVOS package source tree")
    paths: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        paths.update(package.rglob(pattern))
    for relative in REQUIRED_NON_PACKAGE_SOURCES:
        candidate = repo_root / relative
        _reject_nested_symlinks(
            candidate,
            anchor=repo_root,
            label="NVOS required non-package source",
        )
        paths.add(candidate)
    return tuple(sorted(paths))


def source_snapshot_permissions(repo_root: str | Path) -> dict[str, object]:
    root = Path(os.path.abspath(os.fspath(repo_root)))
    selected_sources = _source_paths(root)
    entries: set[Path] = {root, root / "radio_gs"}
    for current, directories, files in os.walk(
        root / "radio_gs",
        topdown=True,
        followlinks=False,
    ):
        entries.add(Path(current))
        entries.update(Path(current) / name for name in directories)
        entries.update(Path(current) / name for name in files)
    for source in selected_sources:
        entries.add(source)
        current = source.parent
        while current != root:
            entries.add(current)
            current = current.parent
    modes: dict[str, str] = {}
    writable: list[str] = []
    for entry in sorted(entries):
        info = os.stat(entry, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"NVOS source snapshot contains a symlink: {entry}")
        relative = "." if entry == root else str(entry.relative_to(root))
        permission = stat.S_IMODE(info.st_mode)
        modes[relative] = f"{permission:04o}"
        if permission & 0o222:
            writable.append(relative)
    payload: dict[str, object] = {
        "snapshot_root": str(root),
        "selection": (
            "snapshot-root+radio_gs-all-descendants+required-source-ancestors"
        ),
        "entry_modes": modes,
        "writable_entries": writable,
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def verify_readonly_source_snapshot(repo_root: str | Path) -> dict[str, object]:
    record = source_snapshot_permissions(repo_root)
    if Path(str(record["snapshot_root"])) == Path("/root/RADIO-GS"):
        raise ValueError("NVOS GPU launch refuses the mutable live worktree")
    if record["writable_entries"]:
        raise ValueError(
            "NVOS GPU launch requires a read-only source snapshot; writable entries: "
            f"{record['writable_entries']}"
        )
    return {
        "status": "readonly_non_live_source_snapshot_verified",
        "source_permissions": record,
    }


def output_identity(
    logical_main_root: str | Path,
    logical_output_root: str | Path,
) -> dict[str, object]:
    main = Path(os.path.abspath(os.fspath(logical_main_root)))
    output = Path(os.path.abspath(os.fspath(logical_output_root)))
    if str(main) != "/root/RADIO-GS/output":
        raise ValueError("NVOS main output authority must be /root/RADIO-GS/output")
    try:
        output.relative_to(main)
    except ValueError as error:
        raise ValueError("NVOS output escaped the logical main output root") from error
    if output == main:
        raise ValueError("NVOS output cannot equal the main output root")
    main_info = os.stat(main)
    if not stat.S_ISDIR(main_info.st_mode):
        raise ValueError("NVOS main output target is not a directory")
    _reject_nested_symlinks(output, anchor=main, label="NVOS output")
    resolved_main = main.resolve(strict=True)
    resolved_output = output.resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_main)
    except ValueError as error:
        raise ValueError("NVOS output escaped the resolved main output target") from error
    record: dict[str, object] = {
        "logical_main_root": str(main),
        "resolved_main_root": str(resolved_main),
        "logical_output_root": str(output),
        "resolved_output_root": str(resolved_output),
        "main_target_device": int(main_info.st_dev),
        "main_target_inode": int(main_info.st_ino),
    }
    if output.exists():
        output_info = os.stat(output, follow_symlinks=False)
        if not stat.S_ISDIR(output_info.st_mode):
            raise ValueError("NVOS output root is not a real directory")
        record.update(
            {
                "output_device": int(output_info.st_dev),
                "output_inode": int(output_info.st_ino),
            }
        )
    else:
        record.update({"output_device": None, "output_inode": None})
    return record


def _validated_output_identity(record: object) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != OUTPUT_IDENTITY_FIELDS:
        raise ValueError("NVOS output identity schema differs")
    expected = dict(record)
    if not all(
        isinstance(expected.get(key), int)
        and not isinstance(expected.get(key), bool)
        and int(expected[key]) > 0
        for key in (
            "main_target_device",
            "main_target_inode",
            "output_device",
            "output_inode",
        )
    ):
        raise ValueError("NVOS output identity lacks frozen directory identities")
    current = output_identity(
        str(expected["logical_main_root"]),
        str(expected["logical_output_root"]),
    )
    if current != expected:
        raise ValueError("NVOS output root identity changed")
    return expected


def verify_output_tree(record: object) -> dict[str, object]:
    identity = _validated_output_identity(record)
    resolved_root = Path(str(identity["resolved_output_root"]))
    _reject_tree_symlinks(resolved_root, label="NVOS output tree")
    return {
        "status": "output_tree_real_directory_only",
        "output_identity": identity,
    }


def _artifact_binding(
    path: str | Path,
    *,
    output_identity_record: object,
    label: str,
) -> dict[str, object]:
    identity = _validated_output_identity(output_identity_record)
    raw = Path(os.path.abspath(os.fspath(path)))
    logical_root = Path(str(identity["logical_output_root"]))
    resolved_root = Path(str(identity["resolved_output_root"]))
    try:
        relative = raw.relative_to(logical_root)
    except ValueError:
        try:
            relative = raw.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"{label} escaped the frozen NVOS output root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} is not a file below the NVOS output root")
    logical_path = logical_root / relative
    resolved_path = resolved_root / relative
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("NVOS output binding requires O_NOFOLLOW and O_DIRECTORY")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(resolved_root, flags)
    try:
        root_info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or int(root_info.st_dev) != int(identity["output_device"])
            or int(root_info.st_ino) != int(identity["output_inode"])
        ):
            raise ValueError("NVOS output root descriptor identity changed")
        for component in relative.parts[:-1]:
            child_descriptor = os.open(component, flags, dir_fd=descriptor)
            child_info = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child_descriptor)
                raise ValueError(f"{label} parent is not a real directory")
            os.close(descriptor)
            descriptor = child_descriptor
        parent_info = os.fstat(descriptor)
        try:
            target_info = os.stat(
                relative.parts[-1],
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            target_info = None
        if target_info is not None and not stat.S_ISREG(target_info.st_mode):
            raise ValueError(f"{label} target is not a real regular file")
        return {
            "relative_path": relative.as_posix(),
            "logical_path": str(logical_path),
            "resolved_path": str(resolved_path),
            "parent_device": int(parent_info.st_dev),
            "parent_inode": int(parent_info.st_ino),
        }
    finally:
        os.close(descriptor)


def _validate_artifact_binding(
    record: object,
    *,
    output_identity_record: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping) or set(record) != ARTIFACT_BINDING_FIELDS:
        raise ValueError(f"{label} binding schema differs")
    current = _artifact_binding(
        str(record.get("logical_path", "")),
        output_identity_record=output_identity_record,
        label=label,
    )
    if current != dict(record):
        raise ValueError(f"{label} binding changed")
    return current


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _module_distribution_instances(
    module_name: str,
    declared_distribution: str,
) -> list[dict[str, object]]:
    declared = _normalized_distribution_name(declared_distribution)
    instances: list[dict[str, object]] = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", ""))
        top_level = {
            line.strip()
            for line in (distribution.read_text("top_level.txt") or "").splitlines()
            if line.strip()
        }
        if module_name not in top_level and _normalized_distribution_name(name) != declared:
            continue
        raw_metadata_root = getattr(distribution, "_path", None)
        metadata_root = (
            Path(raw_metadata_root).resolve(strict=True)
            if raw_metadata_root is not None
            else None
        )
        metadata_file = (
            metadata_root / "METADATA" if metadata_root is not None else None
        )
        instances.append(
            {
                "name": name,
                "version": str(distribution.version),
                "metadata_root": (
                    str(metadata_root) if metadata_root is not None else None
                ),
                "metadata": (
                    _stable_file_record(metadata_file, root=metadata_root)
                    if metadata_file is not None and metadata_file.is_file()
                    else None
                ),
            }
        )
    return sorted(
        instances,
        key=lambda item: (
            _normalized_distribution_name(str(item["name"])),
            str(item["version"]),
            str(item["metadata_root"]),
        ),
    )


def _module_extension_records(
    module_name: str,
    module: ModuleType,
    origin: Path,
) -> dict[str, dict[str, object]]:
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)

    def is_extension(path: Path) -> bool:
        return any(path.name.endswith(suffix) for suffix in suffixes)

    candidates: set[Path] = set()
    for path in origin.parent.iterdir():
        if path.is_symlink() and is_extension(path):
            raise ValueError(f"runtime extension binary is a symlink: {path}")
        if path.is_file() and is_extension(path):
            candidates.add(path.resolve(strict=True))
    related_modules = [
        value
        for name, value in sys.modules.items()
        if isinstance(value, ModuleType)
        and (name == module_name or name.startswith(f"{module_name}."))
    ]
    related_modules.extend(
        value for value in vars(module).values() if isinstance(value, ModuleType)
    )
    for related in related_modules:
        raw_related_origin = getattr(related, "__file__", None)
        if not isinstance(raw_related_origin, str) or not raw_related_origin:
            continue
        raw_related_path = Path(raw_related_origin)
        if not raw_related_path.is_absolute() or not raw_related_path.exists():
            continue
        if raw_related_path.is_symlink():
            raise ValueError(
                f"loaded runtime extension origin is a symlink: {raw_related_path}"
            )
        related_origin = raw_related_path.resolve(strict=True)
        if related_origin.is_file() and is_extension(related_origin):
            candidates.add(related_origin)
    return {
        str(path): _stable_file_record(path, root=path.parent)
        for path in sorted(candidates)
    }


def _runtime_module_records() -> dict[str, dict[str, object]]:
    importlib.import_module("radio_gs.scripts.eval_nvos_gaussian_first")
    records: dict[str, dict[str, object]] = {}
    for module_name, distribution in RUNTIME_MODULE_DISTRIBUTIONS.items():
        module = importlib.import_module(module_name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            raise ValueError(f"NVOS runtime module lacks an origin: {module_name}")
        origin = Path(raw_origin).resolve(strict=True)
        origin_record = _stable_file_record(origin, root=origin.parent)
        reported_version = getattr(module, "__version__", None)
        distribution_version = importlib.metadata.version(distribution)
        records[module_name] = {
            "distribution": distribution,
            "distribution_version": distribution_version,
            "version": (
                str(reported_version)
                if reported_version is not None
                else distribution_version
            ),
            "module_reported_version": (
                str(reported_version) if reported_version is not None else None
            ),
            "distribution_instances": _module_distribution_instances(
                module_name,
                distribution,
            ),
            "origin": origin_record,
            "extension_binaries": _module_extension_records(
                module_name,
                module,
                origin,
            ),
        }
    return records


def _gsplat_runtime_record() -> dict[str, object]:
    import gsplat
    import torch

    package_root = Path(gsplat.__file__).resolve(strict=True).parent
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
    sources = {
        str(path.relative_to(package_root)): _stable_file_record(
            path, root=package_root
        )
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
        and (path.suffix in suffixes or path.name == "CMakeLists.txt")
    }
    if not sources or "cuda/_backend.py" not in sources:
        raise ValueError("gsplat source/runtime tree is incomplete")
    cuda_version = str(torch.version.cuda or "")
    if not re.fullmatch(r"[0-9]+[.][0-9]+", cuda_version):
        raise ValueError("torch CUDA build version is unavailable")
    explicit_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if explicit_root:
        build_root = Path(explicit_root).resolve(strict=True)
    else:
        tag = (
            f"py{sys.version_info.major}{sys.version_info.minor}_"
            f"cu{cuda_version.replace('.', '')}"
        )
        build_root = (Path.home() / ".cache/torch_extensions" / tag).resolve(
            strict=True
        )
    binary = build_root / "gsplat_cuda/gsplat_cuda.so"
    if not binary.is_file() or binary.is_symlink():
        raise ValueError(f"frozen gsplat CUDA extension is missing: {binary}")
    source_payload: dict[str, object] = {
        "root": str(package_root),
        "files": sources,
    }
    source_payload["digest"] = canonical_json_sha256(source_payload)
    return {
        "distribution_version": importlib.metadata.version("gsplat"),
        "source_tree": source_payload,
        "torch_extensions_dir_environment": explicit_root,
        "resolved_build_root": str(build_root),
        "cuda_extension": _stable_file_record(binary, root=build_root),
        "selection": (
            "gsplat/**/*.{py,cu,cuh,cpp,h,hpp}+CMakeLists.txt+"
            "resolved_pyXY_cuZZZ/gsplat_cuda/gsplat_cuda.so"
        ),
    }


def build_runtime_closure(repo_root: str | Path) -> dict[str, object]:
    root = Path(os.path.abspath(os.fspath(repo_root)))
    root_info = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"NVOS source snapshot root is not a real directory: {root}")
    environment_root = os.environ.get("RADIO_GS_REPO_ROOT")
    if environment_root is not None and Path(environment_root).resolve() != root:
        raise ValueError("NVOS Python import root escaped the source snapshot")
    selected_paths = _source_paths(root)
    records = {
        str(path.relative_to(root)): _stable_file_record(path, root=root)
        for path in selected_paths
    }
    if _source_paths(root) != selected_paths:
        raise ValueError("NVOS source snapshot tree changed while being hashed")
    required = {
        "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh",
        "radio_gs/scripts/nvos_registered_region_v3_authority.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "radio_gs/scripts/run_repo_python.sh",
        "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py",
        "radio_gs/scripts/aggregate_registered_prompt_closeout.py",
        *REQUIRED_NON_PACKAGE_SOURCES,
    }
    missing = sorted(required - set(records))
    if missing:
        raise ValueError(f"NVOS source snapshot closure is incomplete: {missing}")
    source_payload: dict[str, object] = {
        "selection": ["radio_gs/**/*.py", "radio_gs/**/*.sh", *REQUIRED_NON_PACKAGE_SOURCES],
        "files": records,
    }
    source_payload["digest"] = canonical_json_sha256(source_payload)
    executable = Path(sys.executable).resolve()
    executable_record = _stable_file_record(executable, root=executable.parent)
    packages: dict[str, str] = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_root": str(root),
        "repository_import_root": str(root),
        "repository_sources": source_payload,
        "source_snapshot_permissions": source_snapshot_permissions(root),
        "runtime": {
            "python_executable": executable_record,
            "python_version": sys.version,
            "python_prefix": str(Path(sys.prefix).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
            "loaded_modules": _runtime_module_records(),
            "gsplat_runtime": _gsplat_runtime_record(),
            "environment": {
                key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS
            },
        },
    }
    payload["digest"] = canonical_json_sha256(payload)
    return payload


def _stable_json(path: str | Path) -> tuple[dict, str]:
    source = Path(os.path.abspath(os.fspath(path)))
    record = _stable_file_record(source, root=source.parent)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    if (
        len(encoded) != int(record["bytes"])
        or hashlib.sha256(encoded).hexdigest() != record["sha256"]
    ):
        raise ValueError("NVOS run manifest changed while being read")
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NVOS run manifest is not a JSON object")
    return value, str(record["sha256"])


def verify_manifest_closure(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, object]:
    manifest, manifest_sha256 = _stable_json(manifest_path)
    expected = manifest.get("runtime_closure")
    if not isinstance(expected, Mapping):
        raise ValueError("NVOS run manifest lacks runtime_closure")
    current = build_runtime_closure(repo_root)
    if current != dict(expected):
        expected_sources = expected.get("repository_sources")
        current_sources = current.get("repository_sources")
        expected_files = (
            expected_sources.get("files", {})
            if isinstance(expected_sources, Mapping)
            else {}
        )
        current_files = (
            current_sources.get("files", {})
            if isinstance(current_sources, Mapping)
            else {}
        )
        changed = sorted(
            key
            for key in set(expected_files) | set(current_files)
            if expected_files.get(key) != current_files.get(key)
        )
        suffix = f"; changed source files: {changed}" if changed else ""
        raise ValueError(f"NVOS runtime closure changed{suffix}")
    if (
        manifest.get("source_snapshot_root") != current["source_snapshot_root"]
        or manifest.get("source_snapshot_import_root")
        != current["repository_import_root"]
        or manifest.get("source_snapshot_tree_sha256")
        != current["repository_sources"]["digest"]
    ):
        raise ValueError("NVOS source-snapshot binding differs")
    snapshot_authority = manifest.get("source_snapshot_permissions")
    if not (
        isinstance(snapshot_authority, Mapping)
        and snapshot_authority.get("status")
        == "readonly_non_live_source_snapshot_verified"
        and snapshot_authority.get("source_permissions")
        == current["source_snapshot_permissions"]
        and not current["source_snapshot_permissions"]["writable_entries"]
        and Path(str(current["source_snapshot_root"])) != Path("/root/RADIO-GS")
    ):
        raise ValueError("NVOS read-only source snapshot authority differs")
    runner = current["repository_sources"]["files"].get(
        "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh"
    )
    if not isinstance(runner, Mapping) or manifest.get("runner_sha256") != runner.get(
        "sha256"
    ):
        raise ValueError("NVOS runner hash differs from runtime closure")
    expected_output_identity = manifest.get("output_identity")
    verify_output_tree(expected_output_identity)
    return {
        "status": "runtime_closure_verified",
        "manifest_sha256": manifest_sha256,
        "runtime_closure_sha256": current["digest"],
    }


def _bus_suffix(value: str) -> str:
    parts = str(value).strip().lower().split(":")
    return ":".join(parts[-2:])


def _nvidia_query(arguments: Sequence[str]) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(completed.stdout))
        if row
    ]


def _namespace_process_ids() -> list[int]:
    values = {os.getpid()}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("NSpid:"):
            values.update(int(value) for value in line.split()[1:])
            break
    return sorted(values)


def write_cuda_child_attestation(
    *,
    output: str | Path,
    scene: str,
    expected_uuid: str,
    expected_bus_id: str,
) -> dict[str, object]:
    """Initialize torch cuda:0 and bind its live process owner to GPU1."""

    import torch

    environment = {
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "GPU_OWNER_PID_NAMESPACE_MODE": os.environ.get(
            "GPU_OWNER_PID_NAMESPACE_MODE"
        ),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    if environment != {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": expected_uuid,
        "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
        "NVIDIA_VISIBLE_DEVICES": expected_uuid,
    }:
        raise ValueError("CUDA child visibility environment differs from frozen GPU1 UUID")
    if not re.fullmatch(r"GPU-[0-9a-f-]{32,}", expected_uuid):
        raise ValueError("CUDA child expected GPU UUID is invalid")
    preallocation_owner_rows = _nvidia_query(
        ["--query-compute-apps=gpu_uuid,pid,process_name"]
    )
    if any(row and row[0] == expected_uuid for row in preallocation_owner_rows):
        raise RuntimeError(
            "physical GPU1 was not owner-free immediately before torch allocation"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA child must see exactly one available CUDA device")
    torch.cuda.set_device(0)
    probe = torch.empty((1,), dtype=torch.uint8, device="cuda:0")
    probe.zero_()
    torch.cuda.synchronize(0)
    if torch.cuda.current_device() != 0:
        raise RuntimeError("CUDA child current device is not cuda:0")
    properties = torch.cuda.get_device_properties(0)
    inventory = _nvidia_query(
        ["--query-gpu=index,uuid,pci.bus_id,name"]
    )
    matches = [
        row
        for row in inventory
        if len(row) == 4 and row[1] == expected_uuid
    ]
    if (
        len(matches) != 1
        or matches[0][0] != "1"
        or _bus_suffix(matches[0][2]) != _bus_suffix(expected_bus_id)
    ):
        raise RuntimeError("CUDA child expected UUID is not physical GPU1 on the frozen PCI bus")
    owner_rows = _nvidia_query(
        ["--query-compute-apps=gpu_uuid,pid,process_name"]
    )
    namespace_pids = _namespace_process_ids()
    target_device_rows = [
        row
        for row in owner_rows
        if row and row[0] == expected_uuid
    ]
    foreign_device_rows = [
        row
        for row in owner_rows
        if len(row) >= 2
        and row[0] != expected_uuid
        and row[1].isdigit()
        and int(row[1]) in namespace_pids
    ]
    if (
        len(target_device_rows) != 1
        or len(target_device_rows[0]) != 3
        or not target_device_rows[0][1].isdigit()
        or foreign_device_rows
    ):
        raise RuntimeError("torch cuda:0 process owner did not attest to the frozen GPU1 UUID")
    owner_pid = int(target_device_rows[0][1])
    if owner_pid in namespace_pids:
        owner_pid_binding = "process_namespace_pid"
    elif not Path(f"/proc/{owner_pid}").exists():
        # NVML/nvidia-smi can expose the host PID while /proc is mounted in an
        # inner container namespace.  The thermal guard proves that GPU1 was
        # owner-free immediately before launch and binds the first/only
        # invisible PID, so the live torch allocation must observe exactly the
        # same singleton owner row.
        owner_pid_binding = "exclusive_invisible_host_pid_singleton_after_clear"
    else:
        raise RuntimeError(
            "torch cuda:0 owner is visible but outside the process namespace"
        )
    owned_rows = target_device_rows
    payload: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "nvos-v3-cuda-child-attestation-v2",
        "status": "torch_cuda0_live_owner_matches_physical_gpu1_uuid_and_pci",
        "scene": str(scene),
        "observed_epoch": int(time.time()),
        "hostname": socket.gethostname(),
        "environment": environment,
        "expected_gpu": {
            "physical_index": 1,
            "uuid": expected_uuid,
            "pci_bus_id": expected_bus_id,
        },
        "torch_cuda": {
            "visible_device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()),
            "device": "cuda:0",
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
        },
        "process_namespace_pids": namespace_pids,
        "nvidia_inventory_row": matches[0],
        "nvidia_preallocation_owner_rows": [],
        "nvidia_compute_owner_rows": owned_rows,
        "owner_pid_binding": owner_pid_binding,
        "attestation_mechanism": CUDA_ATTESTATION_MECHANISM,
    }
    write_frozen_json(output, payload)
    del probe
    return payload


def _validate_cuda_attestation(
    path: str | Path,
    *,
    scene: str,
    gpu_uuid: str,
    gpu_bus_id: str,
) -> dict[str, Any]:
    payload, digest, source = load_json_object(path, label="NVOS CUDA attestation")
    expected_environment = {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
        "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
    }
    expected_gpu = {
        "physical_index": 1,
        "uuid": gpu_uuid,
        "pci_bus_id": gpu_bus_id,
    }
    torch_cuda = payload.get("torch_cuda")
    if not (
        set(payload) == CUDA_ATTESTATION_FIELDS
        and payload.get("schema_version") == 2
        and payload.get("artifact_type") == "nvos-v3-cuda-child-attestation-v2"
        and payload.get("status")
        == "torch_cuda0_live_owner_matches_physical_gpu1_uuid_and_pci"
        and payload.get("scene") == scene
        and isinstance(payload.get("observed_epoch"), int)
        and not isinstance(payload.get("observed_epoch"), bool)
        and int(payload["observed_epoch"]) > 0
        and isinstance(payload.get("hostname"), str)
        and bool(payload.get("hostname"))
        and payload.get("environment") == expected_environment
        and payload.get("expected_gpu") == expected_gpu
        and isinstance(torch_cuda, Mapping)
        and set(torch_cuda)
        == {
            "visible_device_count",
            "current_device",
            "device",
            "device_name",
            "compute_capability",
            "total_memory",
            "torch_version",
            "torch_cuda_build",
        }
        and torch_cuda.get("visible_device_count") == 1
        and torch_cuda.get("current_device") == 0
        and torch_cuda.get("device") == "cuda:0"
        and isinstance(torch_cuda.get("device_name"), str)
        and bool(torch_cuda.get("device_name"))
        and isinstance(torch_cuda.get("compute_capability"), list)
        and len(torch_cuda["compute_capability"]) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in torch_cuda["compute_capability"]
        )
        and isinstance(torch_cuda.get("total_memory"), int)
        and not isinstance(torch_cuda.get("total_memory"), bool)
        and int(torch_cuda["total_memory"]) > 0
        and isinstance(torch_cuda.get("torch_version"), str)
        and bool(torch_cuda.get("torch_version"))
        and isinstance(torch_cuda.get("torch_cuda_build"), str)
        and bool(torch_cuda.get("torch_cuda_build"))
        and isinstance(payload.get("process_namespace_pids"), list)
        and payload.get("process_namespace_pids")
        and isinstance(payload.get("nvidia_compute_owner_rows"), list)
        and payload.get("nvidia_compute_owner_rows")
        and payload.get("owner_pid_binding")
        in {
            "process_namespace_pid",
            "exclusive_invisible_host_pid_singleton_after_clear",
        }
        and payload.get("attestation_mechanism") == CUDA_ATTESTATION_MECHANISM
    ):
        raise ValueError("NVOS CUDA child attestation differs")
    inventory = payload.get("nvidia_inventory_row")
    if not (
        isinstance(inventory, list)
        and len(inventory) == 4
        and inventory[0] == "1"
        and inventory[1] == gpu_uuid
        and _bus_suffix(str(inventory[2])) == _bus_suffix(gpu_bus_id)
    ):
        raise ValueError("NVOS CUDA child inventory attestation differs")
    if payload.get("nvidia_preallocation_owner_rows") != []:
        raise ValueError("NVOS CUDA preallocation owner set was not empty")
    namespace_pids = {
        int(value)
        for value in payload["process_namespace_pids"]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    owner_rows = payload["nvidia_compute_owner_rows"]
    if not namespace_pids or len(owner_rows) != 1 or any(
        not isinstance(row, list)
        or len(row) != 3
        or row[0] != gpu_uuid
        or not str(row[1]).isdigit()
        for row in owner_rows
    ):
        raise ValueError("NVOS CUDA child PID/UUID ownership attestation differs")
    owner_pid = int(owner_rows[0][1])
    binding = payload["owner_pid_binding"]
    if (
        binding == "process_namespace_pid"
        and owner_pid not in namespace_pids
    ) or (
        binding == "exclusive_invisible_host_pid_singleton_after_clear"
        and owner_pid in namespace_pids
    ):
        raise ValueError("NVOS CUDA child PID namespace binding differs")
    return {"path": str(source), "sha256": digest, "payload": payload}


def _validate_owner_attestation_correlation(
    owner_audit: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> None:
    payload = attestation.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("NVOS CUDA attestation payload is unavailable")
    owner_rows = payload.get("nvidia_compute_owner_rows")
    if not isinstance(owner_rows, list):
        raise ValueError("NVOS CUDA attestation owner rows are unavailable")
    attested_pids = sorted({str(row[1]) for row in owner_rows})
    if attested_pids != owner_audit.get("child_owner_pids"):
        raise ValueError("NVOS CUDA attestation/guard owner PID binding differs")
    binding = payload.get("owner_pid_binding")
    if binding == "process_namespace_pid":
        expected_audit_pids = owner_audit.get("direct_child_owner_pids")
    elif binding == "exclusive_invisible_host_pid_singleton_after_clear":
        expected_audit_pids = owner_audit.get("host_singleton_owner_pids")
    else:
        raise ValueError("NVOS CUDA attestation owner binding mode differs")
    if attested_pids != expected_audit_pids:
        raise ValueError("NVOS CUDA attestation/guard PID namespace mode differs")


def _load_csv_rows(
    path: str | Path,
    *,
    columns: Sequence[str],
    label: str,
) -> tuple[list[dict[str, str]], str, Path]:
    def load(handle) -> list[dict[str, str]]:
        text = handle.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"{label} header differs")
        return [dict(row) for row in reader]

    rows, digest, source = stable_descriptor_load(path, load, label=label)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows, digest, source


def _validate_owner_audit(
    path: str | Path,
    *,
    gpu_uuid: str,
) -> dict[str, Any]:
    rows, digest, source = _load_csv_rows(
        path,
        columns=OWNER_AUDIT_COLUMNS,
        label="NVOS GPU owner audit",
    )
    child_pgids = {row["child_pgid"] for row in rows}
    allowed_events = {
        "prelaunch_owner_clear",
        "runtime_owner_audit",
        "runtime_owner_audit_host_pid_singleton",
        "postexit_owner_clear",
    }
    runtime_rows = [
        row for row in rows
        if row["event"]
        in {"runtime_owner_audit", "runtime_owner_audit_host_pid_singleton"}
    ]
    def pids(value: str) -> list[str]:
        result = [item for item in value.split(";") if item]
        if any(not item.isdigit() for item in result):
            raise ValueError("NVOS GPU owner audit contains an invalid PID")
        return result

    if (
        rows[0]["event"] != "prelaunch_owner_clear"
        or any(rows[0][key] for key in ("owner_pids", "child_owner_pids", "foreign_owner_pids"))
        or len(child_pgids) != 1
        or not next(iter(child_pgids)).isdigit()
        or any(row["gpu_uuid"] != gpu_uuid for row in rows)
        or any(row["event"] not in allowed_events for row in rows)
        or any(row["foreign_owner_pids"] for row in rows)
        or not runtime_rows
        or not any(pids(row["child_owner_pids"]) for row in runtime_rows)
        or any(
            pids(row["owner_pids"]) != pids(row["child_owner_pids"])
            for row in runtime_rows
        )
        or rows[-1]["event"] != "postexit_owner_clear"
        or any(rows[-1][key] for key in ("owner_pids", "child_owner_pids", "foreign_owner_pids"))
    ):
        raise ValueError("NVOS GPU owner audit does not prove exclusive child-PGID ownership")
    return {
        "path": str(source),
        "sha256": digest,
        "sample_count": len(rows),
        "child_pgid": int(next(iter(child_pgids))),
        "child_owner_pids": sorted(
            {pid for row in runtime_rows for pid in pids(row["child_owner_pids"])}
        ),
        "direct_child_owner_pids": sorted(
            {
                pid
                for row in runtime_rows
                if row["event"] == "runtime_owner_audit"
                for pid in pids(row["child_owner_pids"])
            }
        ),
        "host_singleton_owner_pids": sorted(
            {
                pid
                for row in runtime_rows
                if row["event"] == "runtime_owner_audit_host_pid_singleton"
                for pid in pids(row["child_owner_pids"])
            }
        ),
    }


def _validate_telemetry(
    path: str | Path,
    *,
    gpu_bus_id: str,
) -> dict[str, Any]:
    rows, digest, source = _load_csv_rows(
        path,
        columns=TELEMETRY_COLUMNS,
        label="NVOS GPU telemetry",
    )
    forbidden = ("abort", "failed", "unresponsive", "foreign_compute_owner")
    if any(
        row["gpu"] != "1"
        or _bus_suffix(row["bus_id"]) != _bus_suffix(gpu_bus_id)
        or any(token in row["event"] for token in forbidden)
        for row in rows
    ):
        raise ValueError("NVOS GPU telemetry records a foreign GPU or guard failure")
    if not any(row["event"] == "cuda_release_verified_no_compute_owner" for row in rows):
        raise ValueError("NVOS GPU telemetry lacks successful CUDA release verification")
    return {"path": str(source), "sha256": digest, "sample_count": len(rows)}


def prepare_scene_command(
    *,
    output: Path,
    run_manifest: Path,
    scene: str,
    result: Path,
    telemetry: Path,
    owner_audit: Path,
    attestation: Path,
    postcheck: Path,
    receipt: Path,
    evaluator_log: Path,
    guard: Path,
    gpu_uuid: str,
    gpu_bus_id: str,
    command: Sequence[str],
) -> dict[str, Any]:
    argv = [str(value) for value in command]
    if argv[:1] == ["--"]:
        argv.pop(0)
    if not argv or not all(argv):
        raise ValueError("NVOS guarded scene command is empty")
    manifest, _ = _stable_json(run_manifest)
    output_identity_record = verify_output_tree(
        manifest.get("output_identity")
    )["output_identity"]
    thermal_safety_contract = manifest.get("thermal_safety_contract")
    if not isinstance(thermal_safety_contract, Mapping):
        raise ValueError("NVOS run manifest lacks thermal_safety_contract")
    artifact_paths = {
        "result": result,
        "telemetry": telemetry,
        "owner_audit": owner_audit,
        "cuda_attestation": attestation,
        "command": output,
        "postcheck": postcheck,
        "receipt": receipt,
        "evaluator_log": evaluator_log,
    }
    artifact_bindings = {
        name: _artifact_binding(
            path,
            output_identity_record=output_identity_record,
            label=f"NVOS scene {name}",
        )
        for name, path in artifact_paths.items()
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos-v3-scene-command-v1",
        "scene": scene,
        "run_manifest": file_record(run_manifest),
        "result_path": artifact_bindings["result"]["resolved_path"],
        "telemetry_path": artifact_bindings["telemetry"]["resolved_path"],
        "owner_audit_path": artifact_bindings["owner_audit"]["resolved_path"],
        "cuda_attestation_path": artifact_bindings["cuda_attestation"][
            "resolved_path"
        ],
        "output_identity": output_identity_record,
        "artifact_bindings": artifact_bindings,
        "thermal_safety_contract": dict(thermal_safety_contract),
        "guard": file_record(guard),
        "gpu_identity": {
            "physical_index": 1,
            "uuid": gpu_uuid,
            "pci_bus_id": gpu_bus_id,
        },
        "cuda_environment": {
            "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
            "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
        },
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
    }
    write_frozen_json(output, payload)
    return {"command": file_record(output)}


def _validate_scene_command(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    payload, digest, source = load_json_object(path, label="NVOS scene command")
    required_fields = {
        "schema_version",
        "artifact_type",
        "scene",
        "run_manifest",
        "result_path",
        "telemetry_path",
        "owner_audit_path",
        "cuda_attestation_path",
        "output_identity",
        "artifact_bindings",
        "thermal_safety_contract",
        "guard",
        "gpu_identity",
        "cuda_environment",
        "argv",
        "argv_sha256",
    }
    if not (
        set(payload) == required_fields
        and payload.get("schema_version") == 1
        and payload.get("artifact_type") == "nvos-v3-scene-command-v1"
        and isinstance(payload.get("argv"), list)
        and payload.get("argv")
        and payload.get("argv_sha256") == canonical_json_sha256(payload["argv"])
    ):
        raise ValueError("NVOS scene command record differs")
    validate_file_record(payload.get("guard"), label="NVOS command thermal guard")
    manifest_path = validate_file_record(
        payload.get("run_manifest"), label="NVOS command manifest"
    )
    manifest, _ = _stable_json(manifest_path)
    output_identity_record = verify_output_tree(
        manifest.get("output_identity")
    )["output_identity"]
    if payload.get("output_identity") != output_identity_record:
        raise ValueError("NVOS scene command output identity differs")
    if payload.get("thermal_safety_contract") != manifest.get(
        "thermal_safety_contract"
    ):
        raise ValueError("NVOS scene command thermal safety contract differs")
    artifact_bindings = payload.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping) or set(artifact_bindings) != SCENE_ARTIFACT_NAMES:
        raise ValueError("NVOS scene command artifact binding set differs")
    validated_bindings = {
        name: _validate_artifact_binding(
            artifact_bindings[name],
            output_identity_record=output_identity_record,
            label=f"NVOS scene {name}",
        )
        for name in sorted(SCENE_ARTIFACT_NAMES)
    }
    if Path(str(validated_bindings["command"]["resolved_path"])) != source:
        raise ValueError("NVOS scene command artifact does not bind itself")
    if not (
        payload.get("result_path")
        == validated_bindings["result"]["resolved_path"]
        and payload.get("telemetry_path")
        == validated_bindings["telemetry"]["resolved_path"]
        and payload.get("owner_audit_path")
        == validated_bindings["owner_audit"]["resolved_path"]
        and payload.get("cuda_attestation_path")
        == validated_bindings["cuda_attestation"]["resolved_path"]
    ):
        raise ValueError("NVOS scene command artifact paths differ")
    gpu = payload.get("gpu_identity")
    if not (
        isinstance(gpu, Mapping)
        and set(gpu) == {"physical_index", "uuid", "pci_bus_id"}
        and gpu.get("physical_index") == 1
        and re.fullmatch(r"GPU-[0-9a-f-]{32,}", str(gpu.get("uuid", "")))
        and bool(str(gpu.get("pci_bus_id", "")))
        and payload.get("cuda_environment")
        == {
            "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": gpu.get("uuid"),
            "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
            "NVIDIA_VISIBLE_DEVICES": gpu.get("uuid"),
        }
    ):
        raise ValueError("NVOS scene command GPU identity/environment differs")
    argv = payload["argv"]
    def option(name: str) -> str:
        if argv.count(name) != 1:
            raise ValueError(f"NVOS scene command option differs: {name}")
        index = argv.index(name)
        if index + 1 >= len(argv):
            raise ValueError(f"NVOS scene command option lacks value: {name}")
        return str(argv[index + 1])

    if not (
        option("--scene-id") == payload.get("scene")
        and option("--device") == "cuda:0"
        and option("--candidate-id") == "registered-region-v3"
        and Path(option("--gpu-attestation-output")).resolve()
        == Path(str(payload["cuda_attestation_path"])).resolve()
        and option("--expected-gpu-uuid") == gpu.get("uuid")
        and _bus_suffix(option("--expected-gpu-bus-id"))
        == _bus_suffix(str(gpu.get("pci_bus_id")))
        and Path(option("--output-dir")).resolve()
        == Path(str(payload["result_path"])).resolve().parent
    ):
        raise ValueError("NVOS scene command CUDA attestation argv differs")
    return payload, digest, source


def _validate_scene_postcheck(
    path: str | Path,
    *,
    scene: str,
    manifest_record: Mapping[str, object],
    result_record: Mapping[str, object],
    gpu_identity: Mapping[str, object],
    runtime_closure_sha256: str,
) -> dict[str, Any]:
    payload, digest, source = load_json_object(path, label="NVOS scene postcheck")
    inventory = payload.get("nvidia_inventory_row")
    proc_identity = payload.get("proc_driver_identity")
    global_lock = payload.get("global_lock")
    singleton = payload.get("kernel_singleton")
    if not (
        set(payload) == POSTCHECK_FIELDS
        and payload.get("schema_version") == 1
        and payload.get("artifact_type") == "nvos-v3-scene-postcheck-v1"
        and payload.get("status")
        == "closure_lock_uuid_pci_and_post_owner_verified"
        and payload.get("scene") == scene
        and isinstance(payload.get("observed_epoch"), int)
        and not isinstance(payload.get("observed_epoch"), bool)
        and int(payload["observed_epoch"]) > 0
        and payload.get("run_manifest") == dict(manifest_record)
        and payload.get("result") == dict(result_record)
        and payload.get("runtime_closure_sha256") == runtime_closure_sha256
        and re.fullmatch(r"[0-9a-f]{64}", runtime_closure_sha256)
        and payload.get("gpu_identity") == dict(gpu_identity)
        and payload.get("compute_owners") == []
        and isinstance(inventory, list)
        and len(inventory) == 3
        and inventory[0] == "1"
        and inventory[1] == gpu_identity.get("uuid")
        and _bus_suffix(str(inventory[2]))
        == _bus_suffix(str(gpu_identity.get("pci_bus_id", "")))
        and isinstance(proc_identity, list)
        and len(proc_identity) == 3
        and proc_identity[0] == "1"
        and proc_identity[1] == gpu_identity.get("uuid")
        and _bus_suffix(str(proc_identity[2]))
        == _bus_suffix(str(gpu_identity.get("pci_bus_id", "")))
        and isinstance(payload.get("pcie_config_prefix_hex"), str)
        and re.fullmatch(r"[0-9a-f]{32}", payload["pcie_config_prefix_hex"])
        and set(str(payload["pcie_config_prefix_hex"])) != {"f"}
        and isinstance(global_lock, Mapping)
        and set(global_lock) == {"path", "fd", "device", "inode", "links"}
        and global_lock.get("path") == "/root/RADIO-GS/output/.physical_gpu1.lock"
        and isinstance(global_lock.get("fd"), int)
        and not isinstance(global_lock.get("fd"), bool)
        and int(global_lock["fd"]) >= 0
        and isinstance(global_lock.get("device"), int)
        and not isinstance(global_lock.get("device"), bool)
        and int(global_lock["device"]) > 0
        and isinstance(global_lock.get("inode"), int)
        and not isinstance(global_lock.get("inode"), bool)
        and int(global_lock["inode"]) > 0
        and global_lock.get("links") == 1
        and isinstance(singleton, Mapping)
        and set(singleton) == {"protocol", "fd", "socket_type"}
        and singleton.get("protocol")
        == "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
        and isinstance(singleton.get("fd"), int)
        and not isinstance(singleton.get("fd"), bool)
        and int(singleton["fd"]) >= 0
        and isinstance(singleton.get("socket_type"), int)
        and not isinstance(singleton.get("socket_type"), bool)
        and int(singleton["socket_type"]) > 0
    ):
        raise ValueError("NVOS scene postcheck differs")
    validate_file_record(payload["run_manifest"], label="NVOS postcheck manifest")
    validate_file_record(payload["result"], label="NVOS postcheck result")
    return {
        "path": str(source),
        "sha256": digest,
        "payload": payload,
    }


def validate_scene_receipt(
    path: str | Path,
    *,
    run_manifest: str | Path,
    scene: str,
    result: str | Path,
) -> dict[str, Any]:
    receipt, digest, source = load_json_object(path, label="NVOS scene receipt")
    if not (
        set(receipt) == RECEIPT_FIELDS
        and receipt.get("schema_version") == 1
        and receipt.get("artifact_type") == "nvos-v3-scene-receipt-v1"
        and receipt.get("status")
        == "guard_exit_zero_cuda_attested_exclusive_owner_postchecked"
        and receipt.get("scene") == scene
    ):
        raise ValueError("NVOS scene receipt schema/status differs")
    manifest_record = receipt.get("run_manifest")
    manifest_path = validate_file_record(manifest_record, label="NVOS receipt manifest")
    if manifest_path != Path(run_manifest).resolve():
        raise ValueError("NVOS scene receipt belongs to another run manifest")
    manifest = verify_manifest_closure(
        manifest_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    result_path = validate_file_record(receipt.get("result"), label="NVOS scene result")
    if result_path != Path(result).resolve():
        raise ValueError("NVOS scene receipt belongs to another result")
    command_path = validate_file_record(receipt.get("command"), label="NVOS scene command")
    command, _, _ = _validate_scene_command(command_path)
    gpu = command.get("gpu_identity", {})
    artifact_bindings = command["artifact_bindings"]
    output_identity_record = command["output_identity"]
    if not (
        command.get("scene") == scene
        and command.get("run_manifest") == manifest_record
        and command.get("result_path") == str(result_path)
        and command.get("cuda_environment")
        == {
            "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": gpu.get("uuid"),
            "GPU_OWNER_PID_NAMESPACE_MODE": GPU_OWNER_PID_NAMESPACE_MODE,
            "NVIDIA_VISIBLE_DEVICES": gpu.get("uuid"),
        }
    ):
        raise ValueError("NVOS scene command/receipt binding differs")
    if not (
        receipt.get("output_identity") == output_identity_record
        and receipt.get("artifact_bindings") == artifact_bindings
        and receipt.get("thermal_safety_contract")
        == command.get("thermal_safety_contract")
        and Path(str(artifact_bindings["receipt"]["resolved_path"])) == source
        and Path(str(artifact_bindings["result"]["resolved_path"])) == result_path
    ):
        raise ValueError("NVOS scene receipt artifact containment differs")
    telemetry = _validate_telemetry(
        command["telemetry_path"], gpu_bus_id=str(gpu.get("pci_bus_id", ""))
    )
    owners = _validate_owner_audit(
        command["owner_audit_path"], gpu_uuid=str(gpu.get("uuid", ""))
    )
    attestation = _validate_cuda_attestation(
        command["cuda_attestation_path"],
        scene=scene,
        gpu_uuid=str(gpu.get("uuid", "")),
        gpu_bus_id=str(gpu.get("pci_bus_id", "")),
    )
    _validate_owner_attestation_correlation(owners, attestation)
    for key, observed in (
        ("telemetry", telemetry),
        ("owner_audit", owners),
        ("cuda_attestation", {k: attestation[k] for k in ("path", "sha256")}),
    ):
        if receipt.get(key) != observed:
            raise ValueError(f"NVOS scene receipt {key} changed")
    postcheck_path = validate_file_record(
        receipt.get("postcheck"), label="NVOS scene postcheck"
    )
    if Path(str(artifact_bindings["postcheck"]["resolved_path"])) != postcheck_path:
        raise ValueError("NVOS scene postcheck escaped its planned binding")
    _validate_scene_postcheck(
        postcheck_path,
        scene=scene,
        manifest_record=manifest_record,
        result_record=receipt["result"],
        gpu_identity=gpu,
        runtime_closure_sha256=str(manifest["runtime_closure_sha256"]),
    )
    return {"receipt": {"path": str(source), "sha256": digest}, "payload": receipt}


def write_scene_postcheck(
    *,
    output: Path,
    run_manifest: Path,
    scene: str,
    result: Path,
    gpu_uuid: str,
    gpu_bus_id: str,
    lock_fd: int,
    singleton_fd: int,
) -> dict[str, Any]:
    from radio_gs.scripts.surface_gpu1_lock_supervisor import (
        verify_inherited_lock,
        verify_inherited_singleton,
    )

    manifest, _ = _stable_json(run_manifest)
    output_identity_record = verify_output_tree(
        manifest.get("output_identity")
    )["output_identity"]
    _artifact_binding(
        output,
        output_identity_record=output_identity_record,
        label="NVOS scene postcheck",
    )
    _artifact_binding(
        result,
        output_identity_record=output_identity_record,
        label="NVOS scene result",
    )
    closure = verify_manifest_closure(
        run_manifest,
        repo_root=Path(__file__).resolve().parents[2],
    )
    lock = verify_inherited_lock(lock_fd)
    singleton = verify_inherited_singleton(singleton_fd)
    inventory = _nvidia_query(["--query-gpu=index,uuid,pci.bus_id"])
    matches = [row for row in inventory if len(row) == 3 and row[1] == gpu_uuid]
    owners = _nvidia_query(["--query-compute-apps=gpu_uuid,pid"])
    matching_owners = [row[1] for row in owners if len(row) >= 2 and row[0] == gpu_uuid]
    proc_matches: list[tuple[str, str, str]] = []
    for candidate in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        content = candidate.read_text(encoding="utf-8")
        minor = re.search(r"^Device Minor:\s+(\S+)", content, re.MULTILINE)
        uuid = re.search(r"^GPU UUID:\s+(\S+)", content, re.MULTILINE)
        bus = re.search(r"^Bus Location:\s+(\S+)", content, re.MULTILINE)
        if minor and uuid and bus and minor.group(1) == "1":
            proc_matches.append((minor.group(1), uuid.group(1), bus.group(1)))
    if not (
        len(matches) == 1
        and matches[0][0] == "1"
        and _bus_suffix(matches[0][2]) == _bus_suffix(gpu_bus_id)
        and not matching_owners
        and len(proc_matches) == 1
        and proc_matches[0][1] == gpu_uuid
        and _bus_suffix(proc_matches[0][2]) == _bus_suffix(gpu_bus_id)
    ):
        raise ValueError("NVOS scene GPU1 postcheck identity/owner differs")
    pci_config = Path("/sys/bus/pci/devices") / proc_matches[0][2] / "config"
    try:
        with pci_config.open("rb") as handle:
            prefix = handle.read(16).hex()
    except OSError as error:
        raise ValueError("NVOS scene GPU1 PCI config is unreadable") from error
    if not re.fullmatch(r"[0-9a-f]{32}", prefix) or set(prefix) == {"f"}:
        raise ValueError("NVOS scene GPU1 PCI config is unresponsive")
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos-v3-scene-postcheck-v1",
        "status": "closure_lock_uuid_pci_and_post_owner_verified",
        "scene": scene,
        "observed_epoch": int(time.time()),
        "run_manifest": file_record(run_manifest),
        "result": file_record(result),
        "runtime_closure_sha256": closure["runtime_closure_sha256"],
        "gpu_identity": {
            "physical_index": 1,
            "uuid": gpu_uuid,
            "pci_bus_id": gpu_bus_id,
        },
        "nvidia_inventory_row": matches[0],
        "proc_driver_identity": list(proc_matches[0]),
        "pcie_config_prefix_hex": prefix,
        "compute_owners": [],
        "global_lock": lock,
        "kernel_singleton": singleton,
    }
    write_frozen_json(output, payload)
    return {"postcheck": file_record(output)}


def finalize_scene_receipt(
    *,
    output: Path,
    command_record: Path,
    postcheck: Path,
) -> dict[str, Any]:
    command, _, command_path = _validate_scene_command(command_record)
    scene = str(command["scene"])
    gpu = command["gpu_identity"]
    manifest_path = validate_file_record(
        command["run_manifest"], label="NVOS command manifest"
    )
    closure = verify_manifest_closure(
        manifest_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    result_path = Path(str(command["result_path"]))
    result_record = file_record(result_path)
    output_identity_record = command["output_identity"]
    artifact_bindings = command["artifact_bindings"]
    if _artifact_binding(
        output,
        output_identity_record=output_identity_record,
        label="NVOS scene receipt",
    ) != artifact_bindings["receipt"]:
        raise ValueError("NVOS receipt destination differs from planned binding")
    if _artifact_binding(
        postcheck,
        output_identity_record=output_identity_record,
        label="NVOS scene postcheck",
    ) != artifact_bindings["postcheck"]:
        raise ValueError("NVOS postcheck path differs from planned binding")
    telemetry = _validate_telemetry(
        command["telemetry_path"], gpu_bus_id=str(gpu["pci_bus_id"])
    )
    owners = _validate_owner_audit(
        command["owner_audit_path"], gpu_uuid=str(gpu["uuid"])
    )
    attestation = _validate_cuda_attestation(
        command["cuda_attestation_path"],
        scene=scene,
        gpu_uuid=str(gpu["uuid"]),
        gpu_bus_id=str(gpu["pci_bus_id"]),
    )
    _validate_owner_attestation_correlation(owners, attestation)
    postcheck_record = _validate_scene_postcheck(
        postcheck,
        scene=scene,
        manifest_record=command["run_manifest"],
        result_record=result_record,
        gpu_identity=gpu,
        runtime_closure_sha256=str(closure["runtime_closure_sha256"]),
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos-v3-scene-receipt-v1",
        "status": "guard_exit_zero_cuda_attested_exclusive_owner_postchecked",
        "scene": scene,
        "run_manifest": command["run_manifest"],
        "result": result_record,
        "command": file_record(command_path),
        "telemetry": telemetry,
        "owner_audit": owners,
        "cuda_attestation": {
            "path": attestation["path"],
            "sha256": attestation["sha256"],
        },
        "postcheck": {
            "path": postcheck_record["path"],
            "sha256": postcheck_record["sha256"],
        },
        "output_identity": output_identity_record,
        "artifact_bindings": artifact_bindings,
        "thermal_safety_contract": command["thermal_safety_contract"],
    }
    write_frozen_json(output, payload)
    return validate_scene_receipt(
        output,
        run_manifest=manifest_path,
        scene=scene,
        result=result_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    closure = subparsers.add_parser("closure")
    closure.add_argument("--repo-root", type=Path, required=True)
    verify = subparsers.add_parser("verify-closure")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    readonly = subparsers.add_parser("verify-readonly-snapshot")
    readonly.add_argument("--repo-root", type=Path, required=True)
    output = subparsers.add_parser("output-identity")
    output.add_argument("--main-root", type=Path, required=True)
    output.add_argument("--output-root", type=Path, required=True)
    output_tree = subparsers.add_parser("output-tree")
    output_tree.add_argument("--main-root", type=Path, required=True)
    output_tree.add_argument("--output-root", type=Path, required=True)
    verify_output = subparsers.add_parser("verify-output-tree")
    verify_output.add_argument("--manifest", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-scene")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-manifest", type=Path, required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--result", type=Path, required=True)
    prepare.add_argument("--telemetry", type=Path, required=True)
    prepare.add_argument("--owner-audit", type=Path, required=True)
    prepare.add_argument("--attestation", type=Path, required=True)
    prepare.add_argument("--postcheck", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--evaluator-log", type=Path, required=True)
    prepare.add_argument("--guard", type=Path, required=True)
    prepare.add_argument("--gpu-uuid", required=True)
    prepare.add_argument("--gpu-bus-id", required=True)
    prepare.add_argument("argv", nargs=argparse.REMAINDER)
    postcheck = subparsers.add_parser("postcheck-scene")
    postcheck.add_argument("--output", type=Path, required=True)
    postcheck.add_argument("--run-manifest", type=Path, required=True)
    postcheck.add_argument("--scene", required=True)
    postcheck.add_argument("--result", type=Path, required=True)
    postcheck.add_argument("--gpu-uuid", required=True)
    postcheck.add_argument("--gpu-bus-id", required=True)
    postcheck.add_argument("--lock-fd", type=int, required=True)
    postcheck.add_argument("--singleton-fd", type=int, required=True)
    finalize = subparsers.add_parser("finalize-scene")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--command-record", type=Path, required=True)
    finalize.add_argument("--postcheck", type=Path, required=True)
    validate = subparsers.add_parser("validate-scene")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--run-manifest", type=Path, required=True)
    validate.add_argument("--scene", required=True)
    validate.add_argument("--result", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "closure":
        payload = build_runtime_closure(args.repo_root)
    elif args.command == "verify-closure":
        payload = verify_manifest_closure(args.manifest, repo_root=args.repo_root)
    elif args.command == "verify-readonly-snapshot":
        payload = verify_readonly_source_snapshot(args.repo_root)
    elif args.command == "output-identity":
        payload = output_identity(args.main_root, args.output_root)
    elif args.command == "output-tree":
        payload = verify_output_tree(
            output_identity(args.main_root, args.output_root)
        )
    elif args.command == "verify-output-tree":
        manifest, _ = _stable_json(args.manifest)
        payload = verify_output_tree(manifest.get("output_identity"))
    elif args.command == "prepare-scene":
        payload = prepare_scene_command(
            output=args.output,
            run_manifest=args.run_manifest,
            scene=args.scene,
            result=args.result,
            telemetry=args.telemetry,
            owner_audit=args.owner_audit,
            attestation=args.attestation,
            postcheck=args.postcheck,
            receipt=args.receipt,
            evaluator_log=args.evaluator_log,
            guard=args.guard,
            gpu_uuid=args.gpu_uuid,
            gpu_bus_id=args.gpu_bus_id,
            command=args.argv,
        )
    elif args.command == "postcheck-scene":
        payload = write_scene_postcheck(
            output=args.output,
            run_manifest=args.run_manifest,
            scene=args.scene,
            result=args.result,
            gpu_uuid=args.gpu_uuid,
            gpu_bus_id=args.gpu_bus_id,
            lock_fd=args.lock_fd,
            singleton_fd=args.singleton_fd,
        )
    elif args.command == "finalize-scene":
        payload = finalize_scene_receipt(
            output=args.output,
            command_record=args.command_record,
            postcheck=args.postcheck,
        )
    else:
        payload = validate_scene_receipt(
            args.receipt,
            run_manifest=args.run_manifest,
            scene=args.scene,
            result=args.result,
        )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
