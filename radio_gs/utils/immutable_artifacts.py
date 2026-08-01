"""Small fail-closed primitives for immutable formal-evaluation artifacts.

Reads reject final-component symlinks and keep hashing/deserialization on one
file descriptor.  Writes publish with a same-directory hard link, so an
existing destination is never replaced, then fsync both the file and parent
directory.  The helpers intentionally offer no unsafe pickle fallback.
"""

from __future__ import annotations

import _codecs
import argparse
import collections
import errno
import hashlib
import json
import os
import pickle
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import numpy as np
import torch


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


def canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact is not finite canonical JSON") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_file_path(path: str | Path, *, create_parent: bool = False) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if create_parent:
        raw.parent.mkdir(parents=True, exist_ok=True)
    parent = raw.parent.resolve(strict=True)
    return parent / raw.name


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _open_regular_nofollow(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("immutable artifacts require O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"refuse to follow artifact symlink: {path}") from exc
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"artifact is not a regular file: {path}")
    return descriptor


def _stat_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _path_stat(path: Path) -> os.stat_result:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"artifact path is not a regular file: {path}")
    return info


def _require_same_identity(
    path_info: os.stat_result,
    descriptor_info: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        path_info.st_dev != descriptor_info.st_dev
        or path_info.st_ino != descriptor_info.st_ino
    ):
        raise ValueError(f"{label} path identity changed")


def _hash_handle(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def stable_descriptor_load(
    path: str | Path,
    loader: Callable[[BinaryIO], _T],
    *,
    expected_sha256: str | None = None,
    label: str = "artifact",
) -> tuple[_T, str, Path]:
    """Hash, load, and rehash one unchanged regular file through one fd."""

    source = _canonical_file_path(path)
    expected = (
        _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
        if expected_sha256 is not None
        else None
    )
    descriptor = _open_regular_nofollow(source)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        before = os.fstat(handle.fileno())
        path_before = _path_stat(source)
        _require_same_identity(path_before, before, label=label)
        first_digest = _hash_handle(handle)
        if expected is not None and first_digest != expected:
            raise ValueError(f"{label} SHA-256 differs")
        handle.seek(0)
        value = loader(handle)
        after_load = os.fstat(handle.fileno())
        second_digest = _hash_handle(handle)
        after_rehash = os.fstat(handle.fileno())
        path_after = _path_stat(source)
        fingerprints = {
            _stat_fingerprint(before),
            _stat_fingerprint(path_before),
            _stat_fingerprint(after_load),
            _stat_fingerprint(after_rehash),
            _stat_fingerprint(path_after),
        }
        if len(fingerprints) != 1:
            raise ValueError(f"{label} changed while being read")
        _require_same_identity(path_after, after_rehash, label=label)
        if second_digest != first_digest:
            raise ValueError(f"{label} digest changed while being read")
        return value, first_digest, source


def sha256_file(path: str | Path) -> str:
    _, digest, _ = stable_descriptor_load(
        path,
        lambda handle: None,
        label="hashed artifact",
    )
    return digest


def file_record(path: str | Path) -> dict[str, str]:
    _, digest, source = stable_descriptor_load(
        path,
        lambda handle: None,
        label="recorded artifact",
    )
    return {"path": str(source), "sha256": digest}


def validate_file_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    _, _, source = stable_descriptor_load(
        str(record["path"]),
        lambda handle: None,
        expected_sha256=str(record["sha256"]),
        label=label,
    )
    return source


def load_json_object(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    label: str = "JSON artifact",
) -> tuple[dict[str, Any], str, Path]:
    def load(handle: BinaryIO) -> object:
        try:
            return json.loads(handle.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {label}") from exc

    value, digest, source = stable_descriptor_load(
        path,
        load,
        expected_sha256=expected_sha256,
        label=label,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, digest, source


def load_torch_payload(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    label: str = "torch artifact",
) -> tuple[object, str, Path]:
    def load(handle: BinaryIO) -> object:
        try:
            return torch.load(
                handle,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "immutable torch loading requires weights_only=True support"
            ) from exc

    return stable_descriptor_load(
        path,
        load,
        expected_sha256=expected_sha256,
        label=label,
    )


_RADIO_CHECKPOINT_ALLOWED_GLOBALS: dict[tuple[str, str], object] = {
    ("argparse", "Namespace"): argparse.Namespace,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch", "DoubleStorage"): torch.DoubleStorage,
    ("torch", "FloatStorage"): torch.FloatStorage,
    ("torch", "HalfStorage"): torch.HalfStorage,
    ("torch", "float32"): torch.float32,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
}


_PROJECT_CHECKPOINT_ALLOWED_GLOBALS: dict[tuple[str, str], object] = {
    ("_codecs", "encode"): _codecs.encode,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("pathlib", "PosixPath"): Path,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
}
for _storage_name in (
    "BoolStorage",
    "ByteStorage",
    "CharStorage",
    "ShortStorage",
    "IntStorage",
    "LongStorage",
    "HalfStorage",
    "BFloat16Storage",
    "FloatStorage",
    "DoubleStorage",
    "ComplexFloatStorage",
    "ComplexDoubleStorage",
):
    _PROJECT_CHECKPOINT_ALLOWED_GLOBALS[("torch", _storage_name)] = getattr(
        torch, _storage_name
    )
for _dtype_name in (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
):
    _PROJECT_CHECKPOINT_ALLOWED_GLOBALS[("torch", _dtype_name)] = getattr(
        torch, _dtype_name
    )


class _RestrictedRadioCheckpointUnpickler(pickle.Unpickler):
    """Unpickler for the externally SHA-bound official RADIO release only."""

    def find_class(self, module: str, name: str) -> object:
        value = _RADIO_CHECKPOINT_ALLOWED_GLOBALS.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(
                f"forbidden RADIO checkpoint global: {module}.{name}"
            )
        return value


class _RestrictedRadioCheckpointPickleModule:
    """Minimal module-shaped adapter required by ``torch.load`` on torch 2.0."""

    __name__ = "radio_gs_restricted_radio_checkpoint_pickle"
    Unpickler = _RestrictedRadioCheckpointUnpickler

    @staticmethod
    def load(handle: BinaryIO, **kwargs: Any) -> object:
        return _RestrictedRadioCheckpointUnpickler(handle, **kwargs).load()


class _RestrictedProjectCheckpointUnpickler(pickle.Unpickler):
    """Unpickler for SHA-bound project checkpoints containing legacy Paths."""

    def find_class(self, module: str, name: str) -> object:
        value = _PROJECT_CHECKPOINT_ALLOWED_GLOBALS.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(
                f"forbidden project checkpoint global: {module}.{name}"
            )
        return value


class _RestrictedProjectCheckpointPickleModule:
    __name__ = "radio_gs_restricted_project_checkpoint_pickle"
    Unpickler = _RestrictedProjectCheckpointUnpickler

    @staticmethod
    def load(handle: BinaryIO, **kwargs: Any) -> object:
        return _RestrictedProjectCheckpointUnpickler(handle, **kwargs).load()


def load_fixed_radio_checkpoint_payload(
    path: str | Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
    label: str = "fixed official RADIO checkpoint",
) -> tuple[object, str, Path]:
    """Load one externally SHA-bound RADIO checkpoint with seven globals.

    PyTorch 2.0 does not expose ``safe_globals``.  The official C-RADIOv4-H
    release contains an otherwise inert ``argparse.Namespace`` in addition to
    tensors, so ``weights_only=True`` cannot read it.  This purpose-specific
    path first verifies the caller-supplied SHA-256 on the open descriptor,
    then uses a restricted unpickler.  It deliberately has no unbound or
    general ``weights_only=False`` fallback.
    """

    def load(handle: BinaryIO) -> object:
        return torch.load(
            handle,
            map_location=map_location,
            weights_only=False,
            pickle_module=_RestrictedRadioCheckpointPickleModule,
        )

    return stable_descriptor_load(
        path,
        load,
        expected_sha256=expected_sha256,
        label=label,
    )


def load_sha_bound_project_checkpoint_payload(
    path: str | Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
    label: str = "SHA-bound project checkpoint",
) -> tuple[object, str, Path]:
    """Load a legacy project checkpoint through a minimal restricted pickle.

    Older RADIO-GS trainers serialized ``pathlib.PosixPath`` values inside
    their inert training configuration.  PyTorch 2.0's weights-only loader
    rejects that class.  This compatibility path requires an external digest
    and permits only tensors, OrderedDict, primitive dtypes, and PosixPath; it
    does not provide a general unsafe-pickle fallback.
    """

    def load(handle: BinaryIO) -> object:
        return torch.load(
            handle,
            map_location=map_location,
            weights_only=False,
            pickle_module=_RestrictedProjectCheckpointPickleModule,
        )

    return stable_descriptor_load(
        path,
        load,
        expected_sha256=expected_sha256,
        label=label,
    )


def load_sha_bound_project_checkpoint_mapping(
    path: str | Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
    label: str = "SHA-bound project checkpoint",
) -> tuple[dict[str, Any], str, Path]:
    value, digest, source = load_sha_bound_project_checkpoint_payload(
        path,
        expected_sha256=expected_sha256,
        map_location=map_location,
        label=label,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value), digest, source


def load_torch_mapping(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    label: str = "torch artifact",
) -> tuple[dict[str, Any], str, Path]:
    value, digest, source = load_torch_payload(
        path,
        expected_sha256=expected_sha256,
        map_location=map_location,
        label=label,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(value), digest, source


def load_surface_region_summary_readout_v2(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], str, Path]:
    """Safely reconstruct a v2 readout without its unsafe class loader."""

    from radio_gs.interfaces.surface_region_summary import (
        SurfaceRegionSummaryReadoutV2,
    )

    if expected_sha256 is None:
        payload, digest, source = load_torch_mapping(
            path,
            map_location=map_location,
            label="surface-region summary checkpoint",
        )
    else:
        payload, digest, source = load_sha_bound_project_checkpoint_mapping(
            path,
            expected_sha256=expected_sha256,
            map_location=map_location,
            label="surface-region summary checkpoint",
        )
    if payload.get("schema_version") != 3:
        raise ValueError("invalid v2 surface-region summary checkpoint")
    architecture_value = payload.get("architecture")
    if not isinstance(architecture_value, Mapping):
        raise ValueError("surface-region summary checkpoint lacks architecture")
    architecture = dict(architecture_value)
    expected_digest = architecture.pop("digest", None)
    required_architecture = {
        "name",
        "feature_dim",
        "geometry_dim",
        "hidden_dim",
        "anchor_conditioned",
        "core_context_conditioned",
        "contract_sha256",
    }
    if (
        set(architecture)
        not in (
            required_architecture,
            required_architecture | {"reliability_attention_mode"},
        )
        or architecture.get("name") != "surface_region_summary_readout_v2"
        or not isinstance(architecture.get("feature_dim"), int)
        or isinstance(architecture.get("feature_dim"), bool)
        or not 1 <= int(architecture["feature_dim"]) <= 16_384
        or architecture.get("geometry_dim") != 14
        or not isinstance(architecture.get("hidden_dim"), int)
        or isinstance(architecture.get("hidden_dim"), bool)
        or not 1 <= int(architecture["hidden_dim"]) <= 4096
        or architecture.get("anchor_conditioned") != "true"
        or architecture.get("core_context_conditioned") != "true"
        or not isinstance(architecture.get("contract_sha256"), str)
        or _SHA256.fullmatch(str(architecture["contract_sha256"])) is None
        or architecture.get("reliability_attention_mode", "log_prior")
        not in {"log_prior", "input_only"}
    ):
        raise ValueError("v2 surface-region architecture differs")
    if not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None:
        raise ValueError("surface-region summary architecture lacks digest")
    hidden_dim = int(architecture["hidden_dim"])
    feature_dim = int(architecture["feature_dim"])
    geometry_dim = int(architecture["geometry_dim"])
    query_dim = feature_dim + geometry_dim
    expected_shapes = {
        "feature_encoder.0.weight": (feature_dim,),
        "feature_encoder.0.bias": (feature_dim,),
        "feature_encoder.1.weight": (hidden_dim, feature_dim),
        "feature_encoder.1.bias": (hidden_dim,),
        "geometry_encoder.0.weight": (geometry_dim,),
        "geometry_encoder.0.bias": (geometry_dim,),
        "geometry_encoder.1.weight": (hidden_dim, geometry_dim),
        "geometry_encoder.1.bias": (hidden_dim,),
        "geometry_encoder.3.weight": (hidden_dim, hidden_dim),
        "geometry_encoder.3.bias": (hidden_dim,),
        "query_encoder.0.weight": (query_dim,),
        "query_encoder.0.bias": (query_dim,),
        "query_encoder.1.weight": (hidden_dim, query_dim),
        "query_encoder.1.bias": (hidden_dim,),
        "key.0.weight": (hidden_dim,),
        "key.0.bias": (hidden_dim,),
        "key.1.weight": (hidden_dim, hidden_dim),
        "key.1.bias": (hidden_dim,),
        "residual.0.weight": (hidden_dim,),
        "residual.0.bias": (hidden_dim,),
        "residual.1.weight": (hidden_dim, hidden_dim),
        "residual.1.bias": (hidden_dim,),
        "residual.3.weight": (feature_dim, hidden_dim),
        "residual.3.bias": (feature_dim,),
    }
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(expected_shapes):
        raise ValueError("surface-region summary state_dict fields differ")
    for name, shape in expected_shapes.items():
        value = state_dict[name]
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"surface-region summary state tensor {name} differs")
    model = SurfaceRegionSummaryReadoutV2(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        reliability_attention_mode=str(
            architecture.get("reliability_attention_mode", "log_prior")
        ),
    )
    observed = model.architecture(str(architecture["contract_sha256"]))["digest"]
    if observed != expected_digest:
        raise ValueError("v2 surface-region architecture digest mismatch")
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    return model, payload, digest, source


def fsync_directory(path: str | Path) -> None:
    directory = Path(path).resolve(strict=True)
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary_noclobber(temporary: Path, output: Path) -> None:
    try:
        os.link(temporary, output, follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable artifact already exists: {output}") from exc
    fsync_directory(output.parent)
    temporary.unlink()
    fsync_directory(output.parent)


def write_bytes_noclobber(path: str | Path, value: bytes) -> Path:
    output = _canonical_file_path(path, create_parent=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_noclobber(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_frozen_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_file_path(path, create_parent=True)
    serialized = json.dumps(
        dict(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if output.exists() or output.is_symlink():
        observed, _, _ = load_json_object(output, label="frozen JSON artifact")
        if observed != dict(value):
            raise ValueError(
                f"existing frozen artifact differs from recomputation: {output}"
            )
        return output
    return write_bytes_noclobber(output, serialized)


def write_torch_noclobber(path: str | Path, value: object) -> Path:
    output = _canonical_file_path(path, create_parent=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable torch artifact already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b", closefd=True) as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_noclobber(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
