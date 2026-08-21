"""Bounded-memory exact adjoint for PyTorch ZIP responsibility caches."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
from typing import BinaryIO, Iterator, Mapping
import zipfile

import numpy as np
import torch


_STORAGE_DTYPES = {
    "LongStorage": (np.dtype("<i8"), torch.int64),
    "FloatStorage": (np.dtype("<f4"), torch.float32),
    "DoubleStorage": (np.dtype("<f8"), torch.float64),
}


@dataclass(frozen=True)
class StorageReference:
    key: str
    storage_name: str
    numel: int


@dataclass(frozen=True)
class TensorReference:
    storage: StorageReference
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class StreamingAdjointResult:
    weighted_sum: torch.Tensor
    visible_mass: torch.Tensor
    primitive_probability: torch.Tensor
    hit_count: int
    chunk_hits: int
    visible_mass_max_abs_error: float
    constant_conservation_max_abs_error: float


class _StorageKind:
    def __init__(self, name: str) -> None:
        self.name = name


def _rebuild_tensor(
    storage: StorageReference,
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    *_: object,
) -> TensorReference:
    return TensorReference(storage, int(offset), tuple(shape), tuple(stride))


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageKind(name)
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise ValueError(f"unsupported global in responsibility metadata: {module}.{name}")

    def persistent_load(self, value: object) -> StorageReference:
        if not isinstance(value, tuple) or len(value) < 5 or value[0] != "storage":
            raise ValueError("unsupported PyTorch persistent object")
        storage_type, key, numel = value[1], str(value[2]), int(value[4])
        name = getattr(storage_type, "name", "")
        if name not in _STORAGE_DTYPES:
            raise ValueError(f"unsupported responsibility storage type: {name}")
        return StorageReference(key=key, storage_name=name, numel=numel)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _tensor_digest_prefix(dtype: torch.dtype, shape: tuple[int, ...]) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(dtype), "shape": list(shape)}))
    digest.update(b"\0")
    return digest


def _read_exact(handle: BinaryIO, byte_count: int) -> bytes:
    blocks = bytearray()
    while len(blocks) < byte_count:
        block = handle.read(byte_count - len(blocks))
        if not block:
            raise EOFError("responsibility storage ended before its declared shape")
        blocks.extend(block)
    return bytes(blocks)


def _payload_and_prefix(
    archive: zipfile.ZipFile,
) -> tuple[Mapping[str, object], str]:
    names = archive.namelist()
    metadata = [name for name in names if name.endswith("/data.pkl")]
    if len(metadata) != 1:
        raise ValueError("responsibility ZIP must contain one data.pkl")
    payload = _MetadataUnpickler(io.BytesIO(archive.read(metadata[0]))).load()
    if not isinstance(payload, Mapping):
        raise ValueError("responsibility metadata payload must be a mapping")
    return payload, metadata[0][:-len("data.pkl")]


def _tensor_references(payload: Mapping[str, object]) -> Mapping[str, TensorReference]:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != {
        "gaussian_ids", "pixel_ids", "weights", "visible_mass"
    }:
        raise ValueError("responsibility tensor metadata differs")
    if not all(isinstance(value, TensorReference) for value in tensors.values()):
        raise ValueError("responsibility tensor metadata is not storage-backed")
    return tensors  # type: ignore[return-value]


def _validate_contiguous_vector(reference: TensorReference, expected: str) -> None:
    if (
        reference.storage.storage_name != expected
        or reference.offset != 0
        or len(reference.shape) != 1
        or reference.stride != (1,)
        or reference.shape[0] != reference.storage.numel
    ):
        raise ValueError("responsibility tensor storage layout differs")


def _read_vector(
    archive: zipfile.ZipFile,
    prefix: str,
    reference: TensorReference,
) -> tuple[np.ndarray, str]:
    numpy_dtype, torch_dtype = _STORAGE_DTYPES[reference.storage.storage_name]
    size = int(reference.shape[0]) * numpy_dtype.itemsize
    digest = _tensor_digest_prefix(torch_dtype, reference.shape)
    with archive.open(prefix + "data/" + reference.storage.key) as handle:
        raw = _read_exact(handle, size)
        if handle.read(1):
            raise ValueError("responsibility storage exceeds its declared shape")
    digest.update(raw)
    return np.frombuffer(raw, dtype=numpy_dtype).copy(), digest.hexdigest()


def normalized_adjoint_from_chunks(
    chunks: Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pixel_probability: torch.Tensor,
    *,
    num_gaussians: int,
    visible_mass: torch.Tensor,
    hit_count: int,
    chunk_hits: int,
) -> StreamingAdjointResult:
    """Apply normalized ``W.T`` while retaining only one sparse chunk."""

    probability = torch.as_tensor(pixel_probability, device="cpu").double().reshape(-1)
    mass = torch.as_tensor(visible_mass, device="cpu").double().reshape(-1)
    if mass.shape != (int(num_gaussians),) or not bool(torch.isfinite(mass).all()):
        raise ValueError("visible mass is invalid")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("pixel probability must be finite in [0,1]")
    numerator = torch.zeros(int(num_gaussians), dtype=torch.float64)
    recomputed_mass = torch.zeros_like(numerator)
    consumed = 0
    for gaussian_ids, pixel_ids, weights in chunks:
        gids = torch.as_tensor(gaussian_ids, device="cpu").long().reshape(-1)
        pids = torch.as_tensor(pixel_ids, device="cpu").long().reshape(-1)
        weight = torch.as_tensor(weights, device="cpu").double().reshape(-1)
        if gids.shape != pids.shape or gids.shape != weight.shape or not gids.numel():
            raise ValueError("streamed responsibility triplets do not align")
        if int(gids.min()) < 0 or int(gids.max()) >= int(num_gaussians):
            raise ValueError("streamed Gaussian id is outside authority")
        if int(pids.min()) < 0 or int(pids.max()) >= probability.numel():
            raise ValueError("streamed pixel id is outside probability raster")
        if not bool(torch.isfinite(weight).all()) or bool((weight <= 0).any()):
            raise ValueError("streamed responsibility weight is invalid")
        numerator.index_add_(0, gids, weight * probability[pids])
        recomputed_mass.index_add_(0, gids, weight)
        consumed += int(gids.numel())
    if consumed != int(hit_count):
        raise ValueError("streamed responsibility hit count differs")
    mass_error = float((recomputed_mass - mass).abs().max())
    if mass_error != 0.0:
        raise ValueError("streamed W.T @ 1 differs from stored visible mass")
    visible = mass > 0
    constant_error = float(
        (recomputed_mass[visible] / mass[visible] - 1.0).abs().max()
    ) if bool(visible.any()) else 0.0
    output = torch.zeros_like(numerator)
    output[visible] = numerator[visible] / mass[visible]
    return StreamingAdjointResult(
        numerator, mass, output, consumed, int(chunk_hits), mass_error,
        constant_error,
    )


def streaming_prompt_adjoint(
    cache_path: str | Path,
    pixel_probability: torch.Tensor,
    *,
    expected_file_sha256: str,
    chunk_hits: int = 1_000_000,
) -> tuple[StreamingAdjointResult, Mapping[str, object]]:
    """Read a prompt cache's sparse storages in bounded aligned chunks."""

    source = Path(cache_path).expanduser().resolve(strict=True)
    if len(expected_file_sha256) != 64 or _sha256(source) != expected_file_sha256:
        raise ValueError("responsibility cache SHA-256 differs")
    if int(chunk_hits) <= 0:
        raise ValueError("chunk_hits must be positive")
    with zipfile.ZipFile(source) as archive:
        payload, prefix = _payload_and_prefix(archive)
        refs = _tensor_references(payload)
        _validate_contiguous_vector(refs["gaussian_ids"], "LongStorage")
        _validate_contiguous_vector(refs["pixel_ids"], "LongStorage")
        _validate_contiguous_vector(refs["weights"], "FloatStorage")
        _validate_contiguous_vector(refs["visible_mass"], "DoubleStorage")
        hits = refs["gaussian_ids"].shape[0]
        if refs["pixel_ids"].shape != (hits,) or refs["weights"].shape != (hits,):
            raise ValueError("responsibility sparse storage lengths differ")
        visible, visible_digest = _read_vector(
            archive, prefix, refs["visible_mass"]
        )
        authority = payload.get("authority")
        stored_digests = payload.get("tensor_sha256")
        if not isinstance(authority, Mapping) or not isinstance(stored_digests, Mapping):
            raise ValueError("responsibility authority or digests are absent")
        if int(authority.get("num_gaussians", 0)) != visible.size:
            raise ValueError("responsibility visible mass row count differs")
        if visible_digest != stored_digests.get("visible_mass"):
            raise ValueError("responsibility visible mass digest differs")

        def chunks() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            entries = [
                prefix + "data/" + refs[name].storage.key
                for name in ("gaussian_ids", "pixel_ids", "weights")
            ]
            digests = {
                name: _tensor_digest_prefix(
                    _STORAGE_DTYPES[refs[name].storage.storage_name][1], refs[name].shape
                )
                for name in ("gaussian_ids", "pixel_ids", "weights")
            }
            with archive.open(entries[0]) as gids_file, archive.open(
                entries[1]
            ) as pids_file, archive.open(entries[2]) as weights_file:
                remaining = hits
                while remaining:
                    count = min(int(chunk_hits), remaining)
                    raw_gids = _read_exact(gids_file, count * 8)
                    raw_pids = _read_exact(pids_file, count * 8)
                    raw_weights = _read_exact(weights_file, count * 4)
                    digests["gaussian_ids"].update(raw_gids)
                    digests["pixel_ids"].update(raw_pids)
                    digests["weights"].update(raw_weights)
                    yield (
                        torch.from_numpy(np.frombuffer(raw_gids, dtype="<i8").copy()),
                        torch.from_numpy(np.frombuffer(raw_pids, dtype="<i8").copy()),
                        torch.from_numpy(np.frombuffer(raw_weights, dtype="<f4").copy()),
                    )
                    remaining -= count
                if gids_file.read(1) or pids_file.read(1) or weights_file.read(1):
                    raise ValueError("responsibility sparse storage has trailing bytes")
            actual = {name: digest.hexdigest() for name, digest in digests.items()}
            if any(actual[name] != stored_digests.get(name) for name in actual):
                raise ValueError("responsibility sparse tensor digest differs")

        result = normalized_adjoint_from_chunks(
            chunks(),
            pixel_probability,
            num_gaussians=visible.size,
            visible_mass=torch.from_numpy(visible),
            hit_count=hits,
            chunk_hits=int(chunk_hits),
        )
        return result, payload


def streaming_prompt_cache_metadata(cache_path: str | Path) -> Mapping[str, object]:
    """Read only the small pickle metadata, never any tensor storage."""

    source = Path(cache_path).expanduser().resolve(strict=True)
    with zipfile.ZipFile(source) as archive:
        payload, _prefix = _payload_and_prefix(archive)
        _tensor_references(payload)
        return payload


__all__ = [
    "StreamingAdjointResult",
    "normalized_adjoint_from_chunks",
    "streaming_prompt_cache_metadata",
    "streaming_prompt_adjoint",
]
