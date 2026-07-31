"""Fail-closed loading of canonical primitive capability banks and graphs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Mapping
import zipfile

import numpy as np
import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.querying.support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class CanonicalCapabilityBank:
    """Row-aligned official views derived from exactly one canonical field."""

    xyz: torch.Tensor
    valid: torch.Tensor
    appearance: torch.Tensor
    boundary: torch.Tensor
    signatures: Mapping[str, FeatureSpaceSignature]
    metadata: Mapping[str, Any]
    features_are_compact: bool = False

    @property
    def global_rows(self) -> torch.Tensor:
        return torch.where(self.valid)[0]

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    def valid_feature_banks(self) -> dict[str, torch.Tensor]:
        if self.features_are_compact:
            return {
                "appearance": self.appearance,
                "boundary": self.boundary,
            }
        rows = self.global_rows
        return {
            "appearance": self.appearance[rows],
            "boundary": self.boundary[rows],
        }


@dataclass(frozen=True)
class CanonicalPrimitiveReliability:
    """Row-aligned, query-independent precision of canonical descriptors."""

    xyz: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    components: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]

    def valid_confidence(self) -> torch.Tensor:
        return self.confidence[self.valid]


def _load_payload(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported canonical capability cache: {path}")
    return payload


def _stored_zip_member_memmap(
    archive: Path,
    member: zipfile.ZipInfo,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.memmap:
    """Map one uncompressed PyTorch archive storage without deserializing it."""

    if member.compress_type != zipfile.ZIP_STORED:
        raise ValueError("memory-mapped capability storage must be uncompressed")
    with archive.open("rb") as handle:
        handle.seek(member.header_offset)
        header = handle.read(30)
    if len(header) != 30:
        raise ValueError("truncated capability archive local header")
    (
        signature,
        _version,
        _flags,
        _compression,
        _time,
        _date,
        _crc,
        _compressed_size,
        _uncompressed_size,
        filename_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise ValueError("invalid capability archive local header")
    offset = member.header_offset + 30 + filename_length + extra_length
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if member.file_size != expected_bytes:
        raise ValueError(
            f"capability storage byte count mismatch for {member.filename}"
        )
    # Copy-on-write mode is read-only with respect to the archive while still
    # presenting a writable NumPy buffer to torch.from_numpy.
    return np.memmap(
        archive,
        dtype=dtype,
        mode="c",
        offset=offset,
        shape=shape,
        order="C",
    )


def _load_memory_mapped_capability_payload(path: Path) -> Mapping[str, Any]:
    """Load a dense legacy capability cache with O(metadata) resident memory."""

    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"large capability archive lacks metadata sidecar: {sidecar_path}"
        )
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    count = int(metadata["num_gaussians"])
    appearance_dim = int(metadata["appearance_dim"])
    boundary_dim = int(metadata["boundary_dim"])
    expected_sizes = {
        "xyz": count * 3 * np.dtype(np.float32).itemsize,
        "valid": count * np.dtype(np.uint8).itemsize,
        "appearance": count * appearance_dim * np.dtype(np.float16).itemsize,
        "boundary": count * boundary_dim * np.dtype(np.float16).itemsize,
    }
    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if "/data/" in member.filename
        ]
    resolved: dict[str, zipfile.ZipInfo] = {}
    for name, size in expected_sizes.items():
        matches = [member for member in members if member.file_size == size]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely identify {name} storage in {path}"
            )
        resolved[name] = matches[0]
    xyz = torch.from_numpy(
        _stored_zip_member_memmap(
            path,
            resolved["xyz"],
            dtype=np.float32,
            shape=(count, 3),
        )
    )
    valid = torch.from_numpy(
        _stored_zip_member_memmap(
            path,
            resolved["valid"],
            dtype=np.uint8,
            shape=(count,),
        )
    )
    appearance = torch.from_numpy(
        _stored_zip_member_memmap(
            path,
            resolved["appearance"],
            dtype=np.float16,
            shape=(count, appearance_dim),
        )
    )
    boundary = torch.from_numpy(
        _stored_zip_member_memmap(
            path,
            resolved["boundary"],
            dtype=np.float16,
            shape=(count, boundary_dim),
        )
    )
    return {
        "schema_version": 1,
        "xyz": xyz,
        "valid": valid,
        "appearance_dino_v3": appearance,
        "boundary_sam3": boundary,
        "metadata": metadata,
        "_memory_mapped": True,
    }


def _compact_valid_feature_rows(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    name: str,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Materialize only valid rows from a memory-mapped dense legacy tensor."""

    count = int(valid.sum())
    compact = torch.empty(
        count,
        values.shape[1],
        dtype=values.dtype,
        device="cpu",
    )
    cursor = 0
    for start in range(0, valid.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), valid.numel())
        active = valid[start:stop]
        active_count = int(active.sum())
        if active_count == 0:
            continue
        rows = values[start:stop][active]
        if not bool(torch.isfinite(rows).all()):
            raise ValueError(f"capability {name} contains NaN or infinity")
        compact[cursor : cursor + active_count].copy_(rows)
        cursor += active_count
    if cursor != count:
        raise RuntimeError("valid capability compaction lost primitive rows")
    return compact


def load_canonical_capability_bank(
    path: str | Path,
    *,
    expected_field_checkpoint_sha256: str = "",
    require_signatures: bool = True,
) -> CanonicalCapabilityBank:
    cache_path = Path(path)
    if (
        cache_path.stat().st_size >= 2 * 1024**3
        and cache_path.with_suffix(cache_path.suffix + ".json").is_file()
    ):
        payload = _load_memory_mapped_capability_payload(cache_path)
    else:
        payload = _load_payload(cache_path)
    required = {"xyz", "valid", "appearance_dino_v3", "boundary_sam3", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"capability cache lacks keys: {sorted(required - set(payload))}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("capability cache metadata must be a mapping")
    if metadata.get("source") != "canonical_radio_field_official_frozen_capability_views":
        raise ValueError("capability cache was not derived from a canonical RADIO field")
    if metadata.get("custom_adaptor_head") is not False:
        raise ValueError("canonical capability cache must use official frozen adaptors")
    if metadata.get("query_independent") is not True:
        raise ValueError("canonical capability cache must be query independent")
    actual_field_hash = str(metadata.get("field_checkpoint_sha256", ""))
    if expected_field_checkpoint_sha256 and actual_field_hash != expected_field_checkpoint_sha256:
        raise ValueError("capability cache canonical-field hash mismatch")

    memory_mapped = bool(payload.get("_memory_mapped", False))
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    appearance = torch.as_tensor(payload["appearance_dino_v3"]).cpu()
    boundary = torch.as_tensor(payload["boundary_sam3"]).cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if xyz.ndim != 2 or xyz.shape[1] != 3 or valid.shape != (count,):
        raise ValueError("capability xyz/valid rows are malformed")
    if appearance.ndim != 2 or boundary.ndim != 2:
        raise ValueError("capability features must be matrices")
    if appearance.shape[0] != count or boundary.shape[0] != count:
        raise ValueError("capability feature rows do not align with geometry")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError("capability geometry contains NaN or infinity")
    if memory_mapped:
        # Copy and validate only rows exposed by the capability contract.  The
        # dense memory maps are released on return, and downstream consumers
        # no longer need a second multi-gigabyte advanced-indexing allocation.
        appearance = _compact_valid_feature_rows(
            appearance,
            valid,
            name="appearance",
        )
        boundary = _compact_valid_feature_rows(
            boundary,
            valid,
            name="boundary",
        )
    elif not bool(torch.isfinite(appearance).all()) or not bool(
        torch.isfinite(boundary).all()
    ):
        raise ValueError("capability features contain NaN or infinity")

    raw_signatures = metadata.get("capability_signatures")
    if require_signatures and not isinstance(raw_signatures, Mapping):
        raise ValueError("capability cache lacks fail-closed feature signatures")
    signatures = {
        name: FeatureSpaceSignature.from_mapping(value)
        for name, value in dict(raw_signatures or {}).items()
    }
    for name, matrix in (("appearance", appearance), ("boundary", boundary)):
        signature = signatures.get(name)
        if require_signatures and signature is None:
            raise ValueError(f"capability cache lacks {name} signature")
        if signature is not None and signature.adaptor_output_dim != matrix.shape[1]:
            raise ValueError(f"{name} signature output dimension does not match cache")
    return CanonicalCapabilityBank(
        xyz=xyz,
        valid=valid,
        appearance=appearance,
        boundary=boundary,
        signatures=signatures,
        metadata=metadata,
        features_are_compact=memory_mapped,
    )


def load_canonical_primitive_reliability(
    path: str | Path,
    *,
    expected_xyz: torch.Tensor | None = None,
    expected_valid: torch.Tensor | None = None,
    expected_field_checkpoint_sha256: str = "",
) -> CanonicalPrimitiveReliability:
    """Load a reliability sidecar and fail closed on provenance/alignment."""

    payload = _load_payload(path)
    required = {"xyz", "valid", "confidence", "components", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"reliability cache lacks keys: {sorted(required - set(payload))}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("reliability metadata must be a mapping")
    if metadata.get("source") != "canonical_primitive_reliability_v1":
        raise ValueError("unsupported canonical primitive reliability source")
    safety_requirements = {
        "query_independent": True,
        "uses_query": False,
        "uses_text": False,
        "uses_target_labels": False,
        "uses_target_masks": False,
        "uses_metric_feedback": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    for key, expected in safety_requirements.items():
        if metadata.get(key) is not expected:
            raise ValueError(f"reliability cache violates safety contract: {key}")
    actual_field_hash = str(metadata.get("field_checkpoint_sha256", ""))
    if expected_field_checkpoint_sha256 and actual_field_hash != str(
        expected_field_checkpoint_sha256
    ):
        raise ValueError("reliability cache canonical-field hash mismatch")

    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    confidence = torch.as_tensor(payload["confidence"]).float().cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if xyz.shape != (count, 3) or valid.shape != (count,) or confidence.shape != (count,):
        raise ValueError("reliability geometry/confidence rows are malformed")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(confidence).all()):
        raise ValueError("reliability cache contains NaN or infinity")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError("primitive reliability confidence must be in [0,1]")
    if bool((confidence[~valid] != 0).any()):
        raise ValueError("invalid primitive rows must have zero confidence")
    raw_components = payload["components"]
    if not isinstance(raw_components, Mapping):
        raise ValueError("reliability components must be a mapping")
    components: dict[str, torch.Tensor] = {}
    for name in (
        "observation_evidence",
        "multiview_agreement",
        "reconstruction_fidelity",
    ):
        if name not in raw_components:
            raise ValueError(f"reliability cache lacks component: {name}")
        values = torch.as_tensor(raw_components[name]).float().cpu()
        if values.shape != (count,) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"reliability component {name!r} is malformed")
        if bool((values < 0).any()) or bool((values > 1).any()):
            raise ValueError(f"reliability component {name!r} must be in [0,1]")
        components[name] = values

    if expected_xyz is not None:
        reference_xyz = torch.as_tensor(expected_xyz).float().cpu()
        if reference_xyz.shape != xyz.shape or not torch.allclose(
            reference_xyz, xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("reliability cache geometry does not align")
    if expected_valid is not None:
        reference_valid = torch.as_tensor(expected_valid).bool().cpu()
        if reference_valid.shape != valid.shape or not torch.equal(reference_valid, valid):
            raise ValueError("reliability cache valid rows do not align")
    return CanonicalPrimitiveReliability(
        xyz=xyz,
        valid=valid,
        confidence=confidence,
        components=components,
        metadata=metadata,
    )


def load_canonical_support_graph(
    path: str | Path,
    bank: CanonicalCapabilityBank,
) -> PrimitiveSupportGraph:
    payload = _load_payload(path)
    required = {
        "global_rows",
        "num_global_rows",
        "edge_index",
        "edge_weight",
        "raw_affinity",
        "local_sigma",
        "metadata",
    }
    if not required.issubset(payload):
        raise ValueError(f"support graph lacks keys: {sorted(required - set(payload))}")
    if int(payload["num_global_rows"]) != bank.num_gaussians:
        raise ValueError("support graph and capability bank global row counts differ")
    global_rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    if not torch.equal(global_rows, bank.global_rows):
        raise ValueError("support graph nodes do not match valid capability rows")
    metadata = payload["metadata"]
    capability_metadata = metadata.get("capability_metadata", {})
    if capability_metadata.get("field_checkpoint_sha256") != bank.metadata.get(
        "field_checkpoint_sha256"
    ):
        raise ValueError("support graph and capability bank canonical-field hashes differ")
    if capability_metadata.get("radio_checkpoint_sha256") != bank.metadata.get(
        "radio_checkpoint_sha256"
    ):
        raise ValueError("support graph and capability bank RADIO hashes differ")
    graph_signatures = capability_metadata.get("capability_signatures")
    if not isinstance(graph_signatures, Mapping):
        raise ValueError("support graph lacks source capability signatures")
    for name, signature in bank.signatures.items():
        if graph_signatures.get(name) != signature.to_dict():
            raise ValueError(
                f"support graph and capability bank {name} signatures differ"
            )
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"],
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=payload["local_sigma"],
        num_nodes=int(global_rows.numel()),
        edge_channels={
            str(name): torch.as_tensor(values).float()
            for name, values in dict(payload.get("edge_channels", {})).items()
        },
    )
