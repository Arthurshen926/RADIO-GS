"""Fail-closed tensor-cache loading and MPR validation helpers."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_payload,
    sha256_file,
    stable_descriptor_load,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARDED_MPR_SCHEMA = "radio_gs.channel_sharded_mpr.v1"
_MPR_LOCAL_MIRROR_ENV = "RADIO_GS_MPR_LOCAL_MIRROR_DIR"


def _safe_manifest_member(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if (
        not str(value)
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 1
    ):
        raise ValueError(f"sharded MPR {label} path is unsafe")
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"sharded MPR {label} must not be a symlink")
    return path


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"sharded MPR member is not a regular file: {path}")
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _local_mirror_root() -> Path | None:
    raw = os.environ.get(_MPR_LOCAL_MIRROR_ENV)
    if raw is None or not raw.strip():
        return None
    root = Path(os.path.abspath(os.path.expanduser(raw)))
    try:
        info = os.stat(root, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{_MPR_LOCAL_MIRROR_ENV} does not name an existing directory"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            f"{_MPR_LOCAL_MIRROR_ENV} must name a non-symlink directory"
        )
    return root


def _local_mirror_member(
    mirror_root: Path,
    authority_path: Path,
    *,
    expected_bytes: int,
    label: str,
) -> tuple[Path, tuple[int, int, int, int, int]]:
    """Resolve one same-name mirror member without weakening its authority."""

    mirror_path = mirror_root / authority_path.name
    if mirror_path.is_symlink():
        raise ValueError(f"sharded MPR local mirror {label} must not be a symlink")
    try:
        identity = _path_identity(mirror_path)
    except FileNotFoundError as exc:
        raise ValueError(f"sharded MPR local mirror {label} is missing") from exc
    if identity[2] != int(expected_bytes):
        raise ValueError(f"sharded MPR local mirror {label} byte count differs")
    return mirror_path, identity


@dataclass(frozen=True)
class _ChannelShard:
    # ``path`` remains the manifest-relative authority member.  The optional
    # mirror is only an I/O transport and must never enter frozen provenance.
    path: Path
    mapped_path: Path
    sha256: str
    channel_start: int
    channel_stop: int
    identity: tuple[int, int, int, int, int]


class ShardedMPRCache:
    """Validated, disk-backed MPR whose feature channels are raw fp16 shards.

    Geometry and reliability are intentionally resident because they are only
    O(N).  Feature rows are copied directly from read-only memmaps on demand;
    no operation exposed here can materialize the complete N x D target.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        metadata: dict[str, Any],
        geometry_fingerprint: dict[str, Any],
        xyz: torch.Tensor,
        valid: torch.Tensor,
        view_counts: torch.Tensor,
        reliability: torch.Tensor,
        feature_dim: int,
        shards: list[_ChannelShard],
        support_record: dict[str, str],
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.metadata = metadata
        self.geometry_fingerprint = geometry_fingerprint
        self.xyz = xyz
        self.valid = valid
        self.view_counts = view_counts
        self.observation_count = view_counts.long()
        self.reliability = reliability.float()
        self.feature_dim = int(feature_dim)
        self.num_rows = int(xyz.shape[0])
        self.shards = tuple(shards)
        self.support_record = dict(support_record)
        self._mapped_shards = tuple(
            np.memmap(
                shard.mapped_path,
                mode="r",
                dtype="<f2",
                shape=(
                    self.num_rows,
                    shard.channel_stop - shard.channel_start,
                ),
                order="C",
            )
            for shard in self.shards
        )

    def get(self, key: str, default: Any = None) -> Any:
        return {
            "xyz": self.xyz,
            "valid": self.valid,
            "view_counts": self.view_counts,
            "reliability": self.reliability,
            "metadata": self.metadata,
            "geometry_fingerprint": self.geometry_fingerprint,
        }.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None:
            if key == "features":
                raise RuntimeError(
                    "sharded MPR features require fetch_rows(); full loading is forbidden"
                )
            raise KeyError(key)
        return value

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_rows, self.feature_dim

    def fetch_rows(self, rows: torch.Tensor) -> torch.Tensor:
        raw_indices = torch.as_tensor(rows)
        if raw_indices.ndim != 1:
            raise ValueError("sharded MPR row indices must be one-dimensional")
        indices = raw_indices.detach().long().cpu()
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.num_rows
        ):
            raise IndexError("sharded MPR row index is out of bounds")
        result = torch.empty(
            (indices.numel(), self.feature_dim), dtype=torch.float16
        )
        numpy_rows = indices.numpy()
        for shard, mapped in zip(self.shards, self._mapped_shards):
            selected = np.asarray(mapped[numpy_rows], dtype=np.float16).copy()
            result[:, shard.channel_start : shard.channel_stop].copy_(
                torch.from_numpy(selected)
            )
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "storage": _SHARDED_MPR_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "support": dict(self.support_record),
            "shards": [
                {
                    "path": str(shard.path),
                    "sha256": shard.sha256,
                    "channel_start": shard.channel_start,
                    "channel_stop": shard.channel_stop,
                }
                for shard in self.shards
            ],
        }


def _require_tensor(
    payload: Mapping[str, Any],
    key: str,
) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"MPR cache {key!r} must be a tensor")
    return value


def _all_finite(values: torch.Tensor, *, row_chunk: int = 4096) -> bool:
    if not values.is_floating_point() and not values.is_complex():
        return True
    if values.ndim == 0:
        return bool(torch.isfinite(values).item())
    for start in range(0, int(values.shape[0]), int(row_chunk)):
        if not bool(torch.isfinite(values[start : start + row_chunk]).all()):
            return False
    return True


def _all_zero_on_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    row_chunk: int = 4096,
) -> bool:
    for start in range(0, int(values.shape[0]), int(row_chunk)):
        stop = min(start + int(row_chunk), int(values.shape[0]))
        local_mask = mask[start:stop]
        if bool(local_mask.any()) and bool(
            values[start:stop][local_mask].ne(0).any()
        ):
            return False
    return True


def _xyz_sha256(values: torch.Tensor) -> str:
    array = (
        values.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_mpr_cache_payload(
    value: object,
    *,
    expected_feature_space: str | None = None,
    require_reliability: bool = True,
    require_formal_safety: bool = False,
) -> dict[str, Any]:
    """Deeply validate one primitive multiview-reconstruction cache.

    Validation deliberately happens before any model allocation.  Large
    feature tensors are checked in row chunks so the validator does not create
    another full-size boolean tensor for official 4096-D targets.
    """

    if not isinstance(value, Mapping):
        raise ValueError("MPR cache must contain a mapping")
    payload = dict(value)
    xyz = _require_tensor(payload, "xyz")
    features = _require_tensor(payload, "features")
    valid = _require_tensor(payload, "valid")
    view_counts = _require_tensor(payload, "view_counts")
    if xyz.dtype != torch.float32 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("MPR xyz must be float32 [N,3]")
    if (
        features.dtype not in {torch.float16, torch.float32}
        or features.ndim != 2
    ):
        raise ValueError("MPR features must be float16/float32 [N,D]")
    num_rows = int(xyz.shape[0])
    feature_dim = int(features.shape[1]) if features.ndim == 2 else 0
    if num_rows <= 0 or feature_dim <= 0:
        raise ValueError("MPR cache must contain at least one row and channel")
    if num_rows > 20_000_000 or feature_dim > 16_384:
        raise ValueError("MPR cache dimensions exceed the formal safety bound")
    if features.shape[0] != num_rows:
        raise ValueError("MPR xyz and features do not align")
    if valid.dtype != torch.bool or valid.shape != (num_rows,):
        raise ValueError("MPR valid must be bool [N]")
    if view_counts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    } or view_counts.shape != (num_rows,):
        raise ValueError("MPR view_counts must be an integer tensor [N]")
    if not _all_finite(xyz) or not _all_finite(features):
        raise ValueError("MPR xyz/features contain non-finite values")
    counts = view_counts.to(device="cpu", dtype=torch.int64)
    valid_cpu = valid.to(device="cpu")
    if bool((counts < 0).any()) or not torch.equal(valid_cpu, counts > 0):
        raise ValueError("MPR valid mask must equal view_counts > 0")
    if not _all_zero_on_mask(features.detach().cpu(), ~valid_cpu):
        raise ValueError("MPR unsupported feature rows must be exactly zero")

    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise ValueError("MPR cache lacks metadata")
    metadata = dict(metadata_value)
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("MPR metadata must use schema version 1")
    feature_space = str(metadata.get("feature_space", ""))
    if expected_feature_space is not None and feature_space != str(
        expected_feature_space
    ):
        raise ValueError(
            f"expected {expected_feature_space!r} MPR, got {feature_space!r}"
        )
    num_views = int(metadata.get("num_declared_views", 0))
    if num_views <= 0 or bool((counts > num_views).any()):
        raise ValueError("MPR view counts exceed the declared source views")
    selected_frames = metadata.get("selected_frame_indices")
    if (
        not isinstance(selected_frames, list)
        or len(selected_frames) != num_views
        or len({int(item) for item in selected_frames}) != num_views
    ):
        raise ValueError("MPR selected-frame declaration is incomplete or repeated")

    geometry = payload.get("geometry_fingerprint")
    if not isinstance(geometry, Mapping):
        raise ValueError("MPR cache lacks a geometry fingerprint")
    if (
        int(geometry.get("num_gaussians", -1)) != num_rows
        or geometry.get("xyz_sha256") != _xyz_sha256(xyz)
        or metadata.get("xyz_sha256") != geometry.get("xyz_sha256")
    ):
        raise ValueError("MPR geometry fingerprint differs from xyz")

    reliability_value = payload.get("reliability")
    if reliability_value is None and require_reliability:
        raise ValueError("MPR cache lacks reliability")
    if reliability_value is not None:
        if not torch.is_tensor(reliability_value):
            raise ValueError("MPR reliability must be a tensor")
        reliability = reliability_value
        if (
            reliability.dtype not in {torch.float16, torch.float32}
            or reliability.shape != (num_rows, 3)
            or not _all_finite(reliability)
        ):
            raise ValueError("MPR reliability must be finite float [N,3]")
        reliability_cpu = reliability.detach().float().cpu()
        if bool((reliability_cpu < 0).any()) or bool(
            (reliability_cpu > 1.001).any()
        ):
            raise ValueError("MPR reliability must lie in [0,1]")
        tolerance = 2e-3 if reliability.dtype == torch.float16 else 1e-6
        if not torch.allclose(
            reliability_cpu[:, 2], valid_cpu.float(), atol=tolerance, rtol=0
        ):
            raise ValueError("MPR reliability support channel differs from valid")
        expected_coverage = counts.float() / float(num_views)
        if not torch.allclose(
            reliability_cpu[:, 0], expected_coverage, atol=tolerance, rtol=0
        ):
            raise ValueError("MPR reliability coverage differs from view_counts")
        if not _all_zero_on_mask(reliability_cpu, ~valid_cpu):
            raise ValueError("MPR unsupported reliability rows must be zero")
        reliability_mode = str(
            metadata.get("raster_reliability_mode", "legacy_valid")
        )
        if reliability_mode == "legacy_valid" and not torch.allclose(
            reliability_cpu[:, 1], valid_cpu.float(), atol=tolerance, rtol=0
        ):
            raise ValueError("legacy MPR agreement must equal valid")
        if reliability_mode == "mean_resultant":
            for start in range(0, num_rows, 4096):
                stop = min(start + 4096, num_rows)
                agreement = (
                    features[start:stop]
                    .detach()
                    .float()
                    .cpu()
                    .norm(dim=-1)
                    .clamp(0, 1)
                )
                agreement[~valid_cpu[start:stop]] = 0
                if not torch.allclose(
                    reliability_cpu[start:stop, 1],
                    agreement,
                    atol=4e-3,
                    rtol=2e-3,
                ):
                    raise ValueError(
                        "mean-resultant reliability differs from feature agreement"
                    )
        elif reliability_mode != "legacy_valid":
            raise ValueError("MPR reliability mode is unsupported")

    for key in (
        "benchmark_masks_opened",
        "benchmark_images_opened",
        "text_queries_opened",
    ):
        if require_formal_safety and metadata.get(key) is not False:
            raise ValueError(f"formal MPR safety declaration differs: {key}")
        if bool(metadata.get(key, False)):
            raise ValueError(f"MPR cache is contaminated: {key}")

    bundle_sha256 = str(
        metadata.get(
            "feature_output_bundle_sha256",
            metadata.get("capability_native_map_output_bundle_sha256", ""),
        )
    )
    if require_formal_safety and _SHA256.fullmatch(bundle_sha256) is None:
        raise ValueError("formal MPR is not bound to a feature output bundle")
    responsibility_sha256 = str(
        metadata.get("registration_responsibility_cache_sha256", "")
    )
    if require_formal_safety and (
        metadata.get("aggregation_mode") == "raster_gaussian_top1"
        and (
            metadata.get("shared_registration_responsibility") is not True
            or _SHA256.fullmatch(responsibility_sha256) is None
        )
    ):
        raise ValueError("formal raster MPR lacks shared responsibility provenance")
    return payload


def load_training_tensor_cache(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    purpose: str = "tensor_cache",
    expected_sha256: str | None = None,
) -> Any:
    """Load a tensor cache via one stable descriptor and no pickle fallback."""

    value, _, _ = load_torch_payload(
        path,
        expected_sha256=expected_sha256,
        map_location=map_location,
        label=purpose,
    )
    return value


def _is_json_artifact(path: str | Path) -> bool:
    value, _digest, _source = stable_descriptor_load(
        path,
        lambda handle: handle.read(64),
        label="MPR cache format probe",
    )
    return bytes(value).lstrip().startswith(b"{")


def _validate_sharded_support(
    support: object,
    *,
    metadata: dict[str, Any],
    geometry: dict[str, Any],
    require_reliability: bool,
    require_formal_safety: bool,
) -> dict[str, torch.Tensor]:
    if not isinstance(support, Mapping):
        raise ValueError("sharded MPR support must contain a mapping")
    required = {"xyz", "valid", "view_counts", "reliability"}
    if set(support) != required:
        raise ValueError("sharded MPR support tensor declaration differs")
    xyz = torch.as_tensor(support["xyz"])
    valid = torch.as_tensor(support["valid"])
    counts = torch.as_tensor(support["view_counts"])
    reliability = torch.as_tensor(support["reliability"])
    if xyz.dtype != torch.float32 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("sharded MPR xyz must be float32 [N,3]")
    num_rows = int(xyz.shape[0])
    if num_rows <= 0 or num_rows > 20_000_000 or not _all_finite(xyz):
        raise ValueError("sharded MPR xyz is invalid")
    if valid.dtype != torch.bool or valid.shape != (num_rows,):
        raise ValueError("sharded MPR valid must be bool [N]")
    if counts.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    } or counts.shape != (num_rows,):
        raise ValueError("sharded MPR view_counts must be integer [N]")
    counts64 = counts.long().cpu()
    valid_cpu = valid.bool().cpu()
    if bool((counts64 < 0).any()) or not torch.equal(valid_cpu, counts64 > 0):
        raise ValueError("sharded MPR valid mask must equal view_counts > 0")
    num_views = int(metadata.get("num_declared_views", 0))
    frames = metadata.get("selected_frame_indices")
    if (
        num_views <= 0
        or bool((counts64 > num_views).any())
        or not isinstance(frames, list)
        or len(frames) != num_views
        or len({int(value) for value in frames}) != num_views
    ):
        raise ValueError("sharded MPR declared views are invalid")
    if require_reliability and reliability.numel() == 0:
        raise ValueError("sharded MPR cache lacks reliability")
    if (
        reliability.dtype not in {torch.float16, torch.float32}
        or reliability.shape != (num_rows, 3)
        or not _all_finite(reliability)
    ):
        raise ValueError("sharded MPR reliability must be finite float [N,3]")
    reliability32 = reliability.float().cpu()
    tolerance = 2e-3 if reliability.dtype == torch.float16 else 1e-6
    if bool((reliability32 < 0).any()) or bool((reliability32 > 1.001).any()):
        raise ValueError("sharded MPR reliability must lie in [0,1]")
    if not torch.allclose(
        reliability32[:, 0],
        counts64.float() / float(num_views),
        atol=tolerance,
        rtol=0,
    ):
        raise ValueError("sharded MPR reliability coverage differs from view_counts")
    if not torch.allclose(
        reliability32[:, 2], valid_cpu.float(), atol=tolerance, rtol=0
    ):
        raise ValueError("sharded MPR reliability support differs from valid")
    if not _all_zero_on_mask(reliability32, ~valid_cpu):
        raise ValueError("sharded MPR unsupported reliability rows must be zero")
    if str(metadata.get("raster_reliability_mode", "legacy_valid")) == "legacy_valid":
        if not torch.allclose(
            reliability32[:, 1], valid_cpu.float(), atol=tolerance, rtol=0
        ):
            raise ValueError("legacy sharded MPR agreement must equal valid")
    elif str(metadata.get("raster_reliability_mode")) != "mean_resultant":
        raise ValueError("sharded MPR reliability mode is unsupported")
    xyz_digest = _xyz_sha256(xyz)
    if (
        int(geometry.get("num_gaussians", -1)) != num_rows
        or geometry.get("xyz_sha256") != xyz_digest
        or metadata.get("xyz_sha256") != xyz_digest
    ):
        raise ValueError("sharded MPR geometry fingerprint differs from xyz")
    for key in (
        "benchmark_masks_opened",
        "benchmark_images_opened",
        "text_queries_opened",
    ):
        if require_formal_safety and metadata.get(key) is not False:
            raise ValueError(f"formal sharded MPR safety declaration differs: {key}")
        if bool(metadata.get(key, False)):
            raise ValueError(f"sharded MPR cache is contaminated: {key}")
    bundle_sha256 = str(
        metadata.get(
            "feature_output_bundle_sha256",
            metadata.get("capability_native_map_output_bundle_sha256", ""),
        )
    )
    if require_formal_safety and _SHA256.fullmatch(bundle_sha256) is None:
        raise ValueError("formal sharded MPR lacks a feature output bundle")
    responsibility_sha256 = str(
        metadata.get("registration_responsibility_cache_sha256", "")
    )
    if require_formal_safety and (
        metadata.get("aggregation_mode") == "raster_gaussian_top1"
        and (
            metadata.get("shared_registration_responsibility") is not True
            or _SHA256.fullmatch(responsibility_sha256) is None
        )
    ):
        raise ValueError("formal sharded MPR lacks responsibility provenance")
    return {
        "xyz": xyz.cpu(),
        "valid": valid_cpu,
        "view_counts": counts64,
        "reliability": reliability.cpu(),
    }


def _load_sharded_mpr_cache(
    path: str | Path,
    *,
    expected_sha256: str | None,
    expected_feature_space: str | None,
    require_reliability: bool,
    require_formal_safety: bool,
) -> tuple[ShardedMPRCache, str, Path]:
    manifest, manifest_sha256, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="sharded MPR manifest",
    )
    if set(manifest) != {
        "schema",
        "schema_version",
        "layout",
        "feature_dtype",
        "feature_shape",
        "support",
        "shards",
        "geometry_fingerprint",
        "metadata",
    }:
        raise ValueError("sharded MPR manifest field declaration differs")
    if manifest.get("schema") != _SHARDED_MPR_SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != 1:
        raise ValueError("unsupported sharded MPR manifest schema")
    metadata_value = manifest.get("metadata")
    geometry_value = manifest.get("geometry_fingerprint")
    if not isinstance(metadata_value, Mapping) or not isinstance(
        geometry_value, Mapping
    ):
        raise ValueError("sharded MPR manifest lacks metadata or geometry")
    if set(geometry_value) != {"num_gaussians", "xyz_sha256"}:
        raise ValueError("sharded MPR geometry fingerprint declaration differs")
    metadata = dict(metadata_value)
    geometry = dict(geometry_value)
    feature_space = str(metadata.get("feature_space", ""))
    if expected_feature_space is not None and feature_space != str(
        expected_feature_space
    ):
        raise ValueError(
            f"expected {expected_feature_space!r} MPR, got {feature_space!r}"
        )
    shape = manifest.get("feature_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or int(shape[0]) <= 0
        or int(shape[0]) > 20_000_000
        or int(shape[1]) <= 0
        or int(shape[1]) > 16_384
        or manifest.get("feature_dtype") != "float16"
        or manifest.get("layout") != "row_major_channel_shards"
    ):
        raise ValueError("sharded MPR feature descriptor is invalid")
    root = source.parent
    mirror_root = _local_mirror_root()
    support_record = manifest.get("support")
    if not isinstance(support_record, Mapping) or set(support_record) != {
        "relative_path",
        "sha256",
    }:
        raise ValueError("sharded MPR support record is invalid")
    support_path = _safe_manifest_member(
        root, support_record["relative_path"], label="support"
    )
    support_identity = _path_identity(support_path)
    support_read_path = support_path
    if mirror_root is not None:
        support_read_path, _support_mirror_identity = _local_mirror_member(
            mirror_root,
            support_path,
            expected_bytes=support_identity[2],
            label="support",
        )
    support, support_sha256, _ = load_torch_payload(
        support_read_path,
        expected_sha256=str(support_record["sha256"]),
        map_location="cpu",
        label="sharded MPR support",
    )
    tensors = _validate_sharded_support(
        support,
        metadata=metadata,
        geometry=geometry,
        require_reliability=require_reliability,
        require_formal_safety=require_formal_safety,
    )
    num_rows, feature_dim = int(shape[0]), int(shape[1])
    if tensors["xyz"].shape[0] != num_rows:
        raise ValueError("sharded MPR support and feature rows do not align")
    records = manifest.get("shards")
    if not isinstance(records, list) or not records:
        raise ValueError("sharded MPR manifest has no channel shards")
    shards: list[_ChannelShard] = []
    expected_start = 0
    norm_squared = torch.zeros(num_rows, dtype=torch.float32)
    valid_cpu = tensors["valid"]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("sharded MPR shard record is invalid")
        if set(record) != {
            "relative_path",
            "sha256",
            "channel_start",
            "channel_stop",
            "dtype",
            "shape",
        }:
            raise ValueError("sharded MPR shard field declaration differs")
        start = int(record.get("channel_start", -1))
        stop = int(record.get("channel_stop", -1))
        if (
            start != expected_start
            or stop <= start
            or stop > feature_dim
            or record.get("dtype") != "float16"
            or record.get("shape") != [num_rows, stop - start]
            or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ValueError("sharded MPR channel coverage is invalid")
        shard_path = _safe_manifest_member(
            root, record.get("relative_path"), label=f"shard {index}"
        )
        expected_bytes = num_rows * (stop - start) * 2
        authority_identity = _path_identity(shard_path)
        if authority_identity[2] != expected_bytes:
            raise ValueError("sharded MPR shard byte count differs")
        mapped_path = shard_path
        identity = authority_identity
        if mirror_root is not None:
            mapped_path, identity = _local_mirror_member(
                mirror_root,
                shard_path,
                expected_bytes=expected_bytes,
                label=f"shard {index}",
            )
        digest = sha256_file(mapped_path)
        if digest != str(record["sha256"]):
            label = "local mirror shard" if mirror_root is not None else "shard"
            raise ValueError(f"sharded MPR {label} SHA-256 differs")
        mapped = np.memmap(
            mapped_path,
            mode="r",
            dtype="<f2",
            shape=(num_rows, stop - start),
            order="C",
        )
        for row_start in range(0, num_rows, 4096):
            row_stop = min(row_start + 4096, num_rows)
            values = torch.from_numpy(
                np.asarray(mapped[row_start:row_stop], dtype=np.float16).copy()
            )
            if not bool(torch.isfinite(values).all()):
                raise ValueError("sharded MPR features contain non-finite values")
            invalid = ~valid_cpu[row_start:row_stop]
            if bool(invalid.any()) and bool(values[invalid].ne(0).any()):
                raise ValueError("sharded MPR unsupported feature rows are nonzero")
            if str(metadata.get("raster_reliability_mode")) == "mean_resultant":
                norm_squared[row_start:row_stop].add_(
                    values.float().square().sum(dim=-1)
                )
        del mapped
        shards.append(
            _ChannelShard(
                shard_path,
                mapped_path,
                digest,
                start,
                stop,
                identity,
            )
        )
        expected_start = stop
    if expected_start != feature_dim:
        raise ValueError("sharded MPR channels are incomplete")
    if str(metadata.get("raster_reliability_mode")) == "mean_resultant":
        expected_agreement = norm_squared.sqrt().clamp(0, 1)
        expected_agreement[~valid_cpu] = 0
        if not torch.allclose(
            tensors["reliability"].float()[:, 1],
            expected_agreement,
            atol=4e-3,
            rtol=2e-3,
        ):
            raise ValueError("sharded MPR mean-resultant reliability differs")
    return (
        ShardedMPRCache(
            manifest_path=source,
            manifest_sha256=manifest_sha256,
            metadata=metadata,
            geometry_fingerprint=geometry,
            xyz=tensors["xyz"],
            valid=tensors["valid"],
            view_counts=tensors["view_counts"],
            reliability=tensors["reliability"],
            feature_dim=feature_dim,
            shards=shards,
            support_record={
                "path": str(support_path),
                "sha256": support_sha256,
            },
        ),
        manifest_sha256,
        source,
    )


def load_mpr_cache(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_feature_space: str | None = None,
    require_reliability: bool = True,
    require_formal_safety: bool = False,
) -> tuple[dict[str, Any] | ShardedMPRCache, str, Path]:
    """Safely load, content-bind, and deeply validate one MPR cache."""

    if _is_json_artifact(path):
        return _load_sharded_mpr_cache(
            path,
            expected_sha256=expected_sha256,
            expected_feature_space=expected_feature_space,
            require_reliability=require_reliability,
            require_formal_safety=require_formal_safety,
        )
    value, digest, source = load_torch_payload(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="MPR cache",
    )
    payload = validate_mpr_cache_payload(
        value,
        expected_feature_space=expected_feature_space,
        require_reliability=require_reliability,
        require_formal_safety=require_formal_safety,
    )
    return payload, digest, source
