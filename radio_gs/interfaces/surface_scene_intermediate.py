"""Strict, candidate-independent SurfaceRegion scene intermediates.

This artifact contains only scene quantities that are identical across the
SurfaceRegion capacity candidates.  Region expansion, candidate reliability
semantics, and teacher targets deliberately remain outside the contract.

The implementation is CPU-only.  Trusted loading always requires an external
contract and an external whole-file digest, uses one no-follow file descriptor
for hashing and deserialization, and rejects concurrent file mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Mapping

import torch

from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportGraphConfig,
)


SURFACE_SCENE_INTERMEDIATE_SCHEMA_VERSION = 1
SURFACE_SCENE_INTERMEDIATE_ARTIFACT_TYPE = (
    "surface-scene-intermediate-v1"
)
RADIO_FEATURE_DIMENSION = 1280
EXPECTED_EDGE_CHANNELS = frozenset(
    {"geometry", "appearance", "boundary"}
)
EXPECTED_ADAPTOR_ROLES = frozenset({"appearance", "boundary"})
EXPECTED_IMPLEMENTATION_ROLES = frozenset(
    {
        "scene_builder",
        "scene_intermediate_contract",
        "radio_runtime",
        "radio_adaptors",
        "feature_hash",
        "support_graph",
    }
)
GEOMETRIC_RELIABILITY_MODE = "geometric_mean_observation_agreement"
GEOMETRIC_RELIABILITY_ALGORITHM = (
    "voxel_coverage_one_minus_exp_neg_count_over_2_times_"
    "mean_shifted_cosine_sqrt_clamp_1e-6_v2"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _absolute_without_resolving(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} SHA-256 must be a string")
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite canonical JSON") from error
    return serialized.encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_exact_keys(
    value: object,
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{label} keys differ: missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    return value


def _open_regular_nofollow(path: Path) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("trusted artifacts require O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"refuse to follow a file symlink: {path}") from error
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"bound artifact is not a regular file: {path}")
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


def _path_stat_nofollow(path: Path) -> os.stat_result:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"bound path is not a regular file: {path}")
    return info


def _require_path_matches_descriptor(
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


def sha256_file(path: str | Path) -> str:
    """Hash one stable regular file through one no-follow descriptor."""

    source = _absolute_without_resolving(path)
    descriptor = _open_regular_nofollow(source)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        before = os.fstat(handle.fileno())
        path_before = _path_stat_nofollow(source)
        _require_path_matches_descriptor(
            path_before,
            before,
            label="hashed file",
        )
        digest = _hash_handle(handle)
        after = os.fstat(handle.fileno())
        path_after = _path_stat_nofollow(source)
        if (
            _stat_fingerprint(before) != _stat_fingerprint(after)
            or _stat_fingerprint(path_before) != _stat_fingerprint(path_after)
        ):
            raise ValueError("hashed file changed while being read")
        _require_path_matches_descriptor(
            path_after,
            after,
            label="hashed file",
        )
        return digest


@dataclass(frozen=True)
class SourceFileBinding:
    """An absolute regular-file path and the digest of its exact bytes."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError("source file path must be a string")
        path = _absolute_without_resolving(self.path)
        object.__setattr__(self, "path", str(path))
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, label=f"source file {path}"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SourceFileBinding":
        resolved = Path(path).expanduser().resolve(strict=True)
        return cls(path=str(resolved), sha256=sha256_file(resolved))

    @classmethod
    def from_dict(cls, payload: object) -> "SourceFileBinding":
        value = _require_exact_keys(
            payload,
            {"path", "sha256"},
            label="source file binding",
        )
        return cls(path=value["path"], sha256=value["sha256"])

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    def verify(self) -> None:
        if sha256_file(self.path) != self.sha256:
            raise ValueError(f"bound source file changed: {self.path}")


@dataclass(frozen=True)
class SurfaceSceneFrameBinding:
    """One ordered RGB-D-pose source observation."""

    frame_id: str
    color: SourceFileBinding
    depth: SourceFileBinding
    pose: SourceFileBinding

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id must be a string")
        frame_id = self.frame_id.strip()
        if not frame_id:
            raise ValueError("frame_id cannot be empty")
        object.__setattr__(self, "frame_id", frame_id)
        for label in ("color", "depth", "pose"):
            if not isinstance(getattr(self, label), SourceFileBinding):
                raise TypeError(f"frame {label} must be a SourceFileBinding")

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceSceneFrameBinding":
        value = _require_exact_keys(
            payload,
            {"frame_id", "color", "depth", "pose"},
            label="source frame binding",
        )
        return cls(
            frame_id=value["frame_id"],
            color=SourceFileBinding.from_dict(value["color"]),
            depth=SourceFileBinding.from_dict(value["depth"]),
            pose=SourceFileBinding.from_dict(value["pose"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "color": self.color.to_dict(),
            "depth": self.depth.to_dict(),
            "pose": self.pose.to_dict(),
        }

    def verify(self) -> None:
        self.color.verify()
        self.depth.verify()
        self.pose.verify()


def _canonical_graph_config(
    raw_config: Mapping[str, object],
) -> dict[str, object]:
    if not all(isinstance(key, str) for key in raw_config):
        raise TypeError("graph_config keys must be strings")
    graph_config = dict(raw_config)
    defaults = asdict(SupportGraphConfig())
    if set(graph_config) != set(defaults):
        raise ValueError(
            "graph_config must contain every SupportGraphConfig field"
        )
    canonical: dict[str, object] = {}
    for name, default in defaults.items():
        value = graph_config[name]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError(f"graph_config {name} must be boolean")
            canonical[name] = value
        elif isinstance(default, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"graph_config {name} must be an integer")
            canonical[name] = value
        elif isinstance(default, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"graph_config {name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"graph_config {name} must be finite")
            canonical[name] = numeric
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise TypeError(f"graph_config {name} must be a string")
            canonical[name] = value
        else:
            raise TypeError(f"unsupported graph_config field: {name}")
    if canonical["topology_mode"] != "symmetric_union":
        raise ValueError(
            "scene intermediate graph topology_mode must be symmetric_union"
        )
    if canonical["surface_topology_min_affinity"] != 0.0:
        raise ValueError(
            "scene intermediate surface_topology_min_affinity must be zero"
        )
    if canonical["require_covisibility_topology"] is not False:
        raise ValueError(
            "scene intermediate require_covisibility_topology must be false"
        )
    try:
        SupportGraphConfig(**canonical)
    except (TypeError, ValueError) as error:
        raise ValueError("graph_config is invalid") from error
    _canonical_json_bytes(canonical)
    return dict(sorted(canonical.items()))


@dataclass(frozen=True)
class SurfaceSceneIntermediateContract:
    """Candidate-independent authority required to reuse one scene."""

    scene: str
    source_frames: tuple[SurfaceSceneFrameBinding, ...]
    depth_intrinsic: SourceFileBinding
    color_intrinsic: SourceFileBinding
    radio_checkpoint: SourceFileBinding
    radio_version: str
    radio_resolution: int
    depth_stride: int
    voxel_size: float
    adaptor_names: Mapping[str, str]
    adaptor_batch_size: int
    affinity_dimension: int
    graph_config: Mapping[str, object]
    implementation_sources: Mapping[str, SourceFileBinding]
    geometric_reliability_mode: str = GEOMETRIC_RELIABILITY_MODE
    geometric_reliability_algorithm: str = GEOMETRIC_RELIABILITY_ALGORITHM

    def __post_init__(self) -> None:
        if not isinstance(self.scene, str) or not self.scene.strip():
            raise ValueError("scene must be a non-empty string")
        object.__setattr__(self, "scene", self.scene.strip())

        source_frames = tuple(self.source_frames)
        if not source_frames or not all(
            isinstance(frame, SurfaceSceneFrameBinding)
            for frame in source_frames
        ):
            raise TypeError(
                "source_frames must contain ordered frame bindings"
            )
        frame_ids = [frame.frame_id for frame in source_frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("source frame IDs must be unique")
        object.__setattr__(self, "source_frames", source_frames)

        for label in (
            "depth_intrinsic",
            "color_intrinsic",
            "radio_checkpoint",
        ):
            if not isinstance(getattr(self, label), SourceFileBinding):
                raise TypeError(f"{label} must be a SourceFileBinding")
        if not isinstance(self.radio_version, str) or not self.radio_version.strip():
            raise ValueError("radio_version must be a non-empty string")
        object.__setattr__(self, "radio_version", self.radio_version.strip())

        for label in (
            "radio_resolution",
            "depth_stride",
            "adaptor_batch_size",
            "affinity_dimension",
        ):
            value = getattr(self, label)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not isinstance(self.voxel_size, (int, float)) or isinstance(
            self.voxel_size,
            bool,
        ):
            raise TypeError("voxel_size must be numeric")
        voxel_size = float(self.voxel_size)
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        object.__setattr__(self, "voxel_size", voxel_size)

        adaptor_names = _require_exact_keys(
            self.adaptor_names,
            EXPECTED_ADAPTOR_ROLES,
            label="adaptor role mapping",
        )
        canonical_adaptors: dict[str, str] = {}
        for role, raw_name in adaptor_names.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"adaptor {role} name must be non-empty")
            canonical_adaptors[role] = raw_name.strip()
        if len(set(canonical_adaptors.values())) != len(canonical_adaptors):
            raise ValueError("appearance and boundary adaptors must differ")
        object.__setattr__(
            self,
            "adaptor_names",
            MappingProxyType(dict(sorted(canonical_adaptors.items()))),
        )

        if not isinstance(self.graph_config, Mapping):
            raise TypeError("graph_config must be a mapping")
        object.__setattr__(
            self,
            "graph_config",
            MappingProxyType(_canonical_graph_config(self.graph_config)),
        )

        implementations = _require_exact_keys(
            self.implementation_sources,
            EXPECTED_IMPLEMENTATION_ROLES,
            label="implementation source bindings",
        )
        canonical_implementations: dict[str, SourceFileBinding] = {}
        for role, binding in implementations.items():
            if not isinstance(binding, SourceFileBinding):
                raise TypeError(
                    f"implementation {role} must be a SourceFileBinding"
                )
            canonical_implementations[role] = binding
        object.__setattr__(
            self,
            "implementation_sources",
            MappingProxyType(dict(sorted(canonical_implementations.items()))),
        )

        if self.geometric_reliability_mode != GEOMETRIC_RELIABILITY_MODE:
            raise ValueError("geometric reliability mode differs")
        if (
            self.geometric_reliability_algorithm
            != GEOMETRIC_RELIABILITY_ALGORITHM
        ):
            raise ValueError("geometric reliability algorithm differs")

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceSceneIntermediateContract":
        value = _require_exact_keys(
            payload,
            {
                "scene",
                "source_frames",
                "intrinsics",
                "radio",
                "lifting",
                "adaptors",
                "graph_config",
                "implementation_sources",
                "geometric_reliability",
                "radio_feature_dimension",
            },
            label="scene intermediate contract",
        )
        if value["radio_feature_dimension"] != RADIO_FEATURE_DIMENSION:
            raise ValueError("scene intermediate RADIO feature dimension differs")
        intrinsics = _require_exact_keys(
            value["intrinsics"],
            {"depth", "color"},
            label="intrinsic bindings",
        )
        radio = _require_exact_keys(
            value["radio"],
            {"checkpoint", "version", "resolution"},
            label="RADIO contract",
        )
        lifting = _require_exact_keys(
            value["lifting"],
            {"depth_stride", "voxel_size"},
            label="lifting contract",
        )
        adaptors = _require_exact_keys(
            value["adaptors"],
            {"role_to_name", "batch_size", "affinity_dimension"},
            label="adaptor contract",
        )
        reliability = _require_exact_keys(
            value["geometric_reliability"],
            {"mode", "algorithm"},
            label="geometric reliability contract",
        )
        raw_frames = value["source_frames"]
        if not isinstance(raw_frames, (list, tuple)):
            raise TypeError("source_frames must be a sequence")
        raw_implementations = _require_exact_keys(
            value["implementation_sources"],
            EXPECTED_IMPLEMENTATION_ROLES,
            label="implementation source bindings",
        )
        return cls(
            scene=value["scene"],
            source_frames=tuple(
                SurfaceSceneFrameBinding.from_dict(frame)
                for frame in raw_frames
            ),
            depth_intrinsic=SourceFileBinding.from_dict(intrinsics["depth"]),
            color_intrinsic=SourceFileBinding.from_dict(intrinsics["color"]),
            radio_checkpoint=SourceFileBinding.from_dict(radio["checkpoint"]),
            radio_version=radio["version"],
            radio_resolution=radio["resolution"],
            depth_stride=lifting["depth_stride"],
            voxel_size=lifting["voxel_size"],
            adaptor_names=adaptors["role_to_name"],
            adaptor_batch_size=adaptors["batch_size"],
            affinity_dimension=adaptors["affinity_dimension"],
            graph_config=value["graph_config"],
            implementation_sources={
                role: SourceFileBinding.from_dict(binding)
                for role, binding in raw_implementations.items()
            },
            geometric_reliability_mode=reliability["mode"],
            geometric_reliability_algorithm=reliability["algorithm"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "source_frames": [frame.to_dict() for frame in self.source_frames],
            "intrinsics": {
                "depth": self.depth_intrinsic.to_dict(),
                "color": self.color_intrinsic.to_dict(),
            },
            "radio": {
                "checkpoint": self.radio_checkpoint.to_dict(),
                "version": self.radio_version,
                "resolution": self.radio_resolution,
            },
            "lifting": {
                "depth_stride": self.depth_stride,
                "voxel_size": self.voxel_size,
            },
            "adaptors": {
                "role_to_name": dict(self.adaptor_names),
                "batch_size": self.adaptor_batch_size,
                "affinity_dimension": self.affinity_dimension,
            },
            "graph_config": dict(self.graph_config),
            "implementation_sources": {
                role: binding.to_dict()
                for role, binding in self.implementation_sources.items()
            },
            "geometric_reliability": {
                "mode": self.geometric_reliability_mode,
                "algorithm": self.geometric_reliability_algorithm,
            },
            "radio_feature_dimension": RADIO_FEATURE_DIMENSION,
        }

    @property
    def digest(self) -> str:
        return _json_sha256(self.to_dict())

    def verify_source_files(self) -> None:
        for frame in self.source_frames:
            frame.verify()
        self.depth_intrinsic.verify()
        self.color_intrinsic.verify()
        self.radio_checkpoint.verify()
        for binding in self.implementation_sources.values():
            binding.verify()


def _require_tensor(
    value: object,
    *,
    label: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a tensor")
    tensor = value
    if tensor.device.type != "cpu":
        raise ValueError(f"{label} must be a CPU tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    if tensor.ndim != len(shape) or any(
        expected is not None and tensor.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise ValueError(f"{label} has an invalid shape: {tuple(tensor.shape)}")
    return tensor


def _require_finite(tensor: torch.Tensor, *, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must contain only finite values")


def _require_unit_interval(tensor: torch.Tensor, *, label: str) -> None:
    _require_finite(tensor, label=label)
    if bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError(f"{label} must lie in [0,1]")


def _reverse_edge_indices(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    edge_count = int(edge_index.shape[1])
    source, destination = edge_index
    if bool((source == destination).any()):
        raise ValueError("graph cannot contain self edges")
    codes = source * int(num_nodes) + destination
    if int(torch.unique(codes).numel()) != edge_count:
        raise ValueError("graph contains duplicate directed edges")
    sorted_codes, order = torch.sort(codes)
    reverse_codes = destination * int(num_nodes) + source
    positions = torch.searchsorted(sorted_codes, reverse_codes)
    if bool((positions >= edge_count).any()):
        raise ValueError("graph edge lacks a unique reverse edge")
    if not torch.equal(sorted_codes[positions], reverse_codes):
        raise ValueError("graph edge lacks a unique reverse edge")
    return order[positions]


def _validate_graph(
    graph: PrimitiveSupportGraph,
    *,
    num_nodes: int,
) -> None:
    if not isinstance(graph, PrimitiveSupportGraph):
        raise TypeError("graph must be a PrimitiveSupportGraph")
    if (
        not isinstance(graph.num_nodes, int)
        or isinstance(graph.num_nodes, bool)
        or graph.num_nodes <= 0
    ):
        raise ValueError("graph num_nodes must be a positive integer")
    if graph.num_nodes != num_nodes:
        raise ValueError("graph num_nodes differs from scene tensors")

    edge_index = _require_tensor(
        graph.edge_index,
        label="graph.edge_index",
        dtype=torch.int64,
        shape=(2, None),
    )
    edge_count = int(edge_index.shape[1])
    edge_weight = _require_tensor(
        graph.edge_weight,
        label="graph.edge_weight",
        dtype=torch.float32,
        shape=(edge_count,),
    )
    raw_affinity = _require_tensor(
        graph.raw_affinity,
        label="graph.raw_affinity",
        dtype=torch.float32,
        shape=(edge_count,),
    )
    local_sigma = _require_tensor(
        graph.local_sigma,
        label="graph.local_sigma",
        dtype=torch.float32,
        shape=(num_nodes,),
    )
    _require_unit_interval(edge_weight, label="graph.edge_weight")
    _require_unit_interval(raw_affinity, label="graph.raw_affinity")
    _require_finite(local_sigma, label="graph.local_sigma")
    if bool((local_sigma <= 0).any()):
        raise ValueError("graph.local_sigma must be strictly positive")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("graph.edge_index is outside scene rows")

    channels = _require_exact_keys(
        graph.edge_channels,
        EXPECTED_EDGE_CHANNELS,
        label="graph edge channels",
    )
    canonical_channels: dict[str, torch.Tensor] = {}
    for name, raw_channel in channels.items():
        channel = _require_tensor(
            raw_channel,
            label=f"graph.edge_channels.{name}",
            dtype=torch.float32,
            shape=(edge_count,),
        )
        _require_unit_interval(
            channel,
            label=f"graph.edge_channels.{name}",
        )
        canonical_channels[name] = channel

    if num_nodes == 1:
        if edge_count != 0:
            raise ValueError("a singleton scene graph cannot contain edges")
        return
    if edge_count == 0:
        raise ValueError("a multi-node scene graph cannot be empty")
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    if bool((out_degree == 0).any()):
        raise ValueError("every graph node must have an outgoing edge")

    reverse = _reverse_edge_indices(edge_index, num_nodes=num_nodes)
    if not torch.equal(raw_affinity, raw_affinity[reverse]):
        raise ValueError("reverse raw affinities must be exactly equal")
    for name, channel in canonical_channels.items():
        if not torch.equal(channel, channel[reverse]):
            raise ValueError(
                f"reverse {name} affinities must be exactly equal"
            )

    row_sum = torch.zeros(num_nodes, dtype=torch.float32)
    row_sum.index_add_(0, edge_index[0], raw_affinity)
    expected_weight = raw_affinity / row_sum[edge_index[0]].clamp_min(1e-12)
    if not torch.equal(edge_weight, expected_weight):
        raise ValueError("graph.edge_weight must equal raw affinity / row sum")


@dataclass(frozen=True)
class SurfaceSceneIntermediate:
    """Validated scientific tensors shared by all scene candidates."""

    contract: SurfaceSceneIntermediateContract
    xyz: torch.Tensor
    radio_features: torch.Tensor
    geometric_reliability: torch.Tensor
    graph: PrimitiveSupportGraph

    def __post_init__(self) -> None:
        if not isinstance(self.contract, SurfaceSceneIntermediateContract):
            raise TypeError(
                "contract must be a SurfaceSceneIntermediateContract"
            )
        xyz = _require_tensor(
            self.xyz,
            label="xyz",
            dtype=torch.float32,
            shape=(None, 3),
        )
        num_nodes = int(xyz.shape[0])
        if num_nodes <= 0:
            raise ValueError("scene intermediate cannot be empty")
        features = _require_tensor(
            self.radio_features,
            label="radio_features",
            dtype=torch.float32,
            shape=(num_nodes, RADIO_FEATURE_DIMENSION),
        )
        reliability = _require_tensor(
            self.geometric_reliability,
            label="geometric_reliability",
            dtype=torch.float32,
            shape=(num_nodes,),
        )
        for label, tensor in (
            ("xyz", xyz),
            ("radio_features", features),
            ("geometric_reliability", reliability),
        ):
            _require_finite(tensor, label=label)
        norms = torch.linalg.vector_norm(features, dim=1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("radio_features must be row-wise L2 normalized")
        _require_unit_interval(
            reliability,
            label="geometric_reliability",
        )
        _validate_graph(self.graph, num_nodes=num_nodes)


def scientific_tensors(
    value: SurfaceSceneIntermediate,
) -> dict[str, torch.Tensor]:
    """Return every tensor covered by the scientific bundle digest."""

    if not isinstance(value, SurfaceSceneIntermediate):
        raise TypeError("value must be a SurfaceSceneIntermediate")
    result = {
        "xyz": value.xyz,
        "radio_features": value.radio_features,
        "geometric_reliability": value.geometric_reliability,
        "graph.edge_index": value.graph.edge_index,
        "graph.edge_weight": value.graph.edge_weight,
        "graph.raw_affinity": value.graph.raw_affinity,
        "graph.local_sigma": value.graph.local_sigma,
    }
    for name, tensor in sorted(value.graph.edge_channels.items()):
        result[f"graph.edge_channels.{name}"] = tensor
    return result


def tensor_sha256(value: torch.Tensor) -> str:
    """Digest a tensor's dtype, shape, and contiguous CPU bytes."""

    if not torch.is_tensor(value):
        raise TypeError("tensor_sha256 expects a tensor")
    if value.device.type != "cpu":
        raise ValueError("tensor_sha256 only accepts CPU tensors")
    tensor = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def scientific_tensor_sha256(
    value: SurfaceSceneIntermediate,
) -> dict[str, str]:
    return {
        name: tensor_sha256(tensor)
        for name, tensor in sorted(scientific_tensors(value).items())
    }


def scientific_tensor_bundle_sha256(
    value: SurfaceSceneIntermediate,
) -> str:
    return _json_sha256(scientific_tensor_sha256(value))


@dataclass(frozen=True)
class SurfaceSceneIntermediateArtifact:
    """Content-addressed result of an atomic artifact save."""

    path: str
    file_sha256: str
    contract_sha256: str
    tensor_bundle_sha256: str
    tensor_sha256: Mapping[str, str]


def _artifact_payload(value: SurfaceSceneIntermediate) -> dict[str, object]:
    tensor_digests = scientific_tensor_sha256(value)
    return {
        "artifact_type": SURFACE_SCENE_INTERMEDIATE_ARTIFACT_TYPE,
        "schema_version": SURFACE_SCENE_INTERMEDIATE_SCHEMA_VERSION,
        "contract": value.contract.to_dict(),
        "contract_sha256": value.contract.digest,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": _json_sha256(tensor_digests),
        "tensors": {
            "xyz": value.xyz,
            "radio_features": value.radio_features,
            "geometric_reliability": value.geometric_reliability,
            "graph": {
                "edge_index": value.graph.edge_index,
                "edge_weight": value.graph.edge_weight,
                "raw_affinity": value.graph.raw_affinity,
                "local_sigma": value.graph.local_sigma,
                "num_nodes": value.graph.num_nodes,
                "edge_channels": dict(value.graph.edge_channels),
            },
        },
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_surface_scene_intermediate(
    value: SurfaceSceneIntermediate,
    path: str | Path,
    *,
    overwrite: bool = False,
    verify_source_files: bool = True,
) -> SurfaceSceneIntermediateArtifact:
    """Validate and atomically save one immutable scene intermediate."""

    if not isinstance(value, SurfaceSceneIntermediate):
        raise TypeError("value must be a SurfaceSceneIntermediate")
    SurfaceSceneIntermediate(
        contract=value.contract,
        xyz=value.xyz,
        radio_features=value.radio_features,
        geometric_reliability=value.geometric_reliability,
        graph=value.graph,
    )
    if verify_source_files:
        value.contract.verify_source_files()
    payload = _artifact_payload(value)

    output = _absolute_without_resolving(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and os.path.lexists(output):
        raise FileExistsError(f"scene intermediate already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise FileExistsError(
                    f"scene intermediate already exists: {output}"
                ) from error
            temporary.unlink()
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)

    tensor_digests = dict(payload["tensor_sha256"])
    return SurfaceSceneIntermediateArtifact(
        path=str(output),
        file_sha256=sha256_file(output),
        contract_sha256=value.contract.digest,
        tensor_bundle_sha256=_json_sha256(tensor_digests),
        tensor_sha256=MappingProxyType(tensor_digests),
    )


def _torch_load_from_handle(handle: BinaryIO) -> object:
    try:
        return torch.load(handle, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "scene intermediate loading requires torch.load(weights_only=True)"
        ) from error


def _trusted_payload_from_single_descriptor(
    path: Path,
    *,
    expected_file_sha256: str,
) -> object:
    descriptor = _open_regular_nofollow(path)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        before = os.fstat(handle.fileno())
        path_before = _path_stat_nofollow(path)
        _require_path_matches_descriptor(
            path_before,
            before,
            label="scene intermediate",
        )

        first_digest = _hash_handle(handle)
        if first_digest != expected_file_sha256:
            raise ValueError("scene intermediate file SHA-256 differs")
        handle.seek(0)
        payload = _torch_load_from_handle(handle)

        after_load = os.fstat(handle.fileno())
        handle.seek(0)
        second_digest = _hash_handle(handle)
        after_rehash = os.fstat(handle.fileno())
        path_after = _path_stat_nofollow(path)

        fingerprints = {
            _stat_fingerprint(before),
            _stat_fingerprint(after_load),
            _stat_fingerprint(after_rehash),
            _stat_fingerprint(path_before),
            _stat_fingerprint(path_after),
        }
        if len(fingerprints) != 1:
            raise ValueError("scene intermediate changed during trusted load")
        _require_path_matches_descriptor(
            path_after,
            after_rehash,
            label="scene intermediate",
        )
        if second_digest != first_digest:
            raise ValueError("scene intermediate digest changed during trusted load")
        return payload


def load_surface_scene_intermediate(
    path: str | Path,
    *,
    expected_contract: SurfaceSceneIntermediateContract,
    expected_file_sha256: str,
    verify_source_files: bool = True,
) -> SurfaceSceneIntermediate:
    """Trusted load requiring independent contract and whole-file authority."""

    if not isinstance(expected_contract, SurfaceSceneIntermediateContract):
        raise TypeError(
            "expected_contract must be a SurfaceSceneIntermediateContract"
        )
    expected_file_digest = _require_sha256(
        expected_file_sha256,
        label="expected scene intermediate file",
    )
    source = _absolute_without_resolving(path)
    payload = _require_exact_keys(
        _trusted_payload_from_single_descriptor(
            source,
            expected_file_sha256=expected_file_digest,
        ),
        {
            "artifact_type",
            "schema_version",
            "contract",
            "contract_sha256",
            "tensor_sha256",
            "tensor_bundle_sha256",
            "tensors",
        },
        label="scene intermediate artifact",
    )
    if payload["artifact_type"] != SURFACE_SCENE_INTERMEDIATE_ARTIFACT_TYPE:
        raise ValueError("scene intermediate artifact type differs")
    if (
        not isinstance(payload["schema_version"], int)
        or isinstance(payload["schema_version"], bool)
        or payload["schema_version"]
        != SURFACE_SCENE_INTERMEDIATE_SCHEMA_VERSION
    ):
        raise ValueError("scene intermediate schema version differs")

    contract = SurfaceSceneIntermediateContract.from_dict(payload["contract"])
    stored_contract_digest = _require_sha256(
        payload["contract_sha256"],
        label="stored scene contract",
    )
    if stored_contract_digest != contract.digest:
        raise ValueError("scene intermediate contract digest differs")
    if (
        contract.digest != expected_contract.digest
        or contract.to_dict() != expected_contract.to_dict()
    ):
        raise ValueError("scene intermediate does not match expected contract")
    if verify_source_files:
        contract.verify_source_files()

    tensors = _require_exact_keys(
        payload["tensors"],
        {"xyz", "radio_features", "geometric_reliability", "graph"},
        label="scene intermediate tensors",
    )
    graph_payload = _require_exact_keys(
        tensors["graph"],
        {
            "edge_index",
            "edge_weight",
            "raw_affinity",
            "local_sigma",
            "num_nodes",
            "edge_channels",
        },
        label="scene intermediate graph",
    )
    raw_num_nodes = graph_payload["num_nodes"]
    if (
        not isinstance(raw_num_nodes, int)
        or isinstance(raw_num_nodes, bool)
        or raw_num_nodes <= 0
    ):
        raise ValueError("scene intermediate graph num_nodes is invalid")
    raw_edge_index = _require_tensor(
        graph_payload["edge_index"],
        label="graph.edge_index",
        dtype=torch.int64,
        shape=(2, None),
    )
    raw_edge_count = int(raw_edge_index.shape[1])
    for label in ("edge_weight", "raw_affinity"):
        _require_tensor(
            graph_payload[label],
            label=f"graph.{label}",
            dtype=torch.float32,
            shape=(raw_edge_count,),
        )
    _require_tensor(
        graph_payload["local_sigma"],
        label="graph.local_sigma",
        dtype=torch.float32,
        shape=(raw_num_nodes,),
    )
    raw_channels = _require_exact_keys(
        graph_payload["edge_channels"],
        EXPECTED_EDGE_CHANNELS,
        label="graph edge channels",
    )
    for name, channel in raw_channels.items():
        _require_tensor(
            channel,
            label=f"graph.edge_channels.{name}",
            dtype=torch.float32,
            shape=(raw_edge_count,),
        )
    graph = PrimitiveSupportGraph(
        edge_index=graph_payload["edge_index"],
        edge_weight=graph_payload["edge_weight"],
        raw_affinity=graph_payload["raw_affinity"],
        local_sigma=graph_payload["local_sigma"],
        num_nodes=raw_num_nodes,
        edge_channels=raw_channels,
    )
    value = SurfaceSceneIntermediate(
        contract=contract,
        xyz=tensors["xyz"],
        radio_features=tensors["radio_features"],
        geometric_reliability=tensors["geometric_reliability"],
        graph=graph,
    )

    actual_tensor_digests = scientific_tensor_sha256(value)
    stored_tensor_digests = _require_exact_keys(
        payload["tensor_sha256"],
        set(actual_tensor_digests),
        label="stored tensor digests",
    )
    normalized_stored_digests = {
        name: _require_sha256(digest, label=f"tensor {name}")
        for name, digest in stored_tensor_digests.items()
    }
    if normalized_stored_digests != actual_tensor_digests:
        raise ValueError("scene intermediate tensor digest differs")
    stored_bundle_digest = _require_sha256(
        payload["tensor_bundle_sha256"],
        label="stored tensor bundle",
    )
    if stored_bundle_digest != _json_sha256(actual_tensor_digests):
        raise ValueError("scene intermediate tensor bundle digest differs")
    return value


def assert_exact_surface_scene_replay(
    fresh: SurfaceSceneIntermediate,
    replay: SurfaceSceneIntermediate,
) -> None:
    """Assert bit-exact scientific equality between fresh and replay paths."""

    if not isinstance(fresh, SurfaceSceneIntermediate) or not isinstance(
        replay,
        SurfaceSceneIntermediate,
    ):
        raise TypeError("fresh and replay must be scene intermediates")
    for value in (fresh, replay):
        SurfaceSceneIntermediate(
            contract=value.contract,
            xyz=value.xyz,
            radio_features=value.radio_features,
            geometric_reliability=value.geometric_reliability,
            graph=value.graph,
        )
    if (
        fresh.contract.digest != replay.contract.digest
        or fresh.contract.to_dict() != replay.contract.to_dict()
    ):
        raise AssertionError("fresh and replay contracts differ")
    fresh_tensors = scientific_tensors(fresh)
    replay_tensors = scientific_tensors(replay)
    if set(fresh_tensors) != set(replay_tensors):
        raise AssertionError("fresh and replay tensor sets differ")
    for name in sorted(fresh_tensors):
        if not torch.equal(fresh_tensors[name], replay_tensors[name]):
            raise AssertionError(f"fresh and replay tensor differs: {name}")


def default_graph_config_dict(
    config: SupportGraphConfig | None = None,
) -> dict[str, object]:
    """Return the complete JSON graph config expected by the contract."""

    value = SupportGraphConfig() if config is None else config
    if not isinstance(value, SupportGraphConfig):
        raise TypeError("config must be a SupportGraphConfig")
    return asdict(value)
