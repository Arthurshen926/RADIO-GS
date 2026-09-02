"""Cold-loadable, hash-bound scene state for the v4 surface memory.

This module deliberately owns the carrier configuration instead of accepting
projection defaults from an evaluator.  A saved bundle is sufficient to
reconstruct the exact surface carrier in a fresh process; query and benchmark
inputs are not part of the construction payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import operator
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radio_gs.v4.carrier import Camera, ProjectionTable, SurfaceVoxelCarrier
from radio_gs.v4.object_memory import DenseObjectAssignments

from .geometry_receipt import sha256_file


SCHEMA = "radio_gs.surface_object_memory_v4.surface_scene_bundle.v1"
GEOMETRY_SCHEMA = "radio_gs.surface_object_memory_v4.geometry_receipt.v1"
_SURFACE_INPUT_ROLES = frozenset({"surface_carrier", "fused_surface"})
_DIGEST_LENGTH = 64
SUPPORTED_CAMERA_CONVENTIONS = frozenset({
    "colmap_world_opencv_camera_feature_raster",
    "colmap_world_opencv_camera_pixel_centres",
    "mesh_oracle_to_sparse_surface_feature_raster",
    "nerf_opengl_to_opencv_camera_to_world_feature_raster",
    "nerf_opengl_to_opencv_camera_to_world_pixel_centres",
    "scannet_mesh_nerf_opengl_to_opencv_feature_raster",
})
COMPLETION_RECEIPT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.learned_completion_receipt.v1"
)
_FORBIDDEN_MEMORY_KEY_TERMS = ("label", "target", "ground_truth")
_PROBABILITY_TOLERANCE = 1e-6


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        return False
    if value.lower() == "0" * _DIGEST_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous().clone()
    return tensor


def _exact_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = _cpu_tensor(value)
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _update_content_digest(digest: Any, value: Any) -> None:
    """Hash JSON-like metadata and tensors without relying on torch serialization."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, torch.Tensor):
        tensor = _cpu_tensor(value)
        digest.update(b"T")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(_tensor_bytes(tensor))
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("content-digested mapping keys must be strings")
        digest.update(b"M")
        for key in sorted(value):
            _update_content_digest(digest, key)
            _update_content_digest(digest, value[key])
        digest.update(b"m")
    elif isinstance(value, (tuple, list)):
        digest.update(b"L")
        for item in value:
            _update_content_digest(digest, item)
        digest.update(b"l")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"B1" if bool(value) else b"B0")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, (int, np.integer)):
        encoded = str(int(value)).encode("ascii")
        digest.update(b"I" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("content-digested floating metadata must be finite")
        digest.update(b"F" + struct.pack(">d", number))
    else:
        raise ValueError(f"unsupported content-digested value type: {type(value).__name__}")


def _content_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_content_digest(digest, value)
    return digest.hexdigest()


def _tensor_digest(*values: torch.Tensor | None, configuration: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(configuration), sort_keys=True, separators=(",", ":")).encode())
    for value in values:
        if value is None:
            digest.update(b"<none>")
            continue
        tensor = _cpu_tensor(value)
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _validate_tensor_mapping(
    name: str,
    values: Mapping[str, torch.Tensor],
    *,
    expected_rows: int | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a tensor mapping")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    result: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        lowered = key.lower()
        if any(term in lowered for term in _FORBIDDEN_MEMORY_KEY_TERMS):
            raise ValueError(f"{name}.{key} is a forbidden label/target field")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name}.{key} must be a tensor")
        tensor = _cpu_tensor(value)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name}.{key} must be finite")
        if expected_rows is not None and (tensor.ndim == 0 or tensor.shape[0] != expected_rows):
            raise ValueError(f"{name}.{key} must have one row per surface element")
        result[key] = tensor
    return result


@dataclass(frozen=True)
class ElementTokenObservedEvidence:
    """Explicit positive/negative/unknown facts for every element-token pair."""

    positive: torch.Tensor
    negative: torch.Tensor
    unknown: torch.Tensor
    view_count: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        positive = _cpu_tensor(self.positive, dtype=torch.float32)
        negative = _cpu_tensor(self.negative, dtype=torch.float32)
        unknown = _cpu_tensor(self.unknown, dtype=torch.float32)
        raw_view_count = _cpu_tensor(self.view_count)
        quality = _cpu_tensor(self.quality, dtype=torch.float32)
        if positive.ndim != 2 or positive.shape[1] == 0:
            raise ValueError("observed evidence must have shape [E, K] with K > 0")
        if any(value.shape != positive.shape for value in (negative, unknown, quality)):
            raise ValueError("positive/negative/unknown/view quality must align [E, K]")
        if raw_view_count.shape != positive.shape:
            raise ValueError("view_count must align with element-token evidence [E, K]")
        if raw_view_count.is_floating_point():
            if not bool(torch.isfinite(raw_view_count).all()) or not torch.equal(
                raw_view_count, raw_view_count.round()
            ):
                raise ValueError("view_count must contain exact non-negative integers")
        view_count = raw_view_count.to(torch.int64)
        probabilities = (positive, negative, unknown, quality)
        if not all(bool(torch.isfinite(value).all()) for value in probabilities):
            raise ValueError("observed evidence must be finite")
        if any(bool((value < 0).any()) for value in (positive, negative, unknown)):
            raise ValueError("positive/negative/unknown evidence must be non-negative")
        if bool(((quality < 0) | (quality > 1)).any()):
            raise ValueError("observed evidence quality must lie in [0, 1]")
        if bool((view_count < 0).any()):
            raise ValueError("view_count must contain exact non-negative integers")
        if not torch.allclose(
            positive + negative + unknown,
            torch.ones_like(positive),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("positive/negative/unknown evidence must form a per-pair simplex")
        known = positive + negative > _PROBABILITY_TOLERANCE
        if bool((known & (view_count <= 0)).any()):
            raise ValueError("known observed evidence requires a positive view_count")
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "view_count", view_count)
        object.__setattr__(self, "quality", quality)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.positive.shape[0]), int(self.positive.shape[1])

    @property
    def known(self) -> torch.Tensor:
        return self.positive + self.negative > _PROBABILITY_TOLERANCE

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "unknown": self.unknown,
            "view_count": self.view_count,
            "quality": self.quality,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ElementTokenObservedEvidence":
        required = {"positive", "negative", "unknown", "view_count", "quality"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("observed evidence fields changed")
        return cls(**{key: payload[key] for key in required})


def _validate_completion_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("completion receipt must be a mapping")
    receipt = dict(value)
    required = {
        "schema",
        "method_family",
        "checkpoint_sha256",
        "training_report_sha256",
        "learned_model",
        "training_scenes_disjoint_from_evaluation",
        "writes_unknown_only",
        "observed_known_clamped",
        "completion_confidence_cap",
    }
    if set(receipt) != required or receipt.get("schema") != COMPLETION_RECEIPT_SCHEMA:
        raise ValueError("learned completion receipt fields/schema changed")
    method = receipt.get("method_family")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("learned completion receipt requires a method_family")
    lowered = method.lower()
    if "heuristic" in lowered or "geometry_envelope" in lowered:
        raise ValueError("formal bundles allow only learned completion")
    if not _is_sha256(receipt.get("checkpoint_sha256")) or not _is_sha256(
        receipt.get("training_report_sha256")
    ):
        raise ValueError("learned completion receipt requires non-placeholder checkpoint/report hashes")
    for field in (
        "learned_model",
        "training_scenes_disjoint_from_evaluation",
        "writes_unknown_only",
        "observed_known_clamped",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"learned completion receipt requires {field}=true")
    cap = receipt.get("completion_confidence_cap")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)):
        raise ValueError("completion_confidence_cap must be a finite probability")
    cap = float(cap)
    if not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError("completion_confidence_cap must be a finite probability")
    receipt["completion_confidence_cap"] = cap
    _content_digest(receipt)
    return receipt


@dataclass(frozen=True)
class SurfaceCarrierConfiguration:
    voxel_size: float
    maximum_splat_radius: int
    surface_band_voxels: float
    maximum_contributors_per_pixel: int
    camera_convention: str

    def __post_init__(self) -> None:
        voxel_size = float(self.voxel_size)
        surface_band = float(self.surface_band_voxels)
        radius = _exact_integer(
            self.maximum_splat_radius, name="maximum_splat_radius", minimum=0
        )
        contributor_cap = _exact_integer(
            self.maximum_contributors_per_pixel,
            name="maximum_contributors_per_pixel",
            minimum=1,
        )
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if not math.isfinite(surface_band) or surface_band < 0:
            raise ValueError("surface_band_voxels must be finite and non-negative")
        if self.camera_convention not in SUPPORTED_CAMERA_CONVENTIONS:
            raise ValueError("camera_convention is not in the v4 surface-carrier allowlist")
        object.__setattr__(self, "voxel_size", voxel_size)
        object.__setattr__(self, "maximum_splat_radius", radius)
        object.__setattr__(self, "surface_band_voxels", surface_band)
        object.__setattr__(self, "maximum_contributors_per_pixel", contributor_cap)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurfaceCarrierConfiguration":
        required = {
            "voxel_size",
            "maximum_splat_radius",
            "surface_band_voxels",
            "maximum_contributors_per_pixel",
            "camera_convention",
        }
        if set(value) != required:
            raise ValueError("surface carrier configuration fields changed")
        return cls(**{key: value[key] for key in required})

    def build_carrier(
        self,
        centres: torch.Tensor,
        *,
        normals: torch.Tensor | None,
        confidence: torch.Tensor,
    ) -> SurfaceVoxelCarrier:
        return SurfaceVoxelCarrier(
            centres,
            self.voxel_size,
            normals=normals,
            confidence=confidence,
            maximum_splat_radius=self.maximum_splat_radius,
            surface_band_voxels=self.surface_band_voxels,
            maximum_contributors_per_pixel=self.maximum_contributors_per_pixel,
        )


@dataclass(frozen=True)
class GeometryBinding:
    authority_path: str
    authority_sha256: str
    receipt: dict[str, Any]
    surface_carrier_path: str
    surface_carrier_sha256: str
    configuration: SurfaceCarrierConfiguration


def load_geometry_binding(
    authority_path: str | Path,
    surface_carrier_path: str | Path,
    *,
    verify_all_inputs: bool = False,
) -> tuple[GeometryBinding, dict[str, Any]]:
    """Read a geometry-gate authority and derive every carrier parameter from it."""

    authority_file = Path(authority_path).resolve(strict=True)
    surface_file = Path(surface_carrier_path).resolve(strict=True)
    outer = json.loads(authority_file.read_text())
    if not isinstance(outer, dict) or outer.get("passes_scene_gate") is not True:
        raise ValueError("geometry authority must be an outer scene report with passes_scene_gate=true")
    receipt = outer.get("geometry_receipt", outer)
    if not isinstance(receipt, dict) or receipt.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("geometry authority does not contain a v4 geometry receipt")
    carrier_name = str(receipt.get("carrier", "")).lower()
    if "sparse_surface" not in carrier_name:
        raise ValueError("geometry authority is not a sparse-surface carrier gate")
    for field in (
        "target_rgb_opened",
        "benchmark_images_opened",
        "benchmark_masks_opened",
        "benchmark_labels_opened",
    ):
        if receipt.get(field) is not False:
            raise ValueError(f"geometry authority must seal {field}=false")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("geometry receipt has no sealed inputs")
    surface_digest = sha256_file(surface_file)
    matching = [
        row for row in inputs
        if isinstance(row, dict)
        and row.get("role") in _SURFACE_INPUT_ROLES
        and row.get("sha256") == surface_digest
    ]
    if len(matching) != 1:
        raise ValueError("surface carrier is not uniquely hash-bound by the geometry receipt")
    if verify_all_inputs:
        for row in inputs:
            if not isinstance(row, dict) or not _is_sha256(row.get("sha256")):
                raise ValueError("geometry receipt contains an invalid sealed input")
            path = Path(str(row.get("path", ""))).resolve(strict=True)
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"sealed geometry input changed: {row.get('role')}")

    metadata = receipt.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("geometry receipt metadata is missing")
    required_projection = {
        "maximum_splat_radius",
        "surface_band_voxels",
        "maximum_contributors_per_pixel",
    }
    if not required_projection.issubset(metadata):
        raise ValueError("geometry receipt lacks the frozen projection configuration")
    outer_projection = outer.get("projection_configuration")
    if not isinstance(outer_projection, dict) or any(
        outer_projection.get(key) != metadata.get(key) for key in required_projection
    ):
        raise ValueError("geometry report and embedded receipt disagree on projection configuration")

    surface_payload = torch.load(surface_file, map_location="cpu", weights_only=False)
    if not isinstance(surface_payload, dict) or not {
        "centres", "voxel_size_colmap"
    }.issubset(surface_payload):
        raise ValueError("surface carrier payload is incomplete")
    configuration = SurfaceCarrierConfiguration(
        voxel_size=float(surface_payload["voxel_size_colmap"]),
        maximum_splat_radius=metadata["maximum_splat_radius"],
        surface_band_voxels=metadata["surface_band_voxels"],
        maximum_contributors_per_pixel=metadata["maximum_contributors_per_pixel"],
        camera_convention=str(receipt.get("coordinate_convention", "")),
    )
    binding = GeometryBinding(
        authority_path=str(authority_file),
        authority_sha256=sha256_file(authority_file),
        receipt=receipt,
        surface_carrier_path=str(surface_file),
        surface_carrier_sha256=surface_digest,
        configuration=configuration,
    )
    return binding, surface_payload


@dataclass(frozen=True)
class SurfaceSceneBundle:
    scene_label: str
    configuration: SurfaceCarrierConfiguration
    centres: torch.Tensor
    normals: torch.Tensor | None
    confidence: torch.Tensor
    observed_assignment: DenseObjectAssignments
    observed_evidence: ElementTokenObservedEvidence
    local_surface_memory: Mapping[str, torch.Tensor]
    object_memory: Mapping[str, torch.Tensor]
    source_frames: tuple[int, ...]
    source_input_digests: Mapping[str, str]
    geometry_authority_sha256: str
    source_surface_carrier_sha256: str
    information_policy: Mapping[str, bool]
    completed_assignment: DenseObjectAssignments | None = None
    completion_receipt: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("surface scene bundle schema changed")
        if not isinstance(self.scene_label, str) or not self.scene_label:
            raise ValueError("scene_label must be non-empty")
        centres = _cpu_tensor(self.centres, dtype=torch.float32)
        normals = None if self.normals is None else _cpu_tensor(self.normals, dtype=torch.float32)
        confidence = _cpu_tensor(self.confidence, dtype=torch.float32)
        # Constructing the carrier is the canonical geometry validation.
        self.configuration.build_carrier(centres, normals=normals, confidence=confidence)
        if not isinstance(self.observed_assignment, DenseObjectAssignments):
            raise ValueError("observed_assignment must be DenseObjectAssignments")
        element_count, token_count = self.observed_assignment.token_probability.shape
        if element_count != centres.shape[0]:
            raise ValueError("observed assignment does not align with the carrier")
        if not isinstance(self.observed_evidence, ElementTokenObservedEvidence):
            raise ValueError("observed_evidence must be ElementTokenObservedEvidence")
        if self.observed_evidence.shape != (element_count, token_count):
            raise ValueError("observed evidence must align with carrier elements and object tokens")
        known_pairs = self.observed_evidence.known
        unknown_only_pairs = ~known_pairs
        if bool(
            (
                self.observed_assignment.token_probability[unknown_only_pairs]
                > _PROBABILITY_TOLERANCE
            ).any()
        ):
            raise ValueError("observed assignment cannot write evidence-unknown element-token pairs")
        explicit_negative = (
            self.observed_evidence.negative > _PROBABILITY_TOLERANCE
        ) & (self.observed_evidence.positive <= _PROBABILITY_TOLERANCE)
        if bool(
            (
                self.observed_assignment.token_probability[explicit_negative]
                > _PROBABILITY_TOLERANCE
            ).any()
        ):
            raise ValueError("observed assignment contradicts explicit negative evidence")
        completion_receipt = None
        if self.completed_assignment is not None:
            if self.completion_receipt is None:
                raise ValueError("completed assignment requires a learned-completion receipt")
            if not isinstance(self.completed_assignment, DenseObjectAssignments):
                raise ValueError("completed_assignment must be DenseObjectAssignments")
            if self.completed_assignment.token_probability.shape != self.observed_assignment.token_probability.shape:
                raise ValueError("completed and observed assignments must align")
            completion_receipt = _validate_completion_receipt(self.completion_receipt)
            if not torch.equal(
                self.completed_assignment.token_probability[known_pairs],
                self.observed_assignment.token_probability[known_pairs],
            ):
                raise ValueError("learned completion must strictly clamp all observed known pairs")
            if bool(
                (
                    self.completed_assignment.unknown_probability
                    > self.observed_assignment.unknown_probability + _PROBABILITY_TOLERANCE
                ).any()
            ):
                raise ValueError("learned completion cannot increase assignment unknown mass")
            changed = ~torch.isclose(
                self.completed_assignment.token_probability,
                self.observed_assignment.token_probability,
                atol=_PROBABILITY_TOLERANCE,
                rtol=0,
            )
            if bool((changed & known_pairs).any()):
                raise ValueError("learned completion can write only evidence-unknown pairs")
            cap = float(completion_receipt["completion_confidence_cap"])
            if bool(
                (
                    changed
                    & (self.completed_assignment.token_probability > cap + _PROBABILITY_TOLERANCE)
                ).any()
            ):
                raise ValueError("learned completion exceeds its receipted confidence cap")
        elif self.completion_receipt is not None:
            raise ValueError("completion receipt cannot exist without completed assignments")
        local_memory = _validate_tensor_mapping(
            "local_surface_memory", self.local_surface_memory, expected_rows=int(centres.shape[0])
        )
        object_memory = _validate_tensor_mapping(
            "object_memory", self.object_memory, expected_rows=token_count
        )
        if not _is_sha256(self.geometry_authority_sha256):
            raise ValueError("geometry authority digest is invalid")
        if not _is_sha256(self.source_surface_carrier_sha256):
            raise ValueError("source surface carrier digest is invalid")
        digests = dict(self.source_input_digests)
        if not digests or any(
            not isinstance(key, str) or not key or not _is_sha256(value)
            for key, value in digests.items()
        ):
            raise ValueError("source_input_digests must contain valid sha256 values")
        if digests.get("geometry_authority") != self.geometry_authority_sha256:
            raise ValueError("source_input_digests geometry authority binding differs")
        if digests.get("surface_carrier") != self.source_surface_carrier_sha256:
            raise ValueError("source_input_digests surface carrier binding differs")
        policy = dict(self.information_policy)
        if any(not isinstance(key, str) or not isinstance(value, bool) for key, value in policy.items()):
            raise ValueError("information_policy must contain explicit boolean fields")
        required_policy = {
            "target_rgb_opened_during_construction": False,
            "benchmark_labels_opened_during_construction": False,
            "text_queries_opened_during_construction": False,
        }
        if any(policy.get(key) is not expected for key, expected in required_policy.items()):
            raise ValueError("scene construction information policy is not query-free")
        source_frames = tuple(
            _exact_integer(value, name="source frame", minimum=0) for value in self.source_frames
        )
        if not source_frames or len(source_frames) != len(set(source_frames)):
            raise ValueError("source_frames must be a non-empty unique sequence")
        metadata = {} if self.metadata is None else dict(self.metadata)
        _content_digest(metadata)
        object.__setattr__(self, "centres", centres)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "completion_receipt", completion_receipt)
        object.__setattr__(self, "local_surface_memory", local_memory)
        object.__setattr__(self, "object_memory", object_memory)
        object.__setattr__(self, "source_frames", source_frames)
        object.__setattr__(self, "source_input_digests", digests)
        object.__setattr__(self, "information_policy", policy)
        object.__setattr__(self, "metadata", metadata)

    @property
    def carrier_content_sha256(self) -> str:
        return _tensor_digest(
            self.centres,
            self.normals,
            self.confidence,
            configuration=asdict(self.configuration),
        )

    def build_carrier(self) -> SurfaceVoxelCarrier:
        return self.configuration.build_carrier(
            self.centres, normals=self.normals, confidence=self.confidence
        )

    def _payload_without_content_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scene_label": self.scene_label,
            "carrier_configuration": asdict(self.configuration),
            "carrier": {
                "centres": self.centres,
                "normals": self.normals,
                "confidence": self.confidence,
                "content_sha256": self.carrier_content_sha256,
            },
            "observed_assignment": {
                "token_probability": self.observed_assignment.token_probability.cpu(),
                "unknown_probability": self.observed_assignment.unknown_probability.cpu(),
            },
            "observed_evidence": self.observed_evidence.to_payload(),
            "completed_assignment": None if self.completed_assignment is None else {
                "token_probability": self.completed_assignment.token_probability.cpu(),
                "unknown_probability": self.completed_assignment.unknown_probability.cpu(),
            },
            "completion_receipt": self.completion_receipt,
            "local_surface_memory": dict(self.local_surface_memory),
            "object_memory": dict(self.object_memory),
            "source_frames": list(self.source_frames),
            "source_input_digests": dict(self.source_input_digests),
            "geometry_authority_sha256": self.geometry_authority_sha256,
            "source_surface_carrier_sha256": self.source_surface_carrier_sha256,
            "information_policy": dict(self.information_policy),
            "metadata": dict(self.metadata or {}),
        }

    @property
    def content_sha256(self) -> str:
        """Digest every persistent geometry, object, semantic, and receipt field."""

        return _content_digest(self._payload_without_content_digest())

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_without_content_digest()
        payload["content_sha256"] = _content_digest(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SurfaceSceneBundle":
        if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
            raise ValueError("not a v4 surface scene bundle")
        expected_content_digest = payload.get("content_sha256")
        if not _is_sha256(expected_content_digest):
            raise ValueError("surface scene bundle lacks a valid full-state content digest")
        content = dict(payload)
        del content["content_sha256"]
        if _content_digest(content) != expected_content_digest:
            raise ValueError("surface scene bundle full-state content digest changed")
        carrier = payload.get("carrier")
        observed = payload.get("observed_assignment")
        observed_evidence = payload.get("observed_evidence")
        if (
            not isinstance(carrier, Mapping)
            or not isinstance(observed, Mapping)
            or not isinstance(observed_evidence, Mapping)
        ):
            raise ValueError("surface scene bundle is incomplete")
        completed_payload = payload.get("completed_assignment")
        completed = None
        if completed_payload is not None:
            if not isinstance(completed_payload, Mapping):
                raise ValueError("completed_assignment must be a mapping")
            completed = DenseObjectAssignments(
                completed_payload["token_probability"], completed_payload["unknown_probability"]
            )
        result = cls(
            schema=str(payload["schema"]),
            scene_label=str(payload["scene_label"]),
            configuration=SurfaceCarrierConfiguration.from_dict(payload["carrier_configuration"]),
            centres=carrier["centres"],
            normals=carrier.get("normals"),
            confidence=carrier["confidence"],
            observed_assignment=DenseObjectAssignments(
                observed["token_probability"], observed["unknown_probability"]
            ),
            observed_evidence=ElementTokenObservedEvidence.from_payload(observed_evidence),
            completed_assignment=completed,
            completion_receipt=payload.get("completion_receipt"),
            local_surface_memory=payload.get("local_surface_memory", {}),
            object_memory=payload.get("object_memory", {}),
            source_frames=tuple(payload["source_frames"]),
            source_input_digests=payload["source_input_digests"],
            geometry_authority_sha256=str(payload["geometry_authority_sha256"]),
            source_surface_carrier_sha256=str(payload["source_surface_carrier_sha256"]),
            information_policy=payload["information_policy"],
            metadata=payload.get("metadata", {}),
        )
        if carrier.get("content_sha256") != result.carrier_content_sha256:
            raise ValueError("surface carrier content digest changed")
        if result.content_sha256 != expected_content_digest:
            raise ValueError("surface scene bundle normalized content digest changed")
        return result

    def save(self, path: str | Path) -> str:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(self.to_payload(), temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return sha256_file(destination)

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str) -> "SurfaceSceneBundle":
        source = Path(path).resolve(strict=True)
        if not _is_sha256(expected_sha256):
            raise ValueError("a non-placeholder expected bundle SHA256 is required")
        if sha256_file(source) != expected_sha256:
            raise ValueError("surface scene bundle file digest changed")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        return cls.from_payload(payload)


def projection_digest(projection: ProjectionTable) -> str:
    """Stable cold-load regression digest for one carrier/camera projection."""

    digest = hashlib.sha256()
    for tensor in (
        projection.element_ids,
        projection.pixel_ids,
        projection.depths,
        projection.weights,
        projection.depth_residuals,
    ):
        _update_content_digest(digest, tensor)
    _update_content_digest(
        digest,
        {
            "num_elements": projection.num_elements,
            "height": projection.height,
            "width": projection.width,
            "normalization": projection.normalization,
            "metadata": projection.metadata,
        },
    )
    return digest.hexdigest()


def cold_load_projection_digest(
    bundle_path: str | Path,
    camera: Camera,
    *,
    expected_bundle_sha256: str,
    expected_projection_sha256: str,
) -> str:
    bundle = SurfaceSceneBundle.load(bundle_path, expected_sha256=expected_bundle_sha256)
    observed = projection_digest(bundle.build_carrier().project(camera))
    if not _is_sha256(expected_projection_sha256):
        raise ValueError("a non-placeholder expected projection SHA256 is required")
    if observed != expected_projection_sha256:
        raise ValueError("cold-loaded projection digest differs from the build-time projection")
    return observed
