"""Fail-closed prompt-view Gaussian responsibility artifacts.

The artifact stores the sparse native-resolution front-to-back compositor
matrix ``W``.  It deliberately contains no prompt labels, target images, or
target masks.  A dense prompt posterior can therefore be transferred to
Gaussian rows with ``W.T @ y`` and replayed into the prompt view with ``W @ u``
without reopening benchmark targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

import torch


ARTIFACT_TYPE = "radio_gs.prompt_responsibility_cache"
SCHEMA_VERSION = 1
COMPOSITOR_CONTRACT = (
    "gsplat_3dgs_native_resolution_exact_accepted_hits_front_to_back_"
    "weight_equals_alpha_times_exclusive_transmittance_no_post_alpha_filter_"
    "drop_exact_numeric_zero_weight_only"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and contiguous little-endian CPU tensor bytes."""

    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError("tensor_sha256 requires a CPU tensor")
    tensor = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PromptResponsibilityAuthority:
    scene_id: str
    frame_id: str
    camera_name: str
    colmap_camera_name: str
    geometry_checkpoint_sha256: str
    geometry_xyz_sha256: str
    pose_sha256: str
    intrinsics_sha256: str
    height: int
    width: int
    num_gaussians: int
    alpha_threshold: float = 0.0
    compositor_contract: str = COMPOSITOR_CONTRACT
    target_rgb_opened: bool = False
    target_mask_opened: bool = False
    source_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("scene_id", "frame_id", "camera_name", "colmap_camera_name"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"authority {name} cannot be empty")
        for name in (
            "geometry_checkpoint_sha256",
            "geometry_xyz_sha256",
            "pose_sha256",
            "intrinsics_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"authority {name}")
        if int(self.height) <= 0 or int(self.width) <= 0 or int(self.num_gaussians) <= 0:
            raise ValueError("authority dimensions and row count must be positive")
        if float(self.alpha_threshold) != 0.0:
            raise ValueError("exact responsibility authority forbids post alpha filtering")
        if self.compositor_contract != COMPOSITOR_CONTRACT:
            raise ValueError("unknown prompt responsibility compositor contract")
        if self.target_rgb_opened is not False or self.target_mask_opened is not False:
            raise ValueError("target RGB and target masks must remain unopened")
        sources = dict(self.source_sha256 or {})
        if not sources:
            raise ValueError("authority requires source SHA-256 bindings")
        for name, digest in sources.items():
            if not str(name).strip():
                raise ValueError("source SHA-256 binding name cannot be empty")
            _require_sha256(digest, label=f"source {name}")
        object.__setattr__(self, "source_sha256", MappingProxyType(dict(sorted(sources.items()))))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "colmap_camera_name": self.colmap_camera_name,
            "geometry_checkpoint_sha256": self.geometry_checkpoint_sha256,
            "geometry_xyz_sha256": self.geometry_xyz_sha256,
            "pose_sha256": self.pose_sha256,
            "intrinsics_sha256": self.intrinsics_sha256,
            "height": int(self.height),
            "width": int(self.width),
            "num_gaussians": int(self.num_gaussians),
            "alpha_threshold": float(self.alpha_threshold),
            "compositor_contract": self.compositor_contract,
            "target_rgb_opened": self.target_rgb_opened,
            "target_mask_opened": self.target_mask_opened,
            "source_sha256": dict(self.source_sha256 or {}),
        }

    @property
    def digest(self) -> str:
        return _json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "PromptResponsibilityAuthority":
        if not isinstance(value, dict):
            raise ValueError("responsibility authority must be a dictionary")
        expected = {
            "scene_id", "frame_id", "camera_name", "colmap_camera_name",
            "geometry_checkpoint_sha256", "geometry_xyz_sha256", "pose_sha256",
            "intrinsics_sha256", "height", "width", "num_gaussians",
            "alpha_threshold", "compositor_contract", "target_rgb_opened",
            "target_mask_opened", "source_sha256",
        }
        if set(value) != expected:
            raise ValueError("responsibility authority keys differ from schema")
        if not isinstance(value["target_rgb_opened"], bool) or not isinstance(
            value["target_mask_opened"], bool
        ):
            raise ValueError("target-opened authority flags must be bool")
        return cls(**value)


@dataclass(frozen=True)
class AdjointResult:
    weighted_sum: torch.Tensor
    visible_mass: torch.Tensor
    primitive_probability: torch.Tensor
    visible: torch.Tensor


@dataclass(frozen=True)
class ForwardResult:
    weighted_sum: torch.Tensor
    pixel_mass: torch.Tensor
    normalized_probability: torch.Tensor
    supported: torch.Tensor


@dataclass(frozen=True)
class PromptResponsibilityCache:
    authority: PromptResponsibilityAuthority
    gaussian_ids: torch.Tensor
    pixel_ids: torch.Tensor
    weights: torch.Tensor
    visible_mass: torch.Tensor

    def __post_init__(self) -> None:
        _validate_cache(self)

    @property
    def tensor_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "gaussian_ids": tensor_sha256(self.gaussian_ids),
                "pixel_ids": tensor_sha256(self.pixel_ids),
                "visible_mass": tensor_sha256(self.visible_mass),
                "weights": tensor_sha256(self.weights),
            }
        )

    @property
    def tensor_bundle_sha256(self) -> str:
        return _json_sha256(dict(self.tensor_sha256))

    def adjoint(self, pixel_probability: torch.Tensor) -> AdjointResult:
        y = torch.as_tensor(pixel_probability, device="cpu")
        expected = int(self.authority.height) * int(self.authority.width)
        if y.numel() != expected or y.ndim not in (1, 2):
            raise ValueError("pixel probability must be flat [H*W] or [H,W]")
        if y.ndim == 2 and tuple(y.shape) != (self.authority.height, self.authority.width):
            raise ValueError("pixel probability has the wrong native shape")
        y = y.reshape(-1).to(torch.float64)
        if not bool(torch.isfinite(y).all()) or bool(((y < 0) | (y > 1)).any()):
            raise ValueError("pixel probability must be finite in [0,1]")
        numerator = torch.zeros(self.authority.num_gaussians, dtype=torch.float64)
        numerator.index_add_(
            0, self.gaussian_ids, self.weights.to(torch.float64) * y[self.pixel_ids]
        )
        visible = self.visible_mass > 0
        probability = torch.zeros_like(numerator)
        probability[visible] = numerator[visible] / self.visible_mass[visible]
        return AdjointResult(numerator, self.visible_mass.clone(), probability, visible)

    def forward(self, primitive_probability: torch.Tensor) -> ForwardResult:
        u = torch.as_tensor(primitive_probability, device="cpu").reshape(-1).to(torch.float64)
        if u.shape != (self.authority.num_gaussians,):
            raise ValueError("primitive probability has the wrong row count")
        if not bool(torch.isfinite(u).all()) or bool(((u < 0) | (u > 1)).any()):
            raise ValueError("primitive probability must be finite in [0,1]")
        pixels = self.authority.height * self.authority.width
        numerator = torch.zeros(pixels, dtype=torch.float64)
        mass = torch.zeros(pixels, dtype=torch.float64)
        weight64 = self.weights.to(torch.float64)
        numerator.index_add_(0, self.pixel_ids, weight64 * u[self.gaussian_ids])
        mass.index_add_(0, self.pixel_ids, weight64)
        supported = mass > 0
        probability = torch.zeros_like(numerator)
        probability[supported] = numerator[supported] / mass[supported]
        shape = (self.authority.height, self.authority.width)
        return ForwardResult(
            numerator.reshape(shape), mass.reshape(shape), probability.reshape(shape), supported.reshape(shape)
        )

    def cycle(self, pixel_probability: torch.Tensor) -> ForwardResult:
        return self.forward(self.adjoint(pixel_probability).primitive_probability)


def build_prompt_responsibility_cache(
    *,
    authority: PromptResponsibilityAuthority,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
) -> PromptResponsibilityCache:
    gids = torch.as_tensor(gaussian_ids, device="cpu").detach().contiguous().to(torch.int64)
    pids = torch.as_tensor(pixel_ids, device="cpu").detach().contiguous().to(torch.int64)
    weight = torch.as_tensor(weights, device="cpu").detach().contiguous().to(torch.float32)
    visible = torch.zeros(authority.num_gaussians, dtype=torch.float64)
    if gids.numel():
        visible.index_add_(0, gids, weight.to(torch.float64))
    return PromptResponsibilityCache(authority, gids, pids, weight, visible)


def _validate_cache(value: PromptResponsibilityCache) -> None:
    tensors = {
        "gaussian_ids": (value.gaussian_ids, torch.int64),
        "pixel_ids": (value.pixel_ids, torch.int64),
        "weights": (value.weights, torch.float32),
        "visible_mass": (value.visible_mass, torch.float64),
    }
    for name, (tensor, dtype) in tensors.items():
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise ValueError(f"{name} must be a CPU {dtype} tensor")
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous and one-dimensional")
    count = value.gaussian_ids.numel()
    if count == 0 or value.pixel_ids.numel() != count or value.weights.numel() != count:
        raise ValueError("responsibility triplets must be nonempty and aligned")
    if value.visible_mass.shape != (value.authority.num_gaussians,):
        raise ValueError("visible_mass shape differs from Gaussian row count")
    if int(value.gaussian_ids.min()) < 0 or int(value.gaussian_ids.max()) >= value.authority.num_gaussians:
        raise ValueError("gaussian_ids are outside authority row bounds")
    pixels = value.authority.height * value.authority.width
    if int(value.pixel_ids.min()) < 0 or int(value.pixel_ids.max()) >= pixels:
        raise ValueError("pixel_ids are outside native image bounds")
    if not bool(torch.isfinite(value.weights).all()) or bool((value.weights <= 0).any()):
        raise ValueError("stored weights must be finite and strictly positive")
    if bool((value.pixel_ids[1:] < value.pixel_ids[:-1]).any()):
        raise ValueError("pixel_ids must preserve compositor pixel grouping")
    # Pixel ids are grouped by the exact compositor, so complete pixel groups
    # can be checked in bounded chunks instead of allocating another full-size
    # copy of a native-resolution cache.  This matters on 32-GiB hosts.
    start = 0
    chunk_size = 1_000_000
    while start < count:
        stop = min(count, start + chunk_size)
        if stop < count:
            boundary_pixel = value.pixel_ids[stop - 1]
            while stop < count and value.pixel_ids[stop] == boundary_pixel:
                stop += 1
        packed = (
            value.pixel_ids[start:stop] * int(value.authority.num_gaussians)
            + value.gaussian_ids[start:stop]
        )
        if torch.unique(packed).numel() != packed.numel():
            raise ValueError("duplicate (pixel_id, gaussian_id) responsibility pair")
        start = stop
    pixel_mass = torch.zeros(pixels, dtype=torch.float64)
    pixel_mass.index_add_(0, value.pixel_ids, value.weights.to(torch.float64))
    if bool((pixel_mass > 1.00001).any()):
        raise ValueError("front-to-back responsibility mass exceeds one")
    recomputed = torch.zeros(value.authority.num_gaussians, dtype=torch.float64)
    recomputed.index_add_(0, value.gaussian_ids, value.weights.to(torch.float64))
    if not torch.equal(recomputed, value.visible_mass):
        raise ValueError("visible_mass does not equal W.T @ 1 exactly")


@dataclass(frozen=True)
class PromptResponsibilityArtifact:
    path: str
    file_sha256: str
    authority_sha256: str
    tensor_bundle_sha256: str


def save_prompt_responsibility_cache(
    value: PromptResponsibilityCache,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> PromptResponsibilityArtifact:
    _validate_cache(value)
    tensor_digests = dict(value.tensor_sha256)
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "authority": value.authority.to_dict(),
        "authority_sha256": value.authority.digest,
        "tensor_sha256": tensor_digests,
        "tensor_bundle_sha256": _json_sha256(tensor_digests),
        "tensors": {
            "gaussian_ids": value.gaussian_ids,
            "pixel_ids": value.pixel_ids,
            "weights": value.weights,
            "visible_mass": value.visible_mass,
        },
    }
    output = Path(path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return PromptResponsibilityArtifact(
        str(output), sha256_file(output), value.authority.digest, value.tensor_bundle_sha256
    )


def load_prompt_responsibility_cache(
    path: str | Path,
    *,
    expected_authority: PromptResponsibilityAuthority,
    expected_file_sha256: str | None = None,
) -> PromptResponsibilityCache:
    source = Path(path).expanduser().absolute()
    if expected_file_sha256 is not None:
        expected_digest = _require_sha256(expected_file_sha256, label="expected artifact")
        if sha256_file(source) != expected_digest:
            raise ValueError("responsibility artifact file SHA-256 differs")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("responsibility loading requires weights_only=True") from error
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_type", "schema_version", "authority", "authority_sha256",
        "tensor_sha256", "tensor_bundle_sha256", "tensors",
    }:
        raise ValueError("responsibility artifact keys differ from schema")
    if payload["artifact_type"] != ARTIFACT_TYPE or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("responsibility artifact type or schema differs")
    authority = PromptResponsibilityAuthority.from_dict(payload["authority"])
    if authority.to_dict() != expected_authority.to_dict() or authority.digest != expected_authority.digest:
        raise ValueError("responsibility authority differs from expected authority")
    if _require_sha256(payload["authority_sha256"], label="stored authority") != authority.digest:
        raise ValueError("stored responsibility authority digest differs")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != {
        "gaussian_ids", "pixel_ids", "weights", "visible_mass"
    }:
        raise ValueError("responsibility tensor keys differ from schema")
    cache = PromptResponsibilityCache(
        authority=authority,
        gaussian_ids=tensors["gaussian_ids"],
        pixel_ids=tensors["pixel_ids"],
        weights=tensors["weights"],
        visible_mass=tensors["visible_mass"],
    )
    stored_digests = payload["tensor_sha256"]
    if not isinstance(stored_digests, dict) or set(stored_digests) != set(cache.tensor_sha256):
        raise ValueError("responsibility tensor digest keys differ")
    actual_digests = dict(cache.tensor_sha256)
    if stored_digests != actual_digests:
        raise ValueError("responsibility tensor content differs from stored digest")
    bundle = _require_sha256(payload["tensor_bundle_sha256"], label="stored tensor bundle")
    if bundle != _json_sha256(actual_digests):
        raise ValueError("responsibility tensor bundle digest differs")
    return cache
