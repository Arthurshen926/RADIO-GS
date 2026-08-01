"""Fail-closed tensor-cache loading and MPR validation helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import load_torch_payload


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def load_mpr_cache(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_feature_space: str | None = None,
    require_reliability: bool = True,
    require_formal_safety: bool = False,
) -> tuple[dict[str, Any], str, Path]:
    """Safely load, content-bind, and deeply validate one MPR cache."""

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
