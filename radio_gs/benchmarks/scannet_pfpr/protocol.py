"""Leak-free protocol primitives for ScanNet-PFPR-Small v1.

PFPR evaluates a held-out RGB patch against a fixed geometry-only 3-D point
domain.  Instance IDs, 2-D masks, query pose, query depth, and the private
anchor are deliberately absent from the method manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


PFPR_V1_BENCHMARK_VERSION = "scannet-pfpr-small-v1"
PFPR_V2_BENCHMARK_VERSION = "scannet-pfpr-small-v2"
# New construction defaults to v2.  Readers retain v1 support so frozen
# legacy scores remain reproducible and cannot be silently relabelled.
BENCHMARK_VERSION = PFPR_V2_BENCHMARK_VERSION
SUPPORTED_BENCHMARK_VERSIONS = frozenset(
    {PFPR_V1_BENCHMARK_VERSION, PFPR_V2_BENCHMARK_VERSION}
)
NATIVE_COLOR_QUERY_RASTER_V1 = "native_color_rgb_v1"
DEPTH_ALIGNED_QUERY_RASTER_V2 = "depth_aligned_rgb_v2"


def validate_benchmark_version(value: str) -> str:
    """Return one known immutable PFPR release version."""

    version = str(value)
    if version not in SUPPORTED_BENCHMARK_VERSIONS:
        raise ValueError(f"unsupported ScanNet-PFPR release: {version!r}")
    return version


def expected_query_raster_contract(benchmark_version: str) -> str:
    """Return the one immutable crop-raster rule for a PFPR release."""

    return {
        PFPR_V1_BENCHMARK_VERSION: NATIVE_COLOR_QUERY_RASTER_V1,
        PFPR_V2_BENCHMARK_VERSION: DEPTH_ALIGNED_QUERY_RASTER_V2,
    }[validate_benchmark_version(benchmark_version)]


@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen geometry-only construction and retrieval constants for v1."""

    patch_size_px: int = 128
    anchors_per_scene: int = 10
    depth_grid_stride_px: int = 8
    depth_window_size_px: int = 5
    minimum_window_valid_fraction: float = 0.80
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 8.00
    candidate_voxel_size_m: float = 0.05
    maximum_anchor_to_domain_distance_m: float = 0.05
    query_raster_contract: str = DEPTH_ALIGNED_QUERY_RASTER_V2
    nms_radius_m: float = 0.10
    retrieval_ks: tuple[int, ...] = (1, 5, 10)
    distance_thresholds_m: tuple[float, ...] = (0.05, 0.10, 0.20)

    def __post_init__(self) -> None:
        if self.patch_size_px <= 0 or self.patch_size_px % 2:
            raise ValueError("patch_size_px must be a positive even integer")
        if self.anchors_per_scene <= 0 or self.depth_grid_stride_px <= 0:
            raise ValueError("anchors_per_scene/depth_grid_stride_px must be positive")
        if self.depth_window_size_px <= 0 or self.depth_window_size_px % 2 != 1:
            raise ValueError("depth_window_size_px must be a positive odd integer")
        if not 0.0 < self.minimum_window_valid_fraction <= 1.0:
            raise ValueError("minimum_window_valid_fraction must be in (0,1]")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("depth range is invalid")
        if min(self.candidate_voxel_size_m, self.maximum_anchor_to_domain_distance_m, self.nms_radius_m) <= 0:
            raise ValueError("PFPR geometry radii must be positive")
        if self.query_raster_contract not in {
            NATIVE_COLOR_QUERY_RASTER_V1,
            DEPTH_ALIGNED_QUERY_RASTER_V2,
        }:
            raise ValueError("PFPR query raster contract is invalid")
        if not self.retrieval_ks or any(int(value) <= 0 for value in self.retrieval_ks):
            raise ValueError("retrieval_ks must contain positive values")
        if not self.distance_thresholds_m or any(float(value) <= 0 for value in self.distance_thresholds_m):
            raise ValueError("distance_thresholds_m must contain positive values")


def validate_release_config(
    benchmark_version: str,
    config: ProtocolConfig,
) -> None:
    """Reject a manifest whose declared version and crop raster disagree."""

    expected = expected_query_raster_contract(benchmark_version)
    if config.query_raster_contract != expected:
        raise ValueError(
            f"{benchmark_version} requires query raster {expected!r}; "
            f"got {config.query_raster_contract!r}"
        )


def protocol_config_from_record(
    benchmark_version: str,
    raw: Mapping[str, Any] | None,
) -> ProtocolConfig:
    """Load a release config while preserving pre-raster-field v1 manifests."""

    version = validate_benchmark_version(benchmark_version)
    values = dict(raw or {})
    # Frozen v1 manifests predate the explicit raster key.  Its release name
    # is nevertheless unambiguous, so normalize only that historical omission
    # instead of allowing a v1/v2 crop contract to be mixed silently.
    values.setdefault("query_raster_contract", expected_query_raster_contract(version))
    config = ProtocolConfig(**values)
    validate_release_config(version, config)
    return config


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_frame_exclusion_digest(frame_ids: Sequence[str | int]) -> str:
    """Commit to a withheld source-frame set without publishing frame IDs."""

    try:
        normalized = sorted({int(value) for value in frame_ids})
    except (TypeError, ValueError) as error:
        raise ValueError("PFPR query source frame IDs must be integers") from error
    if not normalized:
        raise ValueError("PFPR query source-frame exclusion set cannot be empty")
    return hashlib.sha256(
        canonical_json_sha256(normalized).encode("utf-8")
    ).hexdigest()


def validate_field_query_exclusion_commitment(
    benchmark_version: str,
    public_digest: str,
    field_digest: str,
) -> None:
    """Require v2 fields to exclude exactly the public query-frame commitment."""

    version = validate_benchmark_version(benchmark_version)
    if version != PFPR_V2_BENCHMARK_VERSION:
        return
    expected = str(public_digest)
    actual = str(field_digest)
    if not expected or not actual or expected != actual:
        raise ValueError(
            "PFPR v2 field/query source-frame exclusion commitment disagrees"
        )


def protocol_record(
    config: ProtocolConfig = ProtocolConfig(),
    *,
    benchmark_version: str = BENCHMARK_VERSION,
) -> dict[str, Any]:
    version = validate_benchmark_version(benchmark_version)
    validate_release_config(version, config)
    return {
        "benchmark_version": version,
        "protocol_config": asdict(config),
    }


def method_query_record(
    *,
    query_id: str,
    scene_id: str,
    crop_rgb_path: str,
    crop_rgb_sha256: str,
    benchmark_version: str = BENCHMARK_VERSION,
) -> dict[str, Any]:
    """Return exactly the query fields a method may inspect."""

    if not all(str(value) for value in (query_id, scene_id, crop_rgb_path, crop_rgb_sha256)):
        raise ValueError("method-visible PFPR query values must be non-empty")
    return {
        "benchmark_version": validate_benchmark_version(benchmark_version),
        "query_id": str(query_id),
        "scene_id": str(scene_id),
        "crop_rgb_path": str(crop_rgb_path),
        "crop_rgb_sha256": str(crop_rgb_sha256),
        "available_method_inputs": ["scene_id", "crop_rgb"],
    }


def stable_voxel_domain(xyz: np.ndarray, *, voxel_size_m: float) -> np.ndarray:
    """Return one input geometry row per occupied shifted ScanNet voxel."""

    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("candidate geometry must be finite [N,3]")
    if len(points) == 0 or voxel_size_m <= 0:
        raise ValueError("candidate geometry/voxel size is invalid")
    discrete = np.floor((points - points.min(axis=0, keepdims=True)) / float(voxel_size_m)).astype(np.int32)
    _unique, first, _inverse = np.unique(
        discrete, axis=0, return_index=True, return_inverse=True
    )
    # Match the stable first-occurrence convention used by ScanNet point
    # protocols; geometry order is public and never depends on a query.
    rows = np.sort(first, kind="stable")
    return np.ascontiguousarray(points[rows])


def fixed_radius_nms(
    xyz: np.ndarray,
    scores: np.ndarray,
    *,
    radius_m: float,
    maximum: int,
) -> np.ndarray:
    """Select score-ranked spatial hypotheses with deterministic Euclidean NMS."""

    points = np.asarray(xyz, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or values.shape != (len(points),):
        raise ValueError("PFPR scores and candidate points must align")
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError("PFPR scores/candidate points must be finite")
    if radius_m <= 0 or maximum <= 0:
        raise ValueError("NMS radius and maximum must be positive")
    # Stable sorting makes equal scores reproducible in public candidate order.
    order = np.argsort(-values, kind="stable")
    selected: list[int] = []
    squared_radius = float(radius_m) ** 2
    for index in order:
        if all(float(((points[index] - points[prior]) ** 2).sum()) > squared_radius for prior in selected):
            selected.append(int(index))
            if len(selected) >= int(maximum):
                break
    return np.asarray(selected, dtype=np.int64)


def _distance_key(threshold_m: float) -> str:
    return f"{int(round(float(threshold_m) * 100.0))}cm"


def evaluate_ranked_locations(
    predicted_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    *,
    config: ProtocolConfig = ProtocolConfig(),
) -> dict[str, Any]:
    """Evaluate locations against a private 3-D anchor, never an instance ID."""

    predicted = np.asarray(predicted_xyz, dtype=np.float32)
    anchor = np.asarray(anchor_xyz, dtype=np.float32).reshape(-1)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or not len(predicted):
        raise ValueError("ranked PFPR locations must be a non-empty [K,3] array")
    if anchor.shape != (3,) or not np.isfinite(predicted).all() or not np.isfinite(anchor).all():
        raise ValueError("PFPR locations/anchor must be finite")
    errors = np.linalg.norm(predicted - anchor[None], axis=1)
    output: dict[str, Any] = {
        "top1_error_m": float(errors[0]),
        "ranked_errors_m": [float(value) for value in errors],
    }
    for threshold in config.distance_thresholds_m:
        name = _distance_key(threshold)
        correct = errors <= float(threshold)
        first = np.flatnonzero(correct)
        output[f"first_correct_rank_{name}"] = int(first[0] + 1) if len(first) else None
        for count in config.retrieval_ks:
            output[f"recall_at_{int(count)}_{name}"] = bool(correct[: int(count)].any())
    return output


def aggregate_query_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ProtocolConfig = ProtocolConfig(),
) -> dict[str, float]:
    """Aggregate PFPR query metrics; caller additionally reports scene macro."""

    if not rows:
        raise ValueError("PFPR needs at least one evaluated query")
    result: dict[str, float] = {
        "top1_mean_error_m": float(np.mean([float(row["top1_error_m"]) for row in rows])),
        "top1_median_error_m": float(np.median([float(row["top1_error_m"]) for row in rows])),
    }
    for threshold in config.distance_thresholds_m:
        name = _distance_key(threshold)
        ranks = [row[f"first_correct_rank_{name}"] for row in rows]
        for count in config.retrieval_ks:
            result[f"R@{int(count)}_{name}"] = float(
                np.mean([bool(row[f"recall_at_{int(count)}_{name}"]) for row in rows])
            )
        result[f"MRR_{name}"] = float(
            np.mean([0.0 if rank is None else 1.0 / int(rank) for rank in ranks])
        )
    return result
