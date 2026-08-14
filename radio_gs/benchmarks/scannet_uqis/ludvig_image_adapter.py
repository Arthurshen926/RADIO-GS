"""Benchmark-local LUDVIG image-query adapter for ScanNet-UQIS.

The pure validation and scoring functions in this module intentionally avoid
importing the CUDA LUDVIG runtime.  That keeps the protocol boundary testable
with small synthetic 40-D fields while the exact runtime remains a separate,
lazy path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    PREDICTION_DOMAIN,
    UQISProtocolConfig,
    canonical_json_sha256,
    sha256_file,
)


IMAGE_MANIFEST_KEYS = {
    "benchmark_version",
    "split_role",
    "release_tier",
    "formal_benchmark_eligible",
    "protocol_config",
    "protocol_config_sha256",
    "query_id_salt_sha256",
    "visibility",
    "modality",
    "prediction_domain",
    "scene_domains",
    "queries",
}
IMAGE_SCENE_DOMAIN_KEYS = {
    "scene_id",
    "mesh_xyz_path",
    "mesh_xyz_sha256",
    "mesh_vertices",
}
IMAGE_QUERY_KEYS = {
    "query_id",
    "scene_id",
    "modality",
    "crop_rgb_path",
    "crop_rgb_sha256",
    "available_method_inputs",
}
IMAGE_AVAILABLE_METHOD_INPUTS = ["scene_id", "crop_rgb"]
CALIBRATION_SCHEMA_VERSION = "scannet_uqis_ludvig_global_sigmoid_v1"
CALIBRATION_KEYS = {
    "schema_version",
    "benchmark_version",
    "modality",
    "scope",
    "fit_split_role",
    "frozen",
    "scale",
    "bias",
}
_QUERY_ID_PATTERN = re.compile(r"uq_[0-9a-f]{32}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
EXACT_CANDIDATE_K = 64
EXACT_QUERY_CROP_SIZE = 224
EXACT_FEATURE_DIMENSION = 40
EXACT_READOUT_EPSILON = 1e-6


def _implementation_binding(argv: Sequence[str]) -> dict[str, Any]:
    adapter_path = Path(__file__).resolve()
    repository_root = adapter_path.parents[3]
    cli_path = repository_root / "reproductions" / "ludvig" / "run_uqis_image.py"
    sources = [
        {
            "role": "adapter",
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
        }
    ]
    if cli_path.is_file():
        sources.append(
            {
                "role": "cli",
                "path": str(cli_path.resolve()),
                "sha256": sha256_file(cli_path),
            }
        )
    return {
        "sources": sources,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "argv": list(map(str, argv)),
    }


@dataclass(frozen=True)
class FrozenSigmoidCalibration:
    """Frozen global monotonic map from readout logits to probabilities."""

    scale: float = 1.0
    bias: float = 0.0
    fit_split_role: str = field(default="protocol_default", compare=False)
    artifact_path: str | None = field(default=None, compare=False)
    artifact_sha256: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or float(self.scale) <= 0.0:
            raise ValueError("frozen sigmoid scale must be finite and positive")
        if not np.isfinite(self.bias):
            raise ValueError("frozen sigmoid bias must be finite")
        if self.fit_split_role not in {"protocol_default", "dev"}:
            if self.fit_split_role == "test":
                raise ValueError("test-split fitting is forbidden")
            raise ValueError("frozen sigmoid fit split must be dev or protocol_default")
        if (self.artifact_path is None) != (self.artifact_sha256 is None):
            raise ValueError("calibration artifact path/hash must be supplied together")
        if self.artifact_sha256 is not None:
            _require_sha256(self.artifact_sha256, "calibration artifact hash")

    def apply(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("readout logits must be finite")
        transformed = float(self.scale) * values + float(self.bias)
        # This branch-stable form avoids overflow without clipping the result.
        probabilities = np.empty_like(transformed)
        positive = transformed >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-transformed[positive]))
        exponential = np.exp(transformed[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
        return probabilities.astype(np.float32)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "kind": "global_monotonic_sigmoid",
            "frozen": True,
            "fit_split_role": self.fit_split_role,
            "scale": float(self.scale),
            "bias": float(self.bias),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "test_split_fitting_allowed": False,
        }


DEFAULT_SIGMOID_CALIBRATION = FrozenSigmoidCalibration()


@dataclass(frozen=True)
class ExactLudvigImageAdapterConfig:
    """Inputs for the hash-bound LUDVIG CUDA image-query runtime."""

    query_manifest_path: Path
    workspace_receipt_path: Path
    phase_b_dir: Path
    expected_phase_b_manifest_sha256: str
    phase_c_dir: Path
    expected_phase_c_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    driver_library_dir: Path = Path("/root/baselines/LUDVIG/.driver535")
    device: str = "cuda:0"
    calibration: FrozenSigmoidCalibration = DEFAULT_SIGMOID_CALIBRATION
    chunk_size: int = 65_536


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label}: method-visible fields changed")


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validated_file(path_value: Any, digest_value: Any, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    expected = _require_sha256(digest_value, f"{label} hash")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")
    return path


def validate_image_method_manifest(path: str | Path) -> dict[str, Any]:
    """Validate exactly the method-visible UQIS image-query contract.

    Exact key allowlists are deliberate: an evaluator-only target identity or
    any future private field makes the adapter fail closed instead of silently
    becoming an additional model input.
    """

    manifest_path = Path(path).resolve()
    payload = _load_json_object(manifest_path, "image query method manifest")
    _require_exact_keys(payload, IMAGE_MANIFEST_KEYS, "image method manifest")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("image method manifest benchmark version changed")
    expected_protocol = json.loads(json.dumps(asdict(UQISProtocolConfig())))
    if payload.get("protocol_config") != expected_protocol:
        raise ValueError("image method manifest protocol config changed")
    if payload.get("protocol_config_sha256") != canonical_json_sha256(
        payload["protocol_config"]
    ):
        raise ValueError("image method manifest protocol config digest changed")
    if (
        payload.get("release_tier") != "pilot_harness"
        or payload.get("formal_benchmark_eligible") is not False
    ):
        raise ValueError("initial LUDVIG adapter accepts only non-formal pilot harnesses")
    _require_sha256(payload.get("query_id_salt_sha256"), "query ID salt hash")
    if payload.get("visibility") != "method_input":
        raise ValueError("image method manifest is not method-visible")
    if payload.get("modality") != "image":
        raise ValueError("image method manifest modality changed")
    if payload.get("prediction_domain") != PREDICTION_DOMAIN:
        raise ValueError("image method manifest prediction domain changed")
    if payload.get("split_role") not in {"pilot", "dev", "test"}:
        raise ValueError("image method manifest split role is invalid")

    domain_rows = payload.get("scene_domains")
    if not isinstance(domain_rows, list) or len(domain_rows) != 1:
        raise ValueError(
            "LUDVIG image execution requires one scene in one-query workspace"
        )
    scene_domains: dict[str, dict[str, Any]] = {}
    normalized_domains: list[dict[str, Any]] = []
    for index, raw_domain in enumerate(domain_rows):
        if not isinstance(raw_domain, Mapping):
            raise ValueError(f"scene domain {index} must be a JSON object")
        _require_exact_keys(raw_domain, IMAGE_SCENE_DOMAIN_KEYS, "public scene domain")
        scene_id = str(raw_domain.get("scene_id", ""))
        if not scene_id or scene_id in scene_domains:
            raise ValueError("image method manifest has invalid/duplicate scene IDs")
        mesh_path = _validated_file(
            raw_domain.get("mesh_xyz_path"),
            raw_domain.get("mesh_xyz_sha256"),
            f"{scene_id} public mesh domain",
        )
        mesh_xyz = np.load(mesh_path, allow_pickle=False)
        if (
            mesh_xyz.ndim != 2
            or mesh_xyz.shape[1] != 3
            or not np.isfinite(mesh_xyz).all()
        ):
            raise ValueError(f"{scene_id}: mesh_xyz must be finite [V,3]")
        vertices = int(raw_domain.get("mesh_vertices", -1))
        if vertices != int(mesh_xyz.shape[0]):
            raise ValueError(f"{scene_id}: public mesh vertex count changed")
        normalized = {
            "scene_id": scene_id,
            "mesh_xyz_path": str(mesh_path),
            "mesh_xyz_sha256": str(raw_domain["mesh_xyz_sha256"]),
            "mesh_vertices": vertices,
        }
        scene_domains[scene_id] = normalized
        normalized_domains.append(normalized)

    query_rows = payload.get("queries")
    if not isinstance(query_rows, list) or len(query_rows) != 1:
        raise ValueError(
            "LUDVIG image execution requires exactly one query per workspace"
        )
    queries: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for index, raw_query in enumerate(query_rows):
        if not isinstance(raw_query, Mapping):
            raise ValueError(f"image query {index} must be a JSON object")
        _require_exact_keys(raw_query, IMAGE_QUERY_KEYS, "image query")
        query_id = str(raw_query.get("query_id", ""))
        if _QUERY_ID_PATTERN.fullmatch(query_id) is None or query_id in query_ids:
            raise ValueError("image method manifest has invalid/duplicate query IDs")
        query_ids.add(query_id)
        scene_id = str(raw_query.get("scene_id", ""))
        if scene_id not in scene_domains:
            raise ValueError(f"{query_id}: unknown scene")
        if raw_query.get("modality") != "image":
            raise ValueError(f"{query_id}: modality changed")
        if raw_query.get("available_method_inputs") != IMAGE_AVAILABLE_METHOD_INPUTS:
            raise ValueError(f"{query_id}: available method inputs changed")
        crop_path = _validated_file(
            raw_query.get("crop_rgb_path"),
            raw_query.get("crop_rgb_sha256"),
            f"{query_id} crop RGB",
        )
        try:
            with Image.open(crop_path) as image:
                image.load()
                expected_size = int(payload["protocol_config"]["crop_size_px"])
                if image.mode != "RGB" or image.size != (expected_size, expected_size):
                    raise ValueError(
                        f"{query_id}: crop must be {expected_size}x{expected_size} RGB"
                    )
        except OSError as error:
            raise ValueError(f"{query_id}: crop RGB is unreadable") from error
        queries.append(
            {
                "query_id": query_id,
                "scene_id": scene_id,
                "modality": "image",
                "crop_rgb_path": str(crop_path),
                "crop_rgb_sha256": str(raw_query["crop_rgb_sha256"]),
                "available_method_inputs": list(IMAGE_AVAILABLE_METHOD_INPUTS),
            }
        )

    if [row["scene_id"] for row in normalized_domains] != sorted(scene_domains):
        raise ValueError("image method manifest scene domains are not sorted")
    if [row["query_id"] for row in queries] != sorted(query_ids):
        raise ValueError("image method manifest queries are not sorted")
    return {
        "path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_version": BENCHMARK_VERSION,
        "split_role": str(payload["split_role"]),
        "protocol_config": dict(payload["protocol_config"]),
        "scene_domains": scene_domains,
        "queries": queries,
    }


def load_frozen_sigmoid_calibration(path: str | Path) -> FrozenSigmoidCalibration:
    """Load a global sigmoid fitted only on the development split.

    The adapter exposes no fitting operation.  In particular, a record that
    claims test-split fitting is rejected before any parameter is consumed.
    """

    calibration_path = Path(path).resolve()
    payload = _load_json_object(calibration_path, "frozen sigmoid calibration")
    _require_exact_keys(payload, CALIBRATION_KEYS, "frozen sigmoid calibration")
    if payload.get("fit_split_role") == "test":
        raise ValueError("test-split fitting is forbidden")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("frozen sigmoid calibration schema changed")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("frozen sigmoid calibration benchmark changed")
    if payload.get("modality") != "image" or payload.get("scope") != "global":
        raise ValueError("calibration must be global and image-only")
    if payload.get("fit_split_role") != "dev":
        raise ValueError("frozen sigmoid calibration must be fit on dev")
    if payload.get("frozen") is not True:
        raise ValueError("sigmoid calibration must be frozen")
    return FrozenSigmoidCalibration(
        scale=float(payload.get("scale")),
        bias=float(payload.get("bias")),
        fit_split_role="dev",
        artifact_path=str(calibration_path),
        artifact_sha256=sha256_file(calibration_path),
    )


def _finite_array(value: Any, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    expected_ndim = 1 + len(shape_tail)
    if array.ndim != expected_ndim or tuple(array.shape[1:]) != shape_tail:
        suffix = ",".join(map(str, shape_tail))
        raise ValueError(f"{label} must have shape [N,{suffix}]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric data")
    return np.asarray(array, dtype=np.float64)


def score_mesh_probabilities(
    *,
    gaussian_features: np.ndarray,
    query_descriptor: np.ndarray,
    gaussian_xyz: np.ndarray,
    gaussian_covariance: np.ndarray,
    gaussian_opacity: np.ndarray,
    mesh_xyz: np.ndarray,
    calibration: FrozenSigmoidCalibration = DEFAULT_SIGMOID_CALIBRATION,
    candidate_k: int = 64,
    candidate_indices: np.ndarray | None = None,
    chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one image descriptor on the official mesh-vertex domain.

    Primitive logits are cosine similarities in LUDVIG's 40-D scene PCA
    space.  Mesh logits use the same normalized opacity-weighted Gaussian
    kernel as :func:`continuous_gaussian_readout`.  This NumPy implementation
    is dependency-light and intended for validation and synthetic smoke runs;
    the exact CUDA runtime calls the repository implementation directly.
    """

    features = np.asarray(gaussian_features)
    descriptor = np.asarray(query_descriptor)
    if features.ndim != 2 or features.shape[1] != 40 or descriptor.shape != (40,):
        raise ValueError("Gaussian features and image descriptor must be exactly 40-D")
    if not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all():
        raise ValueError("Gaussian features must be finite numeric data")
    if (
        not np.issubdtype(descriptor.dtype, np.number)
        or not np.isfinite(descriptor).all()
    ):
        raise ValueError("image descriptor must be finite numeric data")
    xyz = _finite_array(gaussian_xyz, (3,), "gaussian_xyz")
    covariance = _finite_array(gaussian_covariance, (3, 3), "gaussian_covariance")
    points = _finite_array(mesh_xyz, (3,), "mesh_xyz")
    count = int(features.shape[0])
    if count <= 0 or xyz.shape[0] != count or covariance.shape[0] != count:
        raise ValueError("Gaussian feature/geometry rows must be non-empty and aligned")
    opacity = np.asarray(gaussian_opacity)
    if opacity.size != count or opacity.ndim not in {1, 2}:
        raise ValueError("gaussian_opacity must align with Gaussian rows")
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if not np.isfinite(opacity).all() or bool((opacity < 0.0).any()):
        raise ValueError("gaussian_opacity must be finite and non-negative")
    if points.shape[0] <= 0:
        raise ValueError("official mesh domain must be non-empty")
    if not isinstance(calibration, FrozenSigmoidCalibration):
        raise TypeError("calibration must be a FrozenSigmoidCalibration")
    if int(candidate_k) <= 0 or int(chunk_size) <= 0:
        raise ValueError("candidate_k and chunk_size must be positive")

    descriptor64 = np.asarray(descriptor, dtype=np.float64)
    descriptor_norm = float(np.linalg.norm(descriptor64))
    if descriptor_norm <= 1e-12:
        raise ValueError("image descriptor must have non-zero norm")
    feature64 = np.asarray(features, dtype=np.float64)
    feature_norm = np.linalg.norm(feature64, axis=1)
    normalized_features = feature64 / np.maximum(feature_norm[:, None], 1e-12)
    primitive_logits = normalized_features @ (descriptor64 / descriptor_norm)
    if not np.isfinite(primitive_logits).all():
        raise ValueError("primitive cosine logits are non-finite")

    identity = np.eye(3, dtype=np.float64)
    precision = np.linalg.pinv(covariance + 1e-6 * identity[None])
    if not np.isfinite(precision).all():
        raise ValueError("Gaussian precision is non-finite")

    if candidate_indices is None:
        try:
            from scipy.spatial import cKDTree
        except ImportError as error:  # pragma: no cover - production dependency
            raise RuntimeError(
                "scipy is required to build Gaussian candidates"
            ) from error
        selected_k = min(int(candidate_k), count)
        _distance, indices = cKDTree(xyz).query(points, k=selected_k, workers=-1)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
    else:
        raw_indices = np.asarray(candidate_indices)
        if raw_indices.dtype.kind not in "iu":
            raise ValueError("candidate_indices must contain integers")
        indices = np.asarray(raw_indices, dtype=np.int64)
        if (
            indices.ndim != 2
            or indices.shape[0] != points.shape[0]
            or indices.shape[1] == 0
        ):
            raise ValueError("candidate_indices must be a non-empty [V,K] matrix")
        if bool((indices < 0).any()) or bool((indices >= count).any()):
            raise IndexError("candidate_indices contains an invalid Gaussian row")

    mesh_logits = np.empty(points.shape[0], dtype=np.float64)
    support = np.empty(points.shape[0], dtype=np.float64)
    for begin in range(0, points.shape[0], int(chunk_size)):
        end = min(begin + int(chunk_size), points.shape[0])
        local_indices = indices[begin:end]
        delta = xyz[local_indices] - points[begin:end, None, :]
        mahalanobis = np.einsum(
            "vki,vkij,vkj->vk",
            delta,
            precision[local_indices],
            delta,
            optimize=True,
        )
        local_opacity = opacity[local_indices]
        log_weights = np.full_like(mahalanobis, -np.inf, dtype=np.float64)
        positive_opacity = local_opacity > 0.0
        log_weights[positive_opacity] = (
            -0.5 * np.maximum(mahalanobis[positive_opacity], 0.0)
            + np.log(local_opacity[positive_opacity])
        )
        maximum = np.max(log_weights, axis=1)
        if not np.isfinite(maximum).all():
            raise ValueError("official mesh vertex has no finite Gaussian support")
        weights = np.exp(log_weights - maximum[:, None])
        local_normalizer = weights.sum(axis=1)
        if bool((local_normalizer <= 0.0).any()) or not np.isfinite(
            local_normalizer
        ).all():
            raise ValueError("official mesh vertex has no finite Gaussian support")
        mesh_logits[begin:end] = (
            weights * primitive_logits[local_indices]
        ).sum(axis=1) / local_normalizer
        log_support = maximum + np.log(local_normalizer)
        support[begin:end] = np.exp(
            np.clip(
                log_support,
                np.log(np.finfo(np.float32).tiny),
                np.log(np.finfo(np.float32).max),
            )
        )

    probabilities = calibration.apply(mesh_logits)
    if (
        probabilities.shape != (points.shape[0],)
        or not np.isfinite(probabilities).all()
        or bool((probabilities < 0.0).any())
        or bool((probabilities > 1.0).any())
    ):
        raise ValueError("adapter probabilities must be finite and in [0,1]")
    return probabilities, support.astype(np.float32)


def _array_binding(root: Path, path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_precomputed_image_adapter(
    *,
    query_manifest_path: str | Path,
    scene_id: str,
    gaussian_features: np.ndarray,
    query_descriptors: Mapping[str, np.ndarray],
    gaussian_xyz: np.ndarray,
    gaussian_covariance: np.ndarray,
    gaussian_opacity: np.ndarray,
    output_dir: str | Path,
    calibration: FrozenSigmoidCalibration = DEFAULT_SIGMOID_CALIBRATION,
    candidate_indices: np.ndarray | None = None,
    candidate_k: int = 64,
    chunk_size: int = 65_536,
) -> dict[str, Any]:
    """Run a dependency-light precomputed 40-D adapter smoke.

    This seam deliberately accepts already-computed query descriptors and
    Gaussian geometry.  It exercises the complete protocol/scoring/output
    path but is marked result-ineligible and is not the exact LUDVIG runtime.
    """

    method = validate_image_method_manifest(query_manifest_path)
    requested_scene = str(scene_id)
    if requested_scene not in method["scene_domains"]:
        raise ValueError(f"unknown image-adapter scene: {requested_scene}")
    queries = [
        query for query in method["queries"] if query["scene_id"] == requested_scene
    ]
    expected_ids = {query["query_id"] for query in queries}
    supplied_ids = {str(query_id) for query_id in query_descriptors}
    if supplied_ids != expected_ids:
        raise ValueError(
            "precomputed image descriptor inventory is incomplete or unexpected"
        )

    domain = method["scene_domains"][requested_scene]
    mesh_xyz = np.load(Path(domain["mesh_xyz_path"]), allow_pickle=False)
    if tuple(mesh_xyz.shape) != (int(domain["mesh_vertices"]), 3):
        raise ValueError("official mesh domain changed after manifest validation")

    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite adapter output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.uqis_ludvig_tmp_", dir=output.parent)
    )
    query_records: list[dict[str, Any]] = []
    try:
        for query in queries:
            query_id = query["query_id"]
            probabilities, support = score_mesh_probabilities(
                gaussian_features=gaussian_features,
                query_descriptor=query_descriptors[query_id],
                gaussian_xyz=gaussian_xyz,
                gaussian_covariance=gaussian_covariance,
                gaussian_opacity=gaussian_opacity,
                mesh_xyz=mesh_xyz,
                calibration=calibration,
                candidate_k=candidate_k,
                candidate_indices=candidate_indices,
                chunk_size=chunk_size,
            )
            probability_path = temporary / f"{query_id}.npy"
            np.save(probability_path, probabilities, allow_pickle=False)
            query_records.append(
                {
                    "query_id": query_id,
                    "scene_id": requested_scene,
                    "probability": _array_binding(
                        temporary, probability_path, probabilities
                    ),
                    "support": {
                        "minimum": float(support.min()),
                        "maximum": float(support.max()),
                        "positive_vertices": int(np.count_nonzero(support > 0.0)),
                    },
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": "scannet_uqis_ludvig_image_adapter_v1",
            "status": "synthetic_precomputed_smoke_complete",
            "result_eligible": False,
            "formal_benchmark_row_eligible": False,
            "runtime_mode": "synthetic_precomputed_40d_smoke",
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "benchmark_local_adapter": True,
            "implementation": _implementation_binding(()),
            "benchmark_version": method["benchmark_version"],
            "split_role": method["split_role"],
            "modality": "image",
            "scene_id": requested_scene,
            "prediction_domain": PREDICTION_DOMAIN,
            "query_manifest": {
                "path": method["path"],
                "sha256": method["manifest_sha256"],
            },
            "official_mesh_domain": dict(domain),
            "privacy_boundary": {
                "evaluator_manifest_opened": False,
                "private_target_inputs_opened": False,
                "method_visible_inputs_only": True,
            },
            "calibration": calibration.to_record(),
            "protocol": {
                "feature_dimension": 40,
                "primitive_response": "cosine_similarity",
                "mesh_readout": "continuous_opacity_weighted_gaussian",
                "candidate_k": int(candidate_k),
                "probability_map": "frozen_global_monotonic_sigmoid",
                "test_time_fitting": False,
            },
            "queries": query_records,
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise ValueError(f"refusing concurrent adapter overwrite: {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _safe_bound_artifact(
    root: Path, binding: Any, label: str
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"missing {label} artifact binding")
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} artifact path escapes its attempt")
    path = _validated_file(root / relative, binding.get("sha256"), label)
    return path, binding


def run_exact_ludvig_image_adapter(
    config: ExactLudvigImageAdapterConfig,
    *,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Encode public crops with exact LUDVIG DINO/PCA and score mesh vertices.

    Heavy CUDA, LUDVIG, and ``gsplat`` imports are deliberately local to this
    function.  Phase B and Phase C are consumed read-only through their
    hash-bound manifests; no legacy Phase A--E contract is modified.
    """

    try:
        from scipy.spatial import cKDTree
        import torch
        import torch.nn.functional as torch_functional

        from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
            DinoPatchPredictor,
            PhaseBConfig,
            apply_scene_pca_transform,
            audit_cuda_driver_binding,
            audit_ludvig_vendored_dinov2_source,
            audit_model_architecture,
            build_ludvig_vendored_vitg14,
            extract_view_tokens,
            load_checkpoint_exact_ludvig_vendored,
            load_phase_b_transform,
            ludvig_sliding_plan,
        )
        from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import (
            PHASE_C_SCHEMA_VERSION,
            PHASE_C_STATUS,
            _import_ludvig_geometry,
        )
        from radio_gs.benchmarks.scannet_uqis.ludvig_dino_uplift import (
            SCHEMA_VERSION as UQIS_DINO_UPLIFT_SCHEMA_VERSION,
            STATUS as UQIS_DINO_UPLIFT_STATUS,
        )
        from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import (
            center_3x3_descriptor,
            gaussian_covariances,
        )
    except Exception as error:  # pragma: no cover - depends on exact CUDA env
        raise RuntimeError(
            "exact LUDVIG image runtime dependencies are unavailable"
        ) from error

    output = config.output_dir.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite adapter output: {output}")
    if int(config.chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(config.calibration, FrozenSigmoidCalibration):
        raise TypeError("calibration must be frozen before exact runtime")
    method = validate_image_method_manifest(config.query_manifest_path)
    workspace_receipt_path = config.workspace_receipt_path.resolve()
    workspace_receipt = _load_json_object(
        workspace_receipt_path, "one-query workspace receipt"
    )
    _require_exact_keys(
        workspace_receipt,
        {
            "schema_version",
            "status",
            "formal_benchmark_eligible",
            "benchmark_version",
            "release_manifest_sha256",
            "source_query_manifest_sha256",
            "workspace_query_manifest_sha256",
            "query_id",
            "scene_id",
            "modality",
            "query_count",
            "independent_workspace_required",
            "mount_policy",
            "evaluator_private_files_staged",
        },
        "one-query workspace receipt",
    )
    if (
        workspace_receipt["schema_version"] != "scannet_uqis_query_workspace_v1"
        or workspace_receipt["status"] != "staged"
        or workspace_receipt["benchmark_version"] != method["benchmark_version"]
        or workspace_receipt["workspace_query_manifest_sha256"]
        != method["manifest_sha256"]
        or workspace_receipt["modality"] != "image"
        or workspace_receipt["query_count"] != 1
        or workspace_receipt["independent_workspace_required"] is not True
        or workspace_receipt["mount_policy"] != "workspace_only_read_only"
        or workspace_receipt["evaluator_private_files_staged"] is not False
        or len(method["queries"]) != 1
        or workspace_receipt["query_id"] != method["queries"][0]["query_id"]
        or workspace_receipt["scene_id"] != method["queries"][0]["scene_id"]
    ):
        raise ValueError("one-query workspace receipt does not bind this invocation")

    phase_c_root = config.phase_c_dir.resolve()
    phase_c_path = phase_c_root / "run_manifest.json"
    _validated_file(
        phase_c_path,
        config.expected_phase_c_manifest_sha256,
        "Phase-C run manifest",
    )
    phase_c = _load_json_object(phase_c_path, "Phase-C run manifest")
    legacy_phase_c = phase_c.get("schema_version") == PHASE_C_SCHEMA_VERSION
    uqis_uplift = phase_c.get("schema_version") == UQIS_DINO_UPLIFT_SCHEMA_VERSION
    if not (legacy_phase_c or uqis_uplift):
        raise ValueError("unsupported Phase-C/UQIS uplift schema")
    expected_status = PHASE_C_STATUS if legacy_phase_c else UQIS_DINO_UPLIFT_STATUS
    if phase_c.get("status") != expected_status or phase_c.get("result_eligible") is not False:
        raise ValueError("Phase-C status changed")
    if uqis_uplift and (
        phase_c.get("benchmark_version") != method["benchmark_version"]
        or phase_c.get("field_construction_eligible") is not True
        or phase_c.get("field_role") != "ludvig_dino_for_image_point_2d_point_3d"
    ):
        raise ValueError("UQIS DINO field authority or role changed")
    scene_id = str(phase_c.get("scene_id", ""))
    if scene_id not in method["scene_domains"]:
        raise ValueError("Phase-C scene is absent from the image method manifest")
    records = [query for query in method["queries"] if query["scene_id"] == scene_id]
    if not records:
        raise ValueError("image method manifest has no queries for the Phase-C scene")

    phase_c_phase_b = phase_c.get("phase_b")
    if not isinstance(phase_c_phase_b, Mapping):
        raise ValueError("Phase C lacks its Phase-B binding")
    phase_b_root = config.phase_b_dir.resolve()
    bound_phase_b_root = Path(str(phase_c_phase_b.get("root", ""))).resolve()
    expected_phase_b_hash = _require_sha256(
        config.expected_phase_b_manifest_sha256, "expected Phase-B manifest hash"
    )
    if bound_phase_b_root != phase_b_root:
        raise ValueError("Phase-C Phase-B root differs from the requested Phase B")
    if str(phase_c_phase_b.get("manifest_sha256", "")) != expected_phase_b_hash:
        raise ValueError("Phase-C Phase-B hash differs from the requested Phase B")
    transform, phase_b = load_phase_b_transform(
        phase_b_root, expected_manifest_sha256=expected_phase_b_hash
    )
    if str(phase_b.get("scene_id", "")) != scene_id:
        raise ValueError("Phase-B and Phase-C scenes differ")

    feature_path, feature_binding = _safe_bound_artifact(
        phase_c_root,
        phase_c.get("artifacts", {}).get("gaussian_features"),
        "Phase-C Gaussian features",
    )
    feature_array = np.load(feature_path, allow_pickle=False)
    if (
        feature_array.ndim != 2
        or feature_array.shape[1] != EXACT_FEATURE_DIMENSION
        or feature_array.dtype != np.float32
        or not np.isfinite(feature_array).all()
        or list(feature_array.shape) != feature_binding.get("shape")
        or str(feature_array.dtype) != feature_binding.get("dtype")
    ):
        raise ValueError("Phase-C Gaussian features must be finite float32 [G,40]")

    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("exact LUDVIG image query extraction requires CUDA")
    driver_config = PhaseBConfig(
        phase_a_dir=Path("."),
        expected_phase_a_manifest_sha256="0" * 64,
        dino_checkpoint=Path("."),
        ludvig_upstream=config.ludvig_upstream,
        source_adapter_ledger=Path("."),
        dinov2_source=config.ludvig_upstream,
        output_dir=config.output_dir,
        driver_library_dir=config.driver_library_dir,
        device=config.device,
    )
    driver = audit_cuda_driver_binding(driver_config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    checkpoint_record = phase_b.get("phase_a", {}).get("checkpoint", {})
    checkpoint_path = _validated_file(
        checkpoint_record.get("path"),
        checkpoint_record.get("sha256"),
        "official DINO checkpoint",
    )
    source = audit_ludvig_vendored_dinov2_source(
        config.ludvig_upstream,
        expected_tree_sha256=str(
            phase_b.get("ludvig_vendored_dinov2_source", {}).get(
                "audited_files_sha256", ""
            )
        ),
    )

    started = time.monotonic()
    model = build_ludvig_vendored_vitg14(config.ludvig_upstream)
    architecture = audit_model_architecture(model, driver_config)
    checkpoint_load = load_checkpoint_exact_ludvig_vendored(model, checkpoint_path)
    predictor = DinoPatchPredictor(model, device)
    crop_size = int(method["protocol_config"]["crop_size_px"])
    if crop_size != EXACT_QUERY_CROP_SIZE:
        raise ValueError(f"exact adapter is frozen to {EXACT_QUERY_CROP_SIZE}px crops")
    query_plan = ludvig_sliding_plan(crop_size, crop_size)
    if (
        query_plan["patch_count"] != 1
        or query_plan["token_grid_height"] != 16
        or query_plan["token_grid_width"] != 16
    ):
        raise ValueError("UQIS query token geometry changed")
    descriptors: list[np.ndarray] = []
    query_records: list[dict[str, Any]] = []
    for record in records:
        raw = extract_view_tokens(
            Path(record["crop_rgb_path"]),
            predictor,
            query_plan,
            expected_embedding_dim=1536,
        )
        weighted = apply_scene_pca_transform(raw, transform, eigval_weighting=True)
        descriptor = center_3x3_descriptor(weighted)
        if descriptor.shape != (EXACT_FEATURE_DIMENSION,) or not np.isfinite(
            descriptor
        ).all():
            raise ValueError("LUDVIG image descriptor is not finite 40-D")
        descriptor_norm = float(np.linalg.norm(descriptor))
        if descriptor_norm <= 1e-12:
            raise ValueError("LUDVIG image descriptor has zero norm")
        descriptors.append(descriptor)
        query_records.append(
            {
                "query_id": record["query_id"],
                "scene_id": scene_id,
                "crop_rgb_sha256": record["crop_rgb_sha256"],
                "raw_token_shape": list(raw.shape),
                "weighted_token_shape": list(weighted.shape),
                "descriptor_l2_norm": descriptor_norm,
            }
        )
    descriptor_array = np.stack(descriptors).astype(np.float32, copy=False)
    del predictor, model
    gc.collect()
    torch.cuda.empty_cache()

    if uqis_uplift:
        geometry_path, geometry = _safe_bound_artifact(
            phase_c_root,
            phase_c.get("artifacts", {}).get("pruned_geometry"),
            "UQIS pruned Gaussian geometry",
        )
    else:
        geometry = phase_c.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("Phase C lacks its Gaussian geometry binding")
        geometry_path = _validated_file(
            geometry.get("path"), geometry.get("sha256"), "Gaussian geometry"
        )
    GaussianModel, _CamScene = _import_ludvig_geometry(config.ludvig_upstream)
    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    gaussian_xyz = gaussian.get_xyz.detach().float()
    if len(gaussian_xyz) != int(feature_array.shape[0]):
        raise ValueError("Phase-C features and Gaussian geometry rows differ")
    covariance = gaussian_covariances(gaussian.get_scaling, gaussian.get_rotation)
    opacity = gaussian.get_opacity.detach().float().reshape(-1)
    field_features = torch_functional.normalize(
        torch.from_numpy(feature_array).to(device), dim=-1, eps=1e-8
    )
    query_tensors = torch.from_numpy(descriptor_array).to(device)
    primitive_logits = field_features @ query_tensors.T
    precision = torch.linalg.pinv(
        covariance
        + EXACT_READOUT_EPSILON * torch.eye(3, device=device, dtype=torch.float32)[None]
    )

    domain = method["scene_domains"][scene_id]
    mesh_xyz = np.load(Path(domain["mesh_xyz_path"]), allow_pickle=False)
    if tuple(mesh_xyz.shape) != (int(domain["mesh_vertices"]), 3):
        raise ValueError("official mesh domain changed after manifest validation")
    mesh_xyz = np.asarray(mesh_xyz, dtype=np.float32)
    tree = cKDTree(gaussian_xyz.detach().cpu().numpy())
    mesh_logits = np.empty((len(records), len(mesh_xyz)), dtype=np.float32)
    support_array = np.empty(len(mesh_xyz), dtype=np.float32)
    log_support_array = np.empty(len(mesh_xyz), dtype=np.float32)
    for begin in range(0, len(mesh_xyz), int(config.chunk_size)):
        end = min(begin + int(config.chunk_size), len(mesh_xyz))
        points_np = mesh_xyz[begin:end]
        _distances, indices_np = tree.query(
            points_np, k=min(EXACT_CANDIDATE_K, len(gaussian_xyz)), workers=-1
        )
        indices_np = np.asarray(indices_np, dtype=np.int64)
        if indices_np.ndim == 1:
            indices_np = indices_np[:, None]
        points = torch.from_numpy(np.ascontiguousarray(points_np)).to(device)
        indices = torch.from_numpy(np.ascontiguousarray(indices_np)).to(device)
        with torch.no_grad():
            delta = gaussian_xyz[indices] - points[:, None, :]
            mahalanobis = torch.einsum(
                "vki,vkij,vkj->vk",
                delta,
                precision[indices],
                delta,
            )
            local_opacity = opacity[indices]
            log_weights = torch.full_like(mahalanobis, -torch.inf)
            positive_opacity = local_opacity > 0
            log_weights[positive_opacity] = (
                -0.5 * mahalanobis[positive_opacity].clamp_min(0.0)
                + torch.log(local_opacity[positive_opacity])
            )
            maximum = log_weights.max(dim=1).values
            if not bool(torch.isfinite(maximum).all()):
                raise ValueError("official mesh domain has unsupported Gaussian points")
            stable_weights = torch.exp(log_weights - maximum[:, None])
            normalizer = stable_weights.sum(dim=1)
            if not bool(torch.isfinite(normalizer).all()) or not bool(
                (normalizer > 0).all()
            ):
                raise ValueError("official mesh domain has unsupported Gaussian points")
            normalized_weights = stable_weights / normalizer[:, None]
            log_support = maximum + torch.log(normalizer)
            finfo = torch.finfo(torch.float32)
            support = torch.exp(
                log_support.clamp(min=float(np.log(finfo.tiny)), max=float(np.log(finfo.max)))
            )
            support_array[begin:end] = (
                support.cpu().numpy().astype(np.float32, copy=False)
            )
            log_support_array[begin:end] = (
                log_support.cpu().numpy().astype(np.float32, copy=False)
            )
            for query_index in range(len(records)):
                scores = (
                    normalized_weights
                    * primitive_logits[indices, query_index]
                ).sum(dim=1)
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError(
                        "log-stable Gaussian readout produced non-finite scores"
                    )
                mesh_logits[query_index, begin:end] = (
                    scores.cpu().numpy().astype(np.float32, copy=False)
                )

    probability_arrays = [config.calibration.apply(logits) for logits in mesh_logits]
    if any(
        values.shape != (len(mesh_xyz),)
        or not np.isfinite(values).all()
        or bool(((values < 0.0) | (values > 1.0)).any())
        for values in probability_arrays
    ):
        raise ValueError("exact adapter produced invalid mesh probabilities")
    elapsed = time.monotonic() - started
    peak = {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.uqis_ludvig_tmp_", dir=output.parent)
    )
    try:
        adapter_dir = temporary / "adapter"
        adapter_dir.mkdir()
        descriptor_path = adapter_dir / "query_descriptors.npy"
        support_path = adapter_dir / "mesh_support.npy"
        log_support_path = adapter_dir / "mesh_log_support.npy"
        np.save(descriptor_path, descriptor_array, allow_pickle=False)
        np.save(support_path, support_array, allow_pickle=False)
        np.save(log_support_path, log_support_array, allow_pickle=False)
        predictions: list[dict[str, Any]] = []
        for record, probabilities in zip(query_records, probability_arrays):
            probability_path = temporary / f"{record['query_id']}.npy"
            np.save(probability_path, probabilities, allow_pickle=False)
            predictions.append(
                {
                    **record,
                    "probability": _array_binding(
                        temporary, probability_path, probabilities
                    ),
                }
            )
        manifest = {
            "schema_version": "scannet_uqis_ludvig_image_adapter_v1",
            "status": "exact_runtime_smoke_complete",
            "result_eligible": False,
            "formal_benchmark_row_eligible": False,
            "runtime_mode": "exact_ludvig_dinov2_phase_b_pca_phase_c_field",
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "benchmark_local_adapter": True,
            "implementation": _implementation_binding(argv),
            "benchmark_version": method["benchmark_version"],
            "split_role": method["split_role"],
            "modality": "image",
            "scene_id": scene_id,
            "prediction_domain": PREDICTION_DOMAIN,
            "query_manifest": {
                "path": method["path"],
                "sha256": method["manifest_sha256"],
            },
            "workspace_receipt": {
                "path": str(workspace_receipt_path),
                "sha256": sha256_file(workspace_receipt_path),
            },
            "phase_b": {
                "root": str(phase_b_root),
                "manifest_sha256": expected_phase_b_hash,
            },
            "phase_c": {
                "root": str(phase_c_root),
                "manifest_sha256": str(config.expected_phase_c_manifest_sha256),
                "gaussian_features_sha256": str(feature_binding["sha256"]),
            },
            "official_mesh_domain": dict(domain),
            "privacy_boundary": {
                "evaluator_manifest_opened": False,
                "private_target_inputs_opened": False,
                "method_visible_inputs_only": True,
            },
            "calibration": config.calibration.to_record(),
            "ludvig_vendored_dinov2_source": source,
            "model": architecture,
            "checkpoint_exact_ludvig_load": checkpoint_load,
            "cuda_driver_binding": driver,
            "cuda_peak_memory": peak,
            "protocol": {
                "available_method_inputs": list(IMAGE_AVAILABLE_METHOD_INPUTS),
                "query_alignment": "bilinear_224x224_patch_aligned",
                "query_tokens": "one_16x16_DINOv2_grid",
                "query_scene_transform": (
                    "reuse_frozen_Phase_B_standardization_PCA40_and_singular_values"
                ),
                "query_pooling": "center_3x3_mean_then_l2",
                "primitive_response": "cosine_similarity",
                "mesh_readout": "log_stable_continuous_opacity_weighted_gaussian",
                "readout_precision": "pinv(covariance+1e-6_identity)",
                "normalization": "logsumexp_equivalent_normalized_kernel",
                "candidate_k": EXACT_CANDIDATE_K,
                "support_rule": (
                    "positive_opacity_candidate_set_log_stable_no_tuned_gate"
                ),
                "probability_map": "frozen_global_monotonic_sigmoid",
                "test_time_fitting": False,
            },
            "support": {
                "positive_vertices": int(np.count_nonzero(support_array > 0.0)),
                "total_vertices": int(len(support_array)),
                "minimum": float(support_array.min()),
                "maximum": float(support_array.max()),
                "minimum_log_kernel_mass": float(log_support_array.min()),
                "maximum_log_kernel_mass": float(log_support_array.max()),
            },
            "elapsed_seconds": float(elapsed),
            "artifacts": {
                "query_descriptors": _array_binding(
                    temporary, descriptor_path, descriptor_array
                ),
                "mesh_support": _array_binding(temporary, support_path, support_array),
                "mesh_log_support": _array_binding(
                    temporary, log_support_path, log_support_array
                ),
            },
            "queries": predictions,
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "run_manifest.sha256").write_text(
            sha256_file(manifest_path) + "\n", encoding="ascii"
        )
        if output.exists():
            raise ValueError(f"refusing concurrent adapter overwrite: {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
