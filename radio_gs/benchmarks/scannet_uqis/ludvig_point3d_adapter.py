"""Benchmark-local LUDVIG adapter for a single world-space point prompt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256

from .ludvig_dino_uplift import SCHEMA_VERSION as FIELD_SCHEMA, STATUS as FIELD_STATUS
from .ludvig_image_adapter import (
    DEFAULT_SIGMOID_CALIBRATION,
    IMAGE_MANIFEST_KEYS,
    IMAGE_SCENE_DOMAIN_KEYS,
    FrozenSigmoidCalibration,
    _array_binding,
    _load_json_object,
    _require_exact_keys,
    _require_sha256,
    _safe_bound_artifact,
    _validated_file,
    score_mesh_probabilities,
)
from .protocol import BENCHMARK_VERSION, PREDICTION_DOMAIN, UQISProtocolConfig, sha256_file


QUERY_KEYS = {
    "query_id",
    "scene_id",
    "modality",
    "point_world_xyz",
    "available_method_inputs",
}
AVAILABLE_INPUTS = ["scene_id", "point_world_xyz"]
WORKSPACE_KEYS = {
    "schema_version", "status", "formal_benchmark_eligible", "benchmark_version",
    "release_manifest_sha256", "source_query_manifest_sha256",
    "workspace_query_manifest_sha256", "query_id", "scene_id", "modality",
    "query_count", "independent_workspace_required", "mount_policy",
    "evaluator_private_files_staged",
}


@dataclass(frozen=True)
class Point3DConfig:
    query_manifest_path: Path
    workspace_receipt_path: Path
    field_dir: Path
    expected_field_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    calibration: FrozenSigmoidCalibration = DEFAULT_SIGMOID_CALIBRATION
    candidate_k: int = 64
    chunk_size: int = 65_536


def _validate_workspace(config: Point3DConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    path = config.query_manifest_path.resolve()
    payload = _load_json_object(path, "point_3d query manifest")
    _require_exact_keys(payload, IMAGE_MANIFEST_KEYS, "point_3d method manifest")
    expected_protocol = json.loads(json.dumps(asdict(UQISProtocolConfig())))
    if (
        payload.get("benchmark_version") != BENCHMARK_VERSION
        or payload.get("protocol_config") != expected_protocol
        or payload.get("protocol_config_sha256")
        != canonical_json_sha256(payload["protocol_config"])
        or payload.get("release_tier") != "pilot_harness"
        or payload.get("formal_benchmark_eligible") is not False
        or payload.get("visibility") != "method_input"
        or payload.get("modality") != "point_3d"
        or payload.get("prediction_domain") != PREDICTION_DOMAIN
    ):
        raise ValueError("point_3d method manifest authority changed")
    _require_sha256(payload.get("query_id_salt_sha256"), "query ID salt")
    domains = payload.get("scene_domains")
    queries = payload.get("queries")
    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError("point_3d workspace must contain one scene")
    if not isinstance(queries, list) or len(queries) != 1:
        raise ValueError("point_3d workspace must contain one query")
    domain = domains[0]
    query = queries[0]
    if not isinstance(domain, Mapping) or not isinstance(query, Mapping):
        raise ValueError("point_3d workspace rows must be objects")
    _require_exact_keys(domain, IMAGE_SCENE_DOMAIN_KEYS, "public scene domain")
    _require_exact_keys(query, QUERY_KEYS, "point_3d query")
    if (
        query.get("modality") != "point_3d"
        or query.get("available_method_inputs") != AVAILABLE_INPUTS
        or query.get("scene_id") != domain.get("scene_id")
    ):
        raise ValueError("point_3d query contract changed")
    point = np.asarray(query.get("point_world_xyz"), dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_world_xyz must be finite [3]")
    mesh_path = _validated_file(
        domain.get("mesh_xyz_path"), domain.get("mesh_xyz_sha256"), "public mesh"
    )
    mesh = np.load(mesh_path, allow_pickle=False)
    if (
        mesh.dtype != np.float32
        or mesh.shape != (int(domain.get("mesh_vertices", -1)), 3)
        or not np.isfinite(mesh).all()
    ):
        raise ValueError("public mesh domain changed")

    receipt_path = config.workspace_receipt_path.resolve()
    receipt = _load_json_object(receipt_path, "point_3d workspace receipt")
    _require_exact_keys(receipt, WORKSPACE_KEYS, "point_3d workspace receipt")
    manifest_hash = sha256_file(path)
    if (
        receipt.get("schema_version") != "scannet_uqis_query_workspace_v1"
        or receipt.get("status") != "staged"
        or receipt.get("benchmark_version") != BENCHMARK_VERSION
        or receipt.get("workspace_query_manifest_sha256") != manifest_hash
        or receipt.get("query_id") != query.get("query_id")
        or receipt.get("scene_id") != query.get("scene_id")
        or receipt.get("modality") != "point_3d"
        or receipt.get("query_count") != 1
        or receipt.get("independent_workspace_required") is not True
        or receipt.get("mount_policy") != "workspace_only_read_only"
        or receipt.get("evaluator_private_files_staged") is not False
    ):
        raise ValueError("point_3d workspace receipt does not bind this query")
    return (
        {
            "path": str(path), "sha256": manifest_hash,
            "split_role": str(payload["split_role"]),
            "query": dict(query), "domain": dict(domain), "mesh": mesh,
        },
        {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
    )


def point_descriptor(
    features: np.ndarray,
    xyz: np.ndarray,
    covariance: np.ndarray,
    opacity: np.ndarray,
    point: np.ndarray,
    *,
    candidate_k: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a field descriptor at one point with the frozen mesh kernel."""

    f = np.asarray(features, dtype=np.float64)
    x = np.asarray(xyz, dtype=np.float64)
    c = np.asarray(covariance, dtype=np.float64)
    o = np.asarray(opacity, dtype=np.float64).reshape(-1)
    p = np.asarray(point, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 40 or x.shape != (len(f), 3):
        raise ValueError("DINO field features/xyz must align as [G,40]/[G,3]")
    if c.shape != (len(f), 3, 3) or o.shape != (len(f),) or p.shape != (3,):
        raise ValueError("point interpolation geometry changed")
    if not all(np.isfinite(value).all() for value in (f, x, c, o, p)):
        raise ValueError("point interpolation inputs must be finite")
    if bool((o < 0).any()) or candidate_k <= 0:
        raise ValueError("point interpolation opacity/K is invalid")
    _distance, indices = cKDTree(x).query(p[None], k=min(candidate_k, len(f)))
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    precision = np.linalg.pinv(c[indices] + 1e-6 * np.eye(3)[None])
    delta = x[indices] - p[None]
    mahalanobis = np.einsum("ki,kij,kj->k", delta, precision, delta)
    log_weights = np.full(len(indices), -np.inf, dtype=np.float64)
    positive = o[indices] > 0
    log_weights[positive] = -0.5 * np.maximum(mahalanobis[positive], 0) + np.log(o[indices][positive])
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        raise ValueError("point prompt has no finite Gaussian support")
    weights = np.exp(log_weights - maximum)
    descriptor = (weights[:, None] * f[indices]).sum(axis=0) / weights.sum()
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(descriptor).all() or norm <= 1e-12:
        raise ValueError("point prompt descriptor is degenerate")
    return (descriptor / norm).astype(np.float32), indices


def run_point3d(config: Point3DConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    output = config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    method, workspace = _validate_workspace(config)
    field_root = config.field_dir.resolve()
    field_path = field_root / "run_manifest.json"
    _validated_file(field_path, config.expected_field_manifest_sha256, "DINO field manifest")
    field = _load_json_object(field_path, "DINO field manifest")
    if (
        field.get("schema_version") != FIELD_SCHEMA
        or field.get("status") != FIELD_STATUS
        or field.get("benchmark_version") != BENCHMARK_VERSION
        or field.get("field_construction_eligible") is not True
        or field.get("scene_id") != method["query"]["scene_id"]
        or field.get("field_role") != "ludvig_dino_for_image_point_2d_point_3d"
    ):
        raise ValueError("DINO field authority/role changed")
    feature_path, feature_binding = _safe_bound_artifact(
        field_root, field.get("artifacts", {}).get("gaussian_features"), "DINO field features"
    )
    geometry_path, geometry_binding = _safe_bound_artifact(
        field_root, field.get("artifacts", {}).get("pruned_geometry"), "DINO field geometry"
    )
    features = np.load(feature_path, allow_pickle=False)
    if features.dtype != np.float32 or features.shape != (600000, 40) or not np.isfinite(features).all():
        raise ValueError("DINO field features changed")

    try:
        from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import _import_ludvig_geometry
        from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import gaussian_covariances
    except Exception as error:
        raise RuntimeError("LUDVIG Gaussian runtime unavailable") from error
    GaussianModel, _ = _import_ludvig_geometry(config.ludvig_upstream)
    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    xyz = gaussian.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    covariance = gaussian_covariances(
        gaussian.get_scaling, gaussian.get_rotation
    ).detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float32, copy=False)
    started = time.monotonic()
    descriptor, prompt_indices = point_descriptor(
        features, xyz, covariance, opacity,
        np.asarray(method["query"]["point_world_xyz"], dtype=np.float32),
        candidate_k=config.candidate_k,
    )
    probabilities, support = score_mesh_probabilities(
        gaussian_features=features,
        query_descriptor=descriptor,
        gaussian_xyz=xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=opacity,
        mesh_xyz=method["mesh"],
        calibration=config.calibration,
        candidate_k=config.candidate_k,
        chunk_size=config.chunk_size,
    )
    elapsed = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        query_id = str(method["query"]["query_id"])
        probability_path = temporary / f"{query_id}.npy"
        descriptor_path = temporary / "adapter" / "query_descriptor.npy"
        indices_path = temporary / "adapter" / "prompt_source_indices.npy"
        support_path = temporary / "adapter" / "mesh_support.npy"
        descriptor_path.parent.mkdir(parents=True)
        np.save(probability_path, probabilities, allow_pickle=False)
        np.save(descriptor_path, descriptor, allow_pickle=False)
        np.save(indices_path, prompt_indices, allow_pickle=False)
        np.save(support_path, support, allow_pickle=False)
        manifest = {
            "schema_version": "scannet_uqis_ludvig_point3d_adapter_v1",
            "status": "exact_runtime_smoke_complete",
            "benchmark_version": BENCHMARK_VERSION,
            "result_eligible": False,
            "formal_benchmark_row_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "benchmark_local_adapter": True,
            "modality": "point_3d",
            "scene_id": field["scene_id"],
            "query_id": query_id,
            "split_role": method["split_role"],
            "prediction_domain": PREDICTION_DOMAIN,
            "query_manifest": {"path": method["path"], "sha256": method["sha256"]},
            "workspace_receipt": workspace,
            "field": {
                "root": str(field_root),
                "manifest_sha256": config.expected_field_manifest_sha256,
                "features_sha256": feature_binding["sha256"],
                "geometry_sha256": geometry_binding["sha256"],
            },
            "privacy_boundary": {
                "captured_rgb_opened": False,
                "evaluator_manifest_opened": False,
                "private_target_inputs_opened": False,
            },
            "protocol": {
                "prompt_compiler": "K64_continuous_Gaussian_field_interpolation_at_world_point",
                "primitive_response": "cosine_similarity",
                "mesh_readout": "K64_continuous_opacity_weighted_gaussian",
                "probability_map": "frozen_global_monotonic_sigmoid",
                "test_time_fitting": False,
            },
            "elapsed_seconds": float(elapsed),
            "artifacts": {
                "query_descriptor": _array_binding(temporary, descriptor_path, descriptor),
                "prompt_source_indices": _array_binding(temporary, indices_path, prompt_indices),
                "mesh_support": _array_binding(temporary, support_path, support),
                "probability": _array_binding(temporary, probability_path, probabilities),
            },
            "argv": list(argv),
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_manifest.sha256").write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
