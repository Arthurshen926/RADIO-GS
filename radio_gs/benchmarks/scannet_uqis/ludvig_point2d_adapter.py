"""Strict RGB-free LUDVIG adapter for one registered 2-D point prompt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256

from .ludvig_dino_uplift import SCHEMA_VERSION as FIELD_SCHEMA, STATUS as FIELD_STATUS
from .ludvig_image_adapter import (
    DEFAULT_SIGMOID_CALIBRATION, IMAGE_MANIFEST_KEYS, IMAGE_SCENE_DOMAIN_KEYS,
    FrozenSigmoidCalibration, _array_binding, _load_json_object,
    _require_exact_keys, _require_sha256, _safe_bound_artifact, _validated_file,
    score_mesh_probabilities,
)
from .ludvig_point3d_adapter import WORKSPACE_KEYS
from .protocol import BENCHMARK_VERSION, PREDICTION_DOMAIN, UQISProtocolConfig, sha256_file


QUERY_KEYS = {
    "query_id", "scene_id", "modality", "camera_to_world", "camera_intrinsics",
    "raster_size", "positive_pixel_uv", "available_method_inputs",
}
AVAILABLE_INPUTS = [
    "scene_id", "camera_to_world", "camera_intrinsics", "raster_size",
    "positive_pixel_uv",
]


@dataclass(frozen=True)
class Point2DConfig:
    query_manifest_path: Path
    workspace_receipt_path: Path
    field_dir: Path
    expected_field_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    calibration: FrozenSigmoidCalibration = DEFAULT_SIGMOID_CALIBRATION
    candidate_k: int = 64
    chunk_size: int = 65_536


def projection_matrix_from_intrinsics(
    intrinsic: np.ndarray, width: int, height: int, *, znear: float = 0.01, zfar: float = 100.0
) -> np.ndarray:
    """Graphdeco projection with the protocol's non-centered principal point."""

    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape != (3, 3) or width <= 0 or height <= 0 or not np.isfinite(k).all():
        raise ValueError("camera intrinsics/raster are invalid")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not (0 < znear < zfar):
        raise ValueError("camera focal/depth range is invalid")
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 2.0 * k[0, 0] / width
    p[1, 1] = 2.0 * k[1, 1] / height
    # CUDA ndc2Pix(v,S)=((v+1)S-1)/2, hence the half-pixel term.
    p[0, 2] = 2.0 * (k[0, 2] + 0.5) / width - 1.0
    p[1, 2] = 2.0 * (k[1, 2] + 0.5) / height - 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    p[3, 2] = 1.0
    return p


def _validate_workspace(config: Point2DConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    path = config.query_manifest_path.resolve()
    payload = _load_json_object(path, "point_2d query manifest")
    _require_exact_keys(payload, IMAGE_MANIFEST_KEYS, "point_2d method manifest")
    expected_protocol = json.loads(json.dumps(asdict(UQISProtocolConfig())))
    if (
        payload.get("benchmark_version") != BENCHMARK_VERSION
        or payload.get("protocol_config") != expected_protocol
        or payload.get("protocol_config_sha256") != canonical_json_sha256(payload["protocol_config"])
        or payload.get("release_tier") != "pilot_harness"
        or payload.get("formal_benchmark_eligible") is not False
        or payload.get("visibility") != "method_input"
        or payload.get("modality") != "point_2d"
        or payload.get("prediction_domain") != PREDICTION_DOMAIN
    ):
        raise ValueError("point_2d method manifest authority changed")
    _require_sha256(payload.get("query_id_salt_sha256"), "query ID salt")
    domains, queries = payload.get("scene_domains"), payload.get("queries")
    if not isinstance(domains, list) or len(domains) != 1 or not isinstance(queries, list) or len(queries) != 1:
        raise ValueError("point_2d workspace must contain one scene/query")
    domain, query = domains[0], queries[0]
    if not isinstance(domain, Mapping) or not isinstance(query, Mapping):
        raise ValueError("point_2d workspace rows must be objects")
    _require_exact_keys(domain, IMAGE_SCENE_DOMAIN_KEYS, "public scene domain")
    _require_exact_keys(query, QUERY_KEYS, "point_2d query")
    if (
        query.get("modality") != "point_2d"
        or query.get("available_method_inputs") != AVAILABLE_INPUTS
        or query.get("scene_id") != domain.get("scene_id")
    ):
        raise ValueError("point_2d query contract changed")
    c2w = np.asarray(query.get("camera_to_world"), dtype=np.float64)
    intrinsic = np.asarray(query.get("camera_intrinsics"), dtype=np.float64)
    raster = np.asarray(query.get("raster_size"), dtype=np.int64)
    uv = np.asarray(query.get("positive_pixel_uv"), dtype=np.int64)
    if (
        c2w.shape != (4, 4) or intrinsic.shape != (3, 3) or raster.shape != (2,)
        or uv.shape != (2,) or not np.isfinite(c2w).all() or not np.isfinite(intrinsic).all()
        or bool((raster <= 0).any()) or bool((uv < 0).any())
        or uv[0] >= raster[0] or uv[1] >= raster[1]
    ):
        raise ValueError("point_2d camera/click is invalid")
    projection_matrix_from_intrinsics(intrinsic, int(raster[0]), int(raster[1]))
    mesh_path = _validated_file(domain.get("mesh_xyz_path"), domain.get("mesh_xyz_sha256"), "public mesh")
    mesh = np.load(mesh_path, allow_pickle=False)
    if mesh.dtype != np.float32 or mesh.shape != (int(domain.get("mesh_vertices", -1)), 3):
        raise ValueError("public mesh domain changed")
    receipt_path = config.workspace_receipt_path.resolve()
    receipt = _load_json_object(receipt_path, "point_2d workspace receipt")
    _require_exact_keys(receipt, WORKSPACE_KEYS, "point_2d workspace receipt")
    manifest_hash = sha256_file(path)
    if (
        receipt.get("schema_version") != "scannet_uqis_query_workspace_v1"
        or receipt.get("status") != "staged"
        or receipt.get("workspace_query_manifest_sha256") != manifest_hash
        or receipt.get("query_id") != query.get("query_id")
        or receipt.get("scene_id") != query.get("scene_id")
        or receipt.get("modality") != "point_2d" or receipt.get("query_count") != 1
        or receipt.get("independent_workspace_required") is not True
        or receipt.get("mount_policy") != "workspace_only_read_only"
        or receipt.get("evaluator_private_files_staged") is not False
    ):
        raise ValueError("point_2d workspace receipt does not bind this query")
    return ({
        "path": str(path), "sha256": manifest_hash, "split_role": payload["split_role"],
        "query": dict(query), "domain": dict(domain), "mesh": mesh,
        "c2w": c2w, "intrinsic": intrinsic, "raster": raster, "uv": uv,
    }, {"path": str(receipt_path), "sha256": sha256_file(receipt_path)})


def _render_click_descriptor(gaussian: Any, features: np.ndarray, method: Mapping[str, Any], upstream: Path) -> np.ndarray:
    import torch
    root = upstream.resolve()
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from gaussiansplatting.gaussian_renderer import render

    width, height = map(int, method["raster"])
    u, v = map(int, method["uv"])
    world_view = torch.from_numpy(np.linalg.inv(method["c2w"]).astype(np.float32)).transpose(0, 1).cuda()
    projection = torch.from_numpy(
        projection_matrix_from_intrinsics(method["intrinsic"], width, height)
    ).transpose(0, 1).cuda()
    camera = SimpleNamespace(
        image_width=width, image_height=height,
        FoVx=2.0 * math.atan(width / (2.0 * float(method["intrinsic"][0, 0]))),
        FoVy=2.0 * math.atan(height / (2.0 * float(method["intrinsic"][1, 1]))),
        world_view_transform=world_view,
        projection_matrix=projection,
        full_proj_transform=world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        camera_center=world_view.inverse()[3, :3],
    )
    values = torch.from_numpy(features).cuda()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False)
    descriptor = torch.zeros(40, dtype=torch.float32, device="cuda")
    counts = torch.zeros(40, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for j in range(0, 40, 3):
            begin = min(j, 37)
            pixel = render(
                camera, gaussian, pipe, background, override_color=values[:, begin:begin + 3]
            )["render"][:, v, u]
            length = min(3, 40 - begin)
            descriptor[begin:begin + length] += pixel[:length]
            counts[begin:begin + length] += 1
        descriptor /= counts
        norm = torch.linalg.vector_norm(descriptor)
        if not bool(torch.isfinite(descriptor).all()) or float(norm) <= 1e-12:
            raise ValueError("rendered click has no finite DINO field descriptor")
        descriptor /= norm
    return descriptor.cpu().numpy().astype(np.float32, copy=False)


def run_point2d(config: Point2DConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    output = config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    method, workspace = _validate_workspace(config)
    field_root = config.field_dir.resolve()
    field_path = field_root / "run_manifest.json"
    _validated_file(field_path, config.expected_field_manifest_sha256, "DINO field manifest")
    field = _load_json_object(field_path, "DINO field manifest")
    if (
        field.get("schema_version") != FIELD_SCHEMA or field.get("status") != FIELD_STATUS
        or field.get("benchmark_version") != BENCHMARK_VERSION
        or field.get("field_construction_eligible") is not True
        or field.get("scene_id") != method["query"]["scene_id"]
        or field.get("field_role") != "ludvig_dino_for_image_point_2d_point_3d"
    ):
        raise ValueError("DINO field authority/role changed")
    feature_path, feature_binding = _safe_bound_artifact(field_root, field["artifacts"]["gaussian_features"], "DINO features")
    geometry_path, geometry_binding = _safe_bound_artifact(field_root, field["artifacts"]["pruned_geometry"], "DINO geometry")
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
    started = time.monotonic()
    descriptor = _render_click_descriptor(gaussian, features, method, config.ludvig_upstream)
    xyz = gaussian.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    covariance = gaussian_covariances(gaussian.get_scaling, gaussian.get_rotation).detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float32, copy=False)
    probabilities, support = score_mesh_probabilities(
        gaussian_features=features, query_descriptor=descriptor, gaussian_xyz=xyz,
        gaussian_covariance=covariance, gaussian_opacity=opacity, mesh_xyz=method["mesh"],
        calibration=config.calibration, candidate_k=config.candidate_k, chunk_size=config.chunk_size,
    )
    elapsed = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        query_id = str(method["query"]["query_id"])
        probability_path = temporary / f"{query_id}.npy"
        descriptor_path = temporary / "adapter/query_descriptor.npy"
        support_path = temporary / "adapter/mesh_support.npy"
        descriptor_path.parent.mkdir(parents=True)
        np.save(probability_path, probabilities, allow_pickle=False)
        np.save(descriptor_path, descriptor, allow_pickle=False)
        np.save(support_path, support, allow_pickle=False)
        manifest = {
            "schema_version": "scannet_uqis_ludvig_point2d_adapter_v1",
            "status": "exact_runtime_smoke_complete", "benchmark_version": BENCHMARK_VERSION,
            "result_eligible": False, "formal_benchmark_row_eligible": False,
            "official_ludvig_reproduction": False, "paper_metric_comparable": False,
            "benchmark_local_adapter": True, "modality": "point_2d",
            "scene_id": field["scene_id"], "query_id": query_id,
            "split_role": method["split_role"], "prediction_domain": PREDICTION_DOMAIN,
            "query_manifest": {"path": method["path"], "sha256": method["sha256"]},
            "workspace_receipt": workspace,
            "field": {"root": str(field_root), "manifest_sha256": config.expected_field_manifest_sha256,
                      "features_sha256": feature_binding["sha256"], "geometry_sha256": geometry_binding["sha256"]},
            "privacy_boundary": {"captured_rgb_opened": False, "captured_depth_opened": False,
                                 "evaluator_manifest_opened": False, "private_target_inputs_opened": False},
            "protocol": {"prompt_compiler": "render_frozen_DINO40_field_then_read_positive_pixel",
                         "render_channels_per_pass": 3, "render_passes": 14,
                         "principal_point": "exact_off_center_projection_half_pixel_aware",
                         "primitive_response": "cosine_similarity",
                         "mesh_readout": "K64_continuous_opacity_weighted_gaussian",
                         "probability_map": "frozen_global_monotonic_sigmoid", "test_time_fitting": False},
            "elapsed_seconds": float(elapsed),
            "artifacts": {"query_descriptor": _array_binding(temporary, descriptor_path, descriptor),
                          "mesh_support": _array_binding(temporary, support_path, support),
                          "probability": _array_binding(temporary, probability_path, probabilities)},
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
