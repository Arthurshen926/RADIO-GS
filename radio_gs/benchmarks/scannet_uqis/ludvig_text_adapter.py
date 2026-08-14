"""LUDVIG/LERF-style text adapter for one isolated UQIS expression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256

from .ludvig_clip_field import SCHEMA_VERSION as FIELD_SCHEMA, STATUS as FIELD_STATUS
from .ludvig_dino_uplift import (
    SCHEMA_VERSION as DINO_FIELD_SCHEMA,
    STATUS as DINO_FIELD_STATUS,
)
from .ludvig_image_adapter import (
    IMAGE_MANIFEST_KEYS, IMAGE_SCENE_DOMAIN_KEYS, _array_binding,
    _load_json_object, _require_exact_keys, _require_sha256, _safe_bound_artifact,
    _validated_file,
)
from .ludvig_point3d_adapter import WORKSPACE_KEYS
from .ludvig_text_diffusion import (
    TextDiffusionConfig,
    align_clip_relevancy_to_dino_carrier,
    build_dino_graph,
    diffuse_clip_relevancy,
)
from .protocol import (
    BENCHMARK_VERSION,
    BENCHMARK_VERSION_V2_CANDIDATE,
    PREDICTION_DOMAIN,
    UQISProtocolConfig,
    sha256_file,
)


QUERY_KEYS = {"query_id", "scene_id", "modality", "expression", "available_method_inputs"}
AVAILABLE_INPUTS = ["scene_id", "expression"]
NEGATIVES = ("object", "things", "stuff", "texture")


@dataclass(frozen=True)
class TextConfig:
    query_manifest_path: Path
    workspace_receipt_path: Path
    field_dir: Path
    expected_field_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    dino_field_dir: Path | None = None
    expected_dino_field_manifest_sha256: str | None = None
    diffusion: TextDiffusionConfig = TextDiffusionConfig()
    device: str = "cuda:0"
    candidate_k: int = 64
    mesh_chunk_size: int = 32_768
    feature_chunk_size: int = 131_072


def lerf_relevancy(similarities: np.ndarray) -> np.ndarray:
    """Upstream hardest-negative softmax relevance from [N,1+4] cosines."""

    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1 + len(NEGATIVES) or not np.isfinite(values).all():
        raise ValueError("LERF similarities must be finite [N,5]")
    delta = 10.0 * (values[:, :1] - values[:, 1:])
    positive = np.empty_like(delta)
    mask = delta >= 0
    positive[mask] = 1.0 / (1.0 + np.exp(-delta[mask]))
    exponential = np.exp(delta[~mask])
    positive[~mask] = exponential / (1.0 + exponential)
    return positive.min(axis=1).astype(np.float32)


def _validate_workspace(config: TextConfig) -> tuple[dict[str, Any], dict[str, str]]:
    path = config.query_manifest_path.resolve()
    payload = _load_json_object(path, "text query manifest")
    _require_exact_keys(payload, IMAGE_MANIFEST_KEYS, "text method manifest")
    protocol = json.loads(json.dumps(asdict(UQISProtocolConfig())))
    if (
        payload.get("benchmark_version")
        not in {BENCHMARK_VERSION, BENCHMARK_VERSION_V2_CANDIDATE}
        or payload.get("protocol_config") != protocol
        or payload.get("protocol_config_sha256") != canonical_json_sha256(payload["protocol_config"])
        or payload.get("release_tier") != "pilot_harness"
        or payload.get("formal_benchmark_eligible") is not False
        or payload.get("visibility") != "method_input" or payload.get("modality") != "text"
        or payload.get("prediction_domain") != PREDICTION_DOMAIN
    ):
        raise ValueError("text method manifest authority changed")
    _require_sha256(payload.get("query_id_salt_sha256"), "query ID salt")
    domains, queries = payload.get("scene_domains"), payload.get("queries")
    if not isinstance(domains, list) or len(domains) != 1 or not isinstance(queries, list) or len(queries) != 1:
        raise ValueError("text workspace must contain one scene/query")
    domain, query = domains[0], queries[0]
    if not isinstance(domain, Mapping) or not isinstance(query, Mapping):
        raise ValueError("text workspace rows must be objects")
    _require_exact_keys(domain, IMAGE_SCENE_DOMAIN_KEYS, "public scene domain")
    _require_exact_keys(query, QUERY_KEYS, "text query")
    expression = str(query.get("expression", ""))
    if (
        query.get("modality") != "text" or query.get("available_method_inputs") != AVAILABLE_INPUTS
        or query.get("scene_id") != domain.get("scene_id") or not expression.strip()
    ):
        raise ValueError("text query contract changed")
    mesh_path = _validated_file(domain.get("mesh_xyz_path"), domain.get("mesh_xyz_sha256"), "public mesh")
    mesh = np.load(mesh_path, allow_pickle=False)
    if mesh.dtype != np.float32 or mesh.shape != (int(domain.get("mesh_vertices", -1)), 3):
        raise ValueError("public mesh changed")
    receipt_path = config.workspace_receipt_path.resolve()
    receipt = _load_json_object(receipt_path, "text workspace receipt")
    _require_exact_keys(receipt, WORKSPACE_KEYS, "text workspace receipt")
    digest = sha256_file(path)
    if (
        receipt.get("schema_version")
        not in {
            "scannet_uqis_query_workspace_v1",
            "scannet_uqis_v2_candidate_query_workspace_v1",
        }
        or receipt.get("status") != "staged" or receipt.get("workspace_query_manifest_sha256") != digest
        or receipt.get("query_id") != query.get("query_id") or receipt.get("scene_id") != query.get("scene_id")
        or receipt.get("modality") != "text" or receipt.get("query_count") != 1
        or receipt.get("benchmark_version") != payload.get("benchmark_version")
        or receipt.get("independent_workspace_required") is not True
        or receipt.get("mount_policy") != "workspace_only_read_only"
        or receipt.get("evaluator_private_files_staged") is not False
    ):
        raise ValueError("text workspace receipt does not bind this query")
    return ({"path": str(path), "sha256": digest, "query": dict(query), "domain": dict(domain),
             "mesh": mesh, "split_role": payload["split_role"],
             "benchmark_version": payload["benchmark_version"]},
            {"path": str(receipt_path), "sha256": sha256_file(receipt_path)})


def _mesh_readout(values: np.ndarray, xyz: np.ndarray, covariance: np.ndarray, opacity: np.ndarray,
                  mesh: np.ndarray, *, k: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    primitive = np.asarray(values, dtype=np.float64)
    x = np.asarray(xyz, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    op = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if primitive.shape != (len(x),) or cov.shape != (len(x), 3, 3) or op.shape != (len(x),):
        raise ValueError("text field/geometry rows changed")
    precision = np.linalg.pinv(cov + 1e-6 * np.eye(3)[None])
    tree = cKDTree(x)
    result = np.empty(len(mesh), dtype=np.float32)
    support = np.empty(len(mesh), dtype=np.float32)
    for begin in range(0, len(mesh), chunk):
        end = min(begin + chunk, len(mesh))
        _distance, indices = tree.query(mesh[begin:end], k=min(k, len(x)), workers=-1)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]
        delta = x[indices] - mesh[begin:end, None]
        mahalanobis = np.einsum("vki,vkij,vkj->vk", delta, precision[indices], delta, optimize=True)
        local_opacity = op[indices]
        log_weights = np.full_like(mahalanobis, -np.inf)
        positive = local_opacity > 0
        log_weights[positive] = -0.5 * np.maximum(mahalanobis[positive], 0) + np.log(local_opacity[positive])
        maximum = log_weights.max(axis=1)
        if not np.isfinite(maximum).all():
            raise ValueError("mesh vertex has no CLIP Gaussian support")
        weights = np.exp(log_weights - maximum[:, None])
        normalizer = weights.sum(axis=1)
        result[begin:end] = ((weights * primitive[indices]).sum(axis=1) / normalizer).astype(np.float32)
        log_support = maximum + np.log(normalizer)
        support[begin:end] = np.exp(np.clip(log_support, np.log(np.finfo(np.float32).tiny), np.log(np.finfo(np.float32).max))).astype(np.float32)
    return result, support


def run_text(config: TextConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    import torch
    output = config.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    method, workspace = _validate_workspace(config)
    field_root = config.field_dir.resolve()
    field_path = field_root / "run_manifest.json"
    _validated_file(field_path, config.expected_field_manifest_sha256, "CLIP field manifest")
    field = _load_json_object(field_path, "CLIP field manifest")
    if (
        field.get("schema_version") != FIELD_SCHEMA or field.get("status") != FIELD_STATUS
        or field.get("benchmark_version") != BENCHMARK_VERSION
        or field.get("field_construction_eligible") is not True
        or field.get("scene_id") != method["query"]["scene_id"]
        or field.get("field_role") != "ludvig_clip_for_text"
    ):
        raise ValueError("CLIP field authority/role changed")
    feature_path, feature_binding = _safe_bound_artifact(field_root, field["artifacts"]["gaussian_features"], "CLIP features")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    geometry = field["geometry"]
    geometry_path = _validated_file(geometry["path"], geometry["sha256"], "CLIP Gaussian geometry")
    if features.dtype != np.float32 or features.shape != (int(geometry["gaussians"]), 512):
        raise ValueError("CLIP feature shape changed")
    site_packages = Path(field["open_clip"]["python_source_root"]).parent
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))
    if str(config.ludvig_upstream.resolve()) not in sys.path:
        sys.path.insert(0, str(config.ludvig_upstream.resolve()))
    from clip_utils.openclip_encoder import OpenCLIPNetwork
    from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import _import_ludvig_geometry
    from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_d import gaussian_covariances
    cuda_device = torch.device(config.device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("LUDVIG text inference requires CUDA")
    torch.cuda.set_device(cuda_device)
    network = OpenCLIPNetwork(config.device)
    phrases = [str(method["query"]["expression"]), *NEGATIVES]
    with torch.no_grad():
        embeddings = network.encode_text(phrases, config.device).float()
        embeddings /= torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
    started = time.monotonic()
    primitive = np.empty(len(features), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(features), config.feature_chunk_size):
            end = min(begin + config.feature_chunk_size, len(features))
            # mmap slices are read-only; materialize a writable bounded chunk
            # before handing it to PyTorch even though inference never mutates it.
            local = torch.from_numpy(np.array(features[begin:end], copy=True)).to(cuda_device)
            similarities = (local @ embeddings.T).cpu().numpy()
            primitive[begin:end] = lerf_relevancy(similarities)
    del network, embeddings
    torch.cuda.empty_cache()
    GaussianModel, _ = _import_ludvig_geometry(config.ludvig_upstream)
    gaussian = GaussianModel(sh_degree=0)
    diffusion_manifest = None
    readout_primitive = primitive
    readout_geometry_path = geometry_path
    if (config.dino_field_dir is None) != (
        config.expected_dino_field_manifest_sha256 is None
    ):
        raise ValueError("DINO field directory and manifest hash must be supplied together")
    if config.dino_field_dir is not None:
        dino_root = config.dino_field_dir.resolve()
        dino_manifest_path = dino_root / "run_manifest.json"
        _validated_file(
            dino_manifest_path,
            str(config.expected_dino_field_manifest_sha256),
            "DINO field manifest",
        )
        dino = _load_json_object(dino_manifest_path, "DINO field manifest")
        if (
            dino.get("schema_version") != DINO_FIELD_SCHEMA
            or dino.get("status") != DINO_FIELD_STATUS
            or dino.get("benchmark_version") != BENCHMARK_VERSION
            or dino.get("field_construction_eligible") is not True
            or dino.get("scene_id") != field.get("scene_id")
            or dino.get("source_geometry", {}).get("sha256") != geometry.get("sha256")
        ):
            raise ValueError("DINO/CLIP field authority or source carrier changed")
        dino_feature_path, dino_feature_binding = _safe_bound_artifact(
            dino_root, dino["artifacts"]["gaussian_features"], "DINO features"
        )
        source_index_path, source_index_binding = _safe_bound_artifact(
            dino_root, dino["artifacts"]["source_indices"], "DINO source indices"
        )
        readout_geometry_path, dino_geometry_binding = _safe_bound_artifact(
            dino_root, dino["artifacts"]["pruned_geometry"], "DINO pruned geometry"
        )
        dino_features = np.load(dino_feature_path, mmap_mode="r", allow_pickle=False)
        source_indices = np.load(source_index_path, allow_pickle=False)
        if (
            dino_features.dtype != np.float32
            or dino_features.ndim != 2
            or dino_features.shape[1] != 40
            or source_indices.shape != (len(dino_features),)
        ):
            raise ValueError("DINO field artifacts changed")
        dino_gaussian = GaussianModel(sh_degree=0)
        dino_gaussian.load_ply(str(readout_geometry_path))
        dino_xyz = dino_gaussian.get_xyz.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        if len(dino_xyz) != len(dino_features):
            raise ValueError("DINO features and pruned geometry disagree")
        aligned = align_clip_relevancy_to_dino_carrier(primitive, source_indices)
        graph = build_dino_graph(dino_xyz, dino_features, config.diffusion)
        readout_primitive = diffuse_clip_relevancy(aligned, graph, config.diffusion)
        diffusion_manifest = {
            "schema_version": "scannet_uqis_ludvig_text_diffusion_v1",
            "algorithm": "benchmark_local_clip_seeded_dino_knn_diffusion",
            "config": asdict(config.diffusion),
            "dino_field": {
                "root": str(dino_root),
                "manifest_sha256": str(config.expected_dino_field_manifest_sha256),
                "features_sha256": dino_feature_binding["sha256"],
                "source_indices_sha256": source_index_binding["sha256"],
                "geometry_sha256": dino_geometry_binding["sha256"],
            },
            "captured_rgb_opened": False,
            "sam_opened": False,
            "evaluator_labels_opened": False,
            "test_fitting": False,
        }
        gaussian = dino_gaussian
    else:
        gaussian.load_ply(str(readout_geometry_path))
    xyz = gaussian.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    covariance = gaussian_covariances(gaussian.get_scaling, gaussian.get_rotation).detach().cpu().numpy().astype(np.float32, copy=False)
    opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float32, copy=False)
    probability, support = _mesh_readout(readout_primitive, xyz, covariance, opacity, method["mesh"],
                                         k=config.candidate_k, chunk=config.mesh_chunk_size)
    elapsed = time.monotonic() - started
    if not np.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("text adapter produced invalid probability")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        query_id = str(method["query"]["query_id"])
        probability_path = temporary / f"{query_id}.npy"
        primitive_path = temporary / "adapter/primitive_relevancy.npy"
        diffused_path = temporary / "adapter/diffused_relevancy.npy"
        support_path = temporary / "adapter/mesh_support.npy"
        primitive_path.parent.mkdir(parents=True)
        np.save(probability_path, probability, allow_pickle=False)
        np.save(primitive_path, primitive, allow_pickle=False)
        if diffusion_manifest is not None:
            np.save(diffused_path, readout_primitive, allow_pickle=False)
        np.save(support_path, support, allow_pickle=False)
        manifest = {
            "schema_version": "scannet_uqis_ludvig_text_adapter_v1", "status": "exact_runtime_smoke_complete",
            "benchmark_version": method["benchmark_version"], "result_eligible": False,
            "formal_benchmark_row_eligible": False, "official_ludvig_reproduction": False,
            "paper_metric_comparable": False, "benchmark_local_adapter": True,
            "modality": "text", "scene_id": field["scene_id"], "query_id": query_id,
            "split_role": method["split_role"], "prediction_domain": PREDICTION_DOMAIN,
            "query_manifest": {"path": method["path"], "sha256": method["sha256"]},
            "workspace_receipt": workspace,
            "field": {"root": str(field_root), "manifest_sha256": config.expected_field_manifest_sha256,
                      "features_sha256": feature_binding["sha256"], "geometry_sha256": geometry["sha256"]},
            "diffusion": diffusion_manifest,
            "privacy_boundary": {"captured_rgb_opened": False, "evaluator_manifest_opened": False,
                                 "private_target_inputs_opened": False},
            "protocol": {"text_encoder": "OpenCLIP_ViT-B-16_laion2b_s34b_b88k",
                         "negative_phrases": list(NEGATIVES), "relevancy": "upstream_LERF_hardest_negative_softmax_temperature_10",
                         "mesh_readout": "K64_continuous_opacity_weighted_gaussian", "test_time_fitting": False,
                         "graph_diffusion_enabled": diffusion_manifest is not None},
            "elapsed_seconds": float(elapsed),
            "artifacts": {"primitive_relevancy": _array_binding(temporary, primitive_path, primitive),
                          **({"diffused_relevancy": _array_binding(
                              temporary, diffused_path, readout_primitive
                          )} if diffusion_manifest is not None else {}),
                          "mesh_support": _array_binding(temporary, support_path, support),
                          "probability": _array_binding(temporary, probability_path, probability)},
            "argv": list(argv),
        }
        path = temporary / "run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_manifest.sha256").write_text(sha256_file(path) + "\n", encoding="ascii")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
