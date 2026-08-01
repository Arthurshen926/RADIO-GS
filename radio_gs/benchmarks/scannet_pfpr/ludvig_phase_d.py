"""Public-query scoring adapter from exact LUDVIG features to ScanNet-PFPR.

LUDVIG does not publish a PFPR task head.  This phase therefore keeps its
released DINO/PCA/uplift features exact and adds one explicit, learning-free
adapter: center-3x3 query pooling, primitive cosine, and continuous Gaussian
readout on the benchmark's public candidate domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import (
    FROZEN_METHOD_MANIFEST_SHA256,
    FROZEN_PUBLIC_MANIFEST_SHA256,
    sha256_file,
)
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
    PhaseCConfig,
    _import_ludvig_geometry,
    _validate_file,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import (
    PFPR_V2_BENCHMARK_VERSION,
    canonical_json_sha256,
    protocol_config_from_record,
)
from radio_gs.querying.query_compilers import continuous_gaussian_readout


PHASE_D_SCHEMA_VERSION = "ludvig_pfpr_public_crop_score_v1"
PHASE_D_STATUS = "phase_d_public_scores_complete_evaluation_not_run"
QUERY_POOLING = "center_3x3_mean_then_l2"
CANDIDATE_K = 64
READOUT_SUPPORT_THRESHOLD = 0.0


class LudvigPFPRPhaseDError(RuntimeError):
    """Raised before publishing scores when a public-method lock is invalid."""


@dataclass(frozen=True)
class PhaseDConfig:
    phase_c_dir: Path
    expected_phase_c_manifest_sha256: str
    benchmark_dir: Path
    ludvig_upstream: Path
    output_dir: Path
    driver_library_dir: Path = Path("/root/baselines/LUDVIG/.driver535")
    device: str = "cuda:0"
    scene_id: str = "scene0050_02"
    candidate_k: int = CANDIDATE_K
    readout_support_threshold: float = READOUT_SUPPORT_THRESHOLD


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LudvigPFPRPhaseDError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LudvigPFPRPhaseDError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise LudvigPFPRPhaseDError(f"{label} must be a JSON object")
    return value


def center_3x3_descriptor(tokens: np.ndarray | torch.Tensor) -> np.ndarray:
    """Pool the center 3x3 tokens and L2-normalize one query descriptor."""

    values = torch.as_tensor(tokens).float()
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or min(values.shape[:2]) < 3 or values.shape[-1] <= 0:
        raise LudvigPFPRPhaseDError("Query token grid must be [H,W,D]")
    center_y, center_x = values.shape[0] // 2, values.shape[1] // 2
    descriptor = values[
        center_y - 1 : center_y + 2, center_x - 1 : center_x + 2
    ].mean(dim=(0, 1))
    descriptor = F.normalize(descriptor, dim=0, eps=1e-8)
    if not torch.isfinite(descriptor).all():
        raise LudvigPFPRPhaseDError("Query descriptor is non-finite")
    return descriptor.cpu().numpy().astype(np.float32)


def gaussian_covariances(
    scaling: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """Construct world covariance from activated GraphDECO scale/quaternion."""

    scale = torch.as_tensor(scaling).float().clamp_min(1e-6)
    quaternion = F.normalize(torch.as_tensor(rotation).float(), dim=-1, eps=1e-8)
    if scale.ndim != 2 or scale.shape[1] != 3 or quaternion.shape != (len(scale), 4):
        raise LudvigPFPRPhaseDError("Gaussian scale/rotation shapes are invalid")
    r, x, y, z = quaternion.unbind(dim=-1)
    entries = (
        1 - 2 * (y.square() + z.square()),
        2 * (x * y - r * z),
        2 * (x * z + r * y),
        2 * (x * y + r * z),
        1 - 2 * (x.square() + z.square()),
        2 * (y * z - r * x),
        2 * (x * z - r * y),
        2 * (y * z + r * x),
        1 - 2 * (x.square() + y.square()),
    )
    matrix = torch.stack(entries, dim=-1).reshape(-1, 3, 3)
    return matrix @ torch.diag_embed(scale.square()) @ matrix.transpose(1, 2)


def _audit_phase_c(config: PhaseDConfig) -> tuple[Path, dict[str, Any], np.ndarray]:
    root = config.phase_c_dir.resolve()
    manifest_path = root / "run_manifest.json"
    _validate_file(
        manifest_path,
        config.expected_phase_c_manifest_sha256,
        "Phase-C run manifest",
    )
    manifest = _load_json(manifest_path, "Phase-C run manifest")
    if manifest.get("schema_version") != PHASE_C_SCHEMA_VERSION:
        raise LudvigPFPRPhaseDError("Phase-C schema changed")
    if manifest.get("status") != PHASE_C_STATUS or manifest.get("result_eligible") is not False:
        raise LudvigPFPRPhaseDError("Phase-C status changed")
    if str(manifest.get("scene_id")) != str(config.scene_id):
        raise LudvigPFPRPhaseDError("Phase-C scene differs from requested scene")
    binding = manifest.get("artifacts", {}).get("gaussian_features")
    if not isinstance(binding, Mapping):
        raise LudvigPFPRPhaseDError("Phase C lacks Gaussian features")
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise LudvigPFPRPhaseDError("Phase-C feature path escapes its attempt")
    feature_path = root / relative
    _validate_file(feature_path, str(binding.get("sha256", "")), "Phase-C features")
    features = np.load(feature_path, allow_pickle=False)
    if tuple(features.shape) != (300000, 40) or features.dtype != np.float32:
        raise LudvigPFPRPhaseDError("Phase-C feature shape/dtype changed")
    if not np.isfinite(features).all():
        raise LudvigPFPRPhaseDError("Phase-C features are non-finite")
    return root, manifest, features


def _audit_public_inputs(config: PhaseDConfig) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    root = config.benchmark_dir.resolve()
    method_path = root / "manifest.method.json"
    public_path = root / "manifest.public.json"
    method_binding = _validate_file(
        method_path, FROZEN_METHOD_MANIFEST_SHA256, "PFPR method manifest"
    )
    public_binding = _validate_file(
        public_path, FROZEN_PUBLIC_MANIFEST_SHA256, "PFPR public manifest"
    )
    method = _load_json(method_path, "PFPR method manifest")
    public = _load_json(public_path, "PFPR public manifest")
    if method.get("benchmark_version") != PFPR_V2_BENCHMARK_VERSION:
        raise LudvigPFPRPhaseDError("PFPR method benchmark version changed")
    if public.get("benchmark_version") != PFPR_V2_BENCHMARK_VERSION:
        raise LudvigPFPRPhaseDError("PFPR public benchmark version changed")
    protocol = protocol_config_from_record(
        PFPR_V2_BENCHMARK_VERSION, method.get("protocol_config", {})
    )
    if public.get("protocol_config") != method.get("protocol_config"):
        raise LudvigPFPRPhaseDError("PFPR public/method protocol configs differ")
    domains = [
        item for item in method.get("scene_domains", [])
        if str(item.get("scene_id", "")) == str(config.scene_id)
    ]
    if len(domains) != 1:
        raise LudvigPFPRPhaseDError("PFPR scene domain is missing or duplicated")
    domain = domains[0]
    candidate_path = Path(str(domain.get("candidate_xyz_path", ""))).resolve()
    candidate_binding = _validate_file(
        candidate_path, str(domain.get("candidate_xyz_sha256", "")), "PFPR candidates"
    )
    candidates = np.load(candidate_path, allow_pickle=False)
    if candidates.shape != (int(domain.get("candidate_points", -1)), 3):
        raise LudvigPFPRPhaseDError("PFPR candidate shape changed")
    if candidates.dtype != np.float32 or not np.isfinite(candidates).all():
        raise LudvigPFPRPhaseDError("PFPR candidates are invalid")
    records = [
        dict(item) for item in method.get("queries", [])
        if str(item.get("scene_id", "")) == str(config.scene_id)
    ]
    if len(records) != int(protocol.anchors_per_scene):
        raise LudvigPFPRPhaseDError("PFPR scene query count changed")
    seen: set[str] = set()
    for record in records:
        query_id = str(record.get("query_id", ""))
        crop = Path(str(record.get("crop_rgb_path", ""))).resolve()
        if not query_id or query_id in seen:
            raise LudvigPFPRPhaseDError("PFPR query IDs are invalid")
        seen.add(query_id)
        if set(record.get("available_method_inputs", ())) != {"scene_id", "crop_rgb"}:
            raise LudvigPFPRPhaseDError("PFPR method inputs changed")
        _validate_file(crop, str(record.get("crop_rgb_sha256", "")), f"query {query_id}")
    return records, np.asarray(candidates, dtype=np.float32), {
        "method_manifest": method_binding,
        "public_manifest": public_binding,
        "candidate_domain": candidate_binding,
        "candidate_voxel_size_m": float(protocol.candidate_voxel_size_m),
    }


def _save_array(root: Path, relative: str, value: np.ndarray) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    np.save(path, array, allow_pickle=False)
    return {
        "relative_path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def run_phase_d(config: PhaseDConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    """Encode public crops and atomically publish finite candidate score vectors."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigPFPRPhaseDError(f"Refusing to overwrite Phase-D output: {output}")
    if int(config.candidate_k) != CANDIDATE_K:
        raise LudvigPFPRPhaseDError(f"Phase D is frozen to candidate_k={CANDIDATE_K}")
    if float(config.readout_support_threshold) != READOUT_SUPPORT_THRESHOLD:
        raise LudvigPFPRPhaseDError("Phase-D support threshold changed")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise LudvigPFPRPhaseDError("Exact LUDVIG query extraction requires CUDA")
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
        raise LudvigPFPRPhaseDError("CUDA is unavailable")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    phase_c_root, phase_c, feature_array = _audit_phase_c(config)
    records, candidates, benchmark = _audit_public_inputs(config)
    phase_b_root = Path(str(phase_c["phase_b"]["root"])).resolve()
    phase_b_hash = str(phase_c["phase_b"]["manifest_sha256"])
    transform, phase_b = load_phase_b_transform(
        phase_b_root, expected_manifest_sha256=phase_b_hash
    )
    checkpoint_record = phase_b.get("phase_a", {}).get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint_record.get("path", ""))).resolve()
    _validate_file(
        checkpoint_path,
        str(checkpoint_record.get("sha256", "")),
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
    query_plan = ludvig_sliding_plan(128, 128)
    if (
        query_plan["patch_count"] != 1
        or query_plan["token_grid_height"] != 9
        or query_plan["token_grid_width"] != 9
    ):
        raise LudvigPFPRPhaseDError("PFPR query token geometry changed")
    descriptors: list[np.ndarray] = []
    query_records: list[dict[str, Any]] = []
    for record in records:
        raw = extract_view_tokens(
            Path(str(record["crop_rgb_path"])),
            predictor,
            query_plan,
            expected_embedding_dim=1536,
        )
        weighted = apply_scene_pca_transform(raw, transform, eigval_weighting=True)
        descriptor = center_3x3_descriptor(weighted)
        descriptors.append(descriptor)
        query_records.append(
            {
                "query_id": str(record["query_id"]),
                "scene_id": str(record["scene_id"]),
                "crop_rgb_sha256": str(record["crop_rgb_sha256"]),
                "raw_token_shape": list(raw.shape),
                "weighted_token_shape": list(weighted.shape),
                "descriptor_l2_norm": float(np.linalg.norm(descriptor)),
            }
        )
    descriptor_array = np.stack(descriptors).astype(np.float32, copy=False)
    del predictor, model
    gc.collect()
    torch.cuda.empty_cache()

    GaussianModel, _CamScene = _import_ludvig_geometry(config.ludvig_upstream)
    geometry_path = Path(str(phase_c["geometry"]["path"])).resolve()
    _validate_file(
        geometry_path, str(phase_c["geometry"]["sha256"]), "Gaussian geometry"
    )
    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    gaussian_xyz = gaussian.get_xyz.detach().float()
    covariance = gaussian_covariances(gaussian.get_scaling, gaussian.get_rotation)
    opacity = gaussian.get_opacity.detach().float().reshape(-1)
    field = F.normalize(torch.from_numpy(feature_array).to(device), dim=-1, eps=1e-8)
    queries = torch.from_numpy(descriptor_array).to(device)
    indices_np = cKDTree(gaussian_xyz.cpu().numpy()).query(
        candidates, k=int(config.candidate_k)
    )[1]
    indices_np = np.asarray(indices_np, dtype=np.int64)
    if indices_np.ndim == 1:
        indices_np = indices_np[:, None]
    indices = torch.from_numpy(np.ascontiguousarray(indices_np)).to(device)
    points = torch.from_numpy(candidates).to(device)
    voxel_size = float(benchmark["candidate_voxel_size_m"])
    variance = voxel_size ** 2 / 12.0
    precision = torch.linalg.pinv(
        covariance + variance * torch.eye(3, device=device, dtype=torch.float32)
    )
    _unused, support = continuous_gaussian_readout(
        gaussian_xyz,
        covariance,
        torch.ones(len(gaussian_xyz), device=device),
        points,
        gaussian_precision=precision,
        opacity=opacity,
        candidate_indices=indices,
    )
    # The existing PFPR direct-DINO baseline does not impose a tuned support
    # threshold.  Require only strictly positive kernel mass and report its
    # full distribution; this keeps the rule target-blind and fail-closed for
    # genuinely unsupported mesh points.
    valid = support > float(config.readout_support_threshold)
    if not bool(valid.all()):
        raise LudvigPFPRPhaseDError(
            "Public candidate domain has unsupported points; refusing non-finite padding"
        )

    score_arrays: list[np.ndarray] = []
    for descriptor in queries:
        primitive_scores = field @ descriptor
        scores, query_support = continuous_gaussian_readout(
            gaussian_xyz,
            covariance,
            primitive_scores,
            points,
            gaussian_precision=precision,
            opacity=opacity,
            candidate_indices=indices,
        )
        if not torch.equal(query_support, support):
            raise LudvigPFPRPhaseDError("Query-independent readout support changed")
        score = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        if score.shape != (len(candidates),) or not np.isfinite(score).all():
            raise LudvigPFPRPhaseDError("PFPR score vector is invalid")
        score_arrays.append(score)
    elapsed = time.monotonic() - started
    peak = {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    support_cpu = support.detach().cpu().numpy().astype(np.float32, copy=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.phase_d_tmp_", dir=output.parent))
    try:
        artifacts = {
            "query_descriptors": _save_array(
                temporary, "adapter/query_descriptors.npy", descriptor_array
            ),
            "candidate_indices": _save_array(
                temporary, "adapter/candidate_indices_k64.npy", indices_np
            ),
            "candidate_support": _save_array(
                temporary, "adapter/candidate_support.npy", support_cpu
            ),
        }
        prediction_records: list[dict[str, Any]] = []
        for record, scores in zip(query_records, score_arrays):
            binding = _save_array(
                temporary, f"predictions/{record['query_id']}.npy", scores
            )
            prediction_records.append({**record, "scores": binding})
        manifest: dict[str, Any] = {
            "schema_version": PHASE_D_SCHEMA_VERSION,
            "status": PHASE_D_STATUS,
            "result_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "scene_id": str(config.scene_id),
            "attempt_dir": str(output),
            "argv": list(argv),
            "device": str(device),
            "cuda_driver_binding": driver,
            "cuda_peak_memory": peak,
            "phase_c": {
                "root": str(phase_c_root),
                "manifest_sha256": str(config.expected_phase_c_manifest_sha256),
            },
            "phase_b": {"root": str(phase_b_root), "manifest_sha256": phase_b_hash},
            "ludvig_vendored_dinov2_source": source,
            "model": architecture,
            "checkpoint_exact_ludvig_load": checkpoint_load,
            "benchmark_public_inputs": benchmark,
            "protocol": {
                "available_method_inputs": ["scene_id", "crop_rgb"],
                "evaluator_private_manifest_opened": False,
                "query_alignment": "bilinear_128x128_to_patch_aligned_126x126",
                "query_tokens": "one_9x9_DINOv2_grid",
                "query_scene_transform": "reuse_frozen_Phase_B_standardization_PCA40_and_singular_values",
                "query_pooling": QUERY_POOLING,
                "primitive_score": "l2_normalized_gaussian_feature_dot_l2_normalized_query",
                "candidate_readout": "continuous_opacity_weighted_gaussian_convolved_with_5cm_voxel_cell",
                "candidate_k": int(config.candidate_k),
                "support_threshold": float(config.readout_support_threshold),
                "support_rule": "strictly_positive_no_tuned_support_gate",
                "learning_or_calibration": "none",
                "adapter_status": "custom_PFPR_adapter_because_LUDVIG_has_no_published_PFPR_head",
            },
            "support": {
                "valid_candidates": int(valid.sum()),
                "total_candidates": int(len(valid)),
                "valid_fraction": float(valid.float().mean()),
                "minimum": float(support.min()),
                "maximum": float(support.max()),
            },
            "elapsed_seconds": float(elapsed),
            "queries": prediction_records,
            "queries_sha256": canonical_json_sha256(prediction_records),
            "artifacts": artifacts,
            "phase_status": {
                "phase_a_cpu_staging": "bound_complete",
                "phase_b_dino_scene_features_and_pca": "bound_complete",
                "phase_c_inverse_render_uplift": "bound_complete",
                "phase_d_pfpr_crop_scoring": "complete",
                "phase_e_pfpr_evaluation": "not_run",
            },
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_sha256 = sha256_file(manifest_path)
        (temporary / "run_manifest.sha256").write_text(manifest_sha256 + "\n", encoding="ascii")
        if output.exists():
            raise LudvigPFPRPhaseDError(f"Refusing concurrent overwrite: {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": manifest_sha256}
