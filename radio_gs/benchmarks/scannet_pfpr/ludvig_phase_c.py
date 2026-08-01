"""Hash-bound LUDVIG inverse-render uplifting for the PFPR adapter.

Phase C consumes only the frozen Phase-B per-view PCA features and the
Phase-A camera/geometry staging.  It deliberately does not open method query
crops or the evaluator-private manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    EXPECTED_DRIVER_LIBCUDA_SHA256,
    EXPECTED_NVIDIA_DRIVER_VERSION,
    PHASE_B_SCHEMA_VERSION,
    PHASE_B_STATUS,
    _require_sha256,
    audit_cuda_driver_binding,
    PhaseBConfig,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256


PHASE_C_SCHEMA_VERSION = "ludvig_pfpr_inverse_render_uplift_v1"
PHASE_C_STATUS = "phase_c_uplift_complete_phase_d_not_run"


class LudvigPFPRPhaseCError(RuntimeError):
    """Raised before publishing a Phase-C artifact when a lock is invalid."""


@dataclass(frozen=True)
class PhaseCConfig:
    phase_b_dir: Path
    expected_phase_b_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    driver_library_dir: Path = Path("/root/baselines/LUDVIG/.driver535")
    device: str = "cuda:0"
    expected_driver_version: str = EXPECTED_NVIDIA_DRIVER_VERSION
    expected_driver_libcuda_sha256: str = EXPECTED_DRIVER_LIBCUDA_SHA256


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LudvigPFPRPhaseCError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LudvigPFPRPhaseCError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise LudvigPFPRPhaseCError(f"{label} must be a JSON object")
    return value


def _validate_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, f"expected {label} hash")
    if not path.is_file():
        raise LudvigPFPRPhaseCError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise LudvigPFPRPhaseCError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def audit_phase_b_attempt(config: PhaseCConfig) -> tuple[Path, dict[str, Any]]:
    root = config.phase_b_dir.resolve()
    manifest_path = root / "run_manifest.json"
    _validate_file(
        manifest_path,
        config.expected_phase_b_manifest_sha256,
        "Phase-B run manifest",
    )
    manifest = _load_json(manifest_path, "Phase-B run manifest")
    if manifest.get("schema_version") != PHASE_B_SCHEMA_VERSION:
        raise LudvigPFPRPhaseCError("Phase-B schema changed")
    if manifest.get("status") != PHASE_B_STATUS:
        raise LudvigPFPRPhaseCError("Phase-B status changed")
    if manifest.get("result_eligible") is not False:
        raise LudvigPFPRPhaseCError("Phase-B must remain result-ineligible")
    views = manifest.get("views")
    if not isinstance(views, list) or len(views) != 120:
        raise LudvigPFPRPhaseCError("Production Phase C requires all 120 Phase-B views")
    if manifest.get("views_sha256") != canonical_json_sha256(views):
        raise LudvigPFPRPhaseCError("Phase-B ordered view binding changed")
    return root, manifest


def reconstruct_ludvig_feature_map(
    tokens: torch.Tensor,
    sliding_window: Mapping[str, Any],
    *,
    output_height: int = 480,
    output_width: int = 640,
) -> torch.Tensor:
    """Reproduce ``SlidingWindow.fill`` followed by LUDVIG's final resize."""

    values = torch.as_tensor(tokens).float()
    indices = sliding_window.get("indices_yx", [])
    crop = int(sliding_window.get("effective_crop_size", 0))
    aligned_h = int(sliding_window.get("aligned_height", 0))
    aligned_w = int(sliding_window.get("aligned_width", 0))
    if values.ndim != 4 or values.shape[0] != len(indices):
        raise LudvigPFPRPhaseCError("Phase-B token/patch geometry is invalid")
    if crop <= 0 or aligned_h <= 0 or aligned_w <= 0:
        raise LudvigPFPRPhaseCError("Phase-B sliding-window dimensions are invalid")
    channels = int(values.shape[-1])
    result = torch.zeros(
        (channels, aligned_h, aligned_w), device=values.device, dtype=torch.float32
    )
    counts = torch.zeros((aligned_h, aligned_w), device=values.device, dtype=torch.float32)
    for index, (y, x) in enumerate(indices):
        patch = values[index].permute(2, 0, 1).contiguous()
        patch = F.interpolate(
            patch[None], size=(crop, crop), mode="bilinear", align_corners=False
        ).squeeze(0)
        y, x = int(y), int(x)
        result[:, y : y + crop, x : x + crop] += patch / float(crop)
        counts[y : y + crop, x : x + crop] += 1.0 / float(crop)
    if bool((counts <= 0).any()):
        raise LudvigPFPRPhaseCError("Sliding-window reconstruction leaves uncovered pixels")
    filled = result / counts[None]
    return F.interpolate(
        filled[None],
        size=(int(output_height), int(output_width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _binding_from_array(root: Path, relative: str, value: np.ndarray) -> dict[str, Any]:
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


def _load_weighted_tokens(
    phase_b_root: Path, record: Mapping[str, Any]
) -> np.ndarray:
    binding = record.get("eigval_weighted_tokens")
    if not isinstance(binding, Mapping):
        raise LudvigPFPRPhaseCError("Phase-B view lacks weighted tokens")
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise LudvigPFPRPhaseCError("Phase-B token path escapes its attempt")
    path = phase_b_root / relative
    _validate_file(path, str(binding.get("sha256", "")), "Phase-B weighted tokens")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != binding.get("shape") or str(array.dtype) != binding.get("dtype"):
        raise LudvigPFPRPhaseCError("Phase-B token shape/dtype changed")
    if tuple(array.shape) != (2, 34, 34, 40) or not np.isfinite(array).all():
        raise LudvigPFPRPhaseCError("Phase-B weighted token content is invalid")
    return np.asarray(array, dtype=np.float32)


def _import_ludvig_geometry(upstream: Path):
    root = upstream.resolve()
    if not root.is_dir():
        raise LudvigPFPRPhaseCError(f"Missing LUDVIG checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from gaussiansplatting.scene import GaussianModel
        from gaussiansplatting.scene.camera_scene import CamScene
    except Exception as error:
        raise LudvigPFPRPhaseCError("Unable to import LUDVIG Gaussian runtime") from error
    return GaussianModel, CamScene


def _driver_config(config: PhaseCConfig) -> PhaseBConfig:
    """Reuse the already audited process-local CUDA driver check."""

    return PhaseBConfig(
        phase_a_dir=Path("."),
        expected_phase_a_manifest_sha256="0" * 64,
        dino_checkpoint=Path("."),
        ludvig_upstream=config.ludvig_upstream,
        source_adapter_ledger=Path("."),
        dinov2_source=config.ludvig_upstream,
        output_dir=config.output_dir,
        driver_library_dir=config.driver_library_dir,
        device=config.device,
        expected_driver_version=config.expected_driver_version,
        expected_driver_libcuda_sha256=config.expected_driver_libcuda_sha256,
    )


def run_phase_c(config: PhaseCConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    """Uplift all frozen view features and atomically publish Phase C."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigPFPRPhaseCError(f"Refusing to overwrite Phase-C output: {output}")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise LudvigPFPRPhaseCError("Exact LUDVIG uplifting requires CUDA")
    driver = audit_cuda_driver_binding(_driver_config(config))
    if not torch.cuda.is_available():
        raise LudvigPFPRPhaseCError("CUDA is unavailable")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    phase_b_root, phase_b = audit_phase_b_attempt(config)
    phase_a_root = Path(str(phase_b.get("phase_a", {}).get("root", ""))).resolve()
    phase_a_manifest = _load_json(phase_a_root / "run_manifest.json", "Phase-A manifest")
    phase_a_binding = phase_b.get("phase_a", {}).get("manifest", {})
    _validate_file(
        phase_a_root / "run_manifest.json",
        str(phase_a_binding.get("sha256", "")),
        "Phase-A run manifest",
    )
    geometry = phase_a_manifest.get("geometry", {})
    geometry_path = Path(str(geometry.get("path", ""))).resolve()
    geometry_binding = _validate_file(
        geometry_path, str(geometry.get("sha256", "")), "Gaussian geometry"
    )
    colmap_root = phase_a_root / "staging" / "colmap"

    GaussianModel, CamScene = _import_ludvig_geometry(config.ludvig_upstream)
    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    scene = CamScene(str(colmap_root), h=480, w=640)
    cameras = list(scene.cameras)
    views = list(phase_b["views"])
    if len(cameras) != len(views):
        raise LudvigPFPRPhaseCError("LUDVIG camera/view counts disagree")
    camera_names = [str(camera.image_name) for camera in cameras]
    expected_names = [Path(str(view["source_staged_name"])).stem for view in views]
    if camera_names != expected_names:
        raise LudvigPFPRPhaseCError("LUDVIG camera sorting differs from Phase-B view order")
    if len(gaussian.get_xyz) != int(geometry.get("gaussians", -1)):
        raise LudvigPFPRPhaseCError("Loaded Gaussian count differs from Phase A")

    weights = torch.zeros_like(gaussian._opacity, dtype=torch.float32)
    features = torch.zeros(
        (len(gaussian._opacity), 40), dtype=torch.float32, device=device
    )
    started = time.monotonic()
    processed: list[dict[str, Any]] = []
    with torch.no_grad():
        for view, camera in zip(views, cameras):
            tokens = _load_weighted_tokens(phase_b_root, view)
            feature_map = reconstruct_ludvig_feature_map(
                torch.from_numpy(tokens).to(device), phase_b["sliding_window"]
            )
            if tuple(feature_map.shape) != (40, 480, 640):
                raise LudvigPFPRPhaseCError("Reconstructed feature-map shape changed")
            gaussian.apply_weights(camera, features, weights, feature_map)
            processed.append(
                {
                    "rank": int(view["rank"]),
                    "frame_id": int(view["frame_id"]),
                    "camera_name": str(camera.image_name),
                    "weighted_tokens_sha256": str(
                        view["eigval_weighted_tokens"]["sha256"]
                    ),
                }
            )
    features /= weights + 1e-8
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise LudvigPFPRPhaseCError("Uplift produced non-finite values")
    elapsed = time.monotonic() - started
    features_cpu = features.detach().cpu().numpy().astype(np.float32, copy=False)
    weights_cpu = weights.detach().cpu().numpy().astype(np.float32, copy=False)
    support = weights_cpu.reshape(-1)
    support_record = {
        "positive_gaussians": int(np.count_nonzero(support > 0)),
        "total_gaussians": int(support.size),
        "positive_fraction": float(np.mean(support > 0)),
        "minimum": float(support.min()),
        "maximum": float(support.max()),
        "quantiles_0_25_50_75_100": [
            float(value) for value in np.quantile(support, [0, 0.25, 0.5, 0.75, 1])
        ],
    }
    peak = {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.phase_c_tmp_", dir=output.parent))
    try:
        artifacts = {
            "gaussian_features": _binding_from_array(
                temporary, "uplift/gaussian_features.npy", features_cpu
            ),
            "inverse_render_weights": _binding_from_array(
                temporary, "uplift/inverse_render_weights.npy", weights_cpu
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": PHASE_C_SCHEMA_VERSION,
            "status": PHASE_C_STATUS,
            "result_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "scene_id": str(phase_b["scene_id"]),
            "attempt_dir": str(output),
            "argv": list(argv),
            "device": str(device),
            "cuda_driver_binding": driver,
            "cuda_peak_memory": peak,
            "phase_b": {
                "root": str(phase_b_root),
                "manifest_sha256": _require_sha256(
                    config.expected_phase_b_manifest_sha256,
                    "Phase-B manifest",
                ),
                "views_sha256": str(phase_b["views_sha256"]),
            },
            "phase_a": {
                "root": str(phase_a_root),
                "manifest_sha256": str(phase_a_binding["sha256"]),
            },
            "geometry": geometry_binding,
            "camera_order": processed,
            "camera_order_sha256": canonical_json_sha256(processed),
            "protocol": {
                "feature_reconstruction": (
                    "official_LUDVIG_SlidingWindow.fill_overlap_mean_then_"
                    "bilinear_resize_480x640"
                ),
                "uplift": "official_LUDVIG_GaussianModel.apply_weights",
                "aggregation": "sum_projected_features_div_sum_inverse_render_weights_plus_1e-8",
                "views": len(views),
                "feature_dimension": 40,
                "pruning": "disabled_because_300000_below_official_min_gaussians_400000",
                "method_queries_opened": False,
                "evaluator_private_manifest_opened": False,
            },
            "elapsed_seconds": float(elapsed),
            "support": support_record,
            "artifacts": artifacts,
            "phase_status": {
                "phase_a_cpu_staging": "bound_complete",
                "phase_b_dino_scene_features_and_pca": "bound_complete",
                "phase_c_inverse_render_uplift": "complete",
                "phase_d_pfpr_crop_scoring": "not_run",
                "phase_e_pfpr_evaluation": "not_run",
            },
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_sha256 = sha256_file(manifest_path)
        (temporary / "run_manifest.sha256").write_text(manifest_sha256 + "\n", encoding="ascii")
        if output.exists():
            raise LudvigPFPRPhaseCError(f"Refusing concurrent overwrite: {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": manifest_sha256}
