"""UQIS-local exact LUDVIG DINO inverse-render uplift and 600k pruning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    EXPECTED_DRIVER_LIBCUDA_SHA256,
    EXPECTED_NVIDIA_DRIVER_VERSION,
    PhaseBConfig,
    audit_cuda_driver_binding,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import (
    _binding_from_array,
    _import_ludvig_geometry,
    _load_json,
    _load_weighted_tokens,
    _validate_file,
    audit_phase_b_attempt,
    reconstruct_ludvig_feature_map,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256

from .protocol import BENCHMARK_VERSION


SCHEMA_VERSION = "scannet_uqis_ludvig_dino_uplift_v1"
STATUS = "dino_field_uplift_complete_queries_not_run"
FEATURE_DIMENSION = 40
OFFICIAL_PRUNE_GAUSSIANS = 600_000
OFFICIAL_MIN_GAUSSIANS = 400_000


class LudvigUQISDinoUpliftError(RuntimeError):
    """Raised before publishing an invalid UQIS LUDVIG DINO field."""


@dataclass(frozen=True)
class UpliftConfig:
    phase_b_dir: Path
    expected_phase_b_manifest_sha256: str
    ludvig_upstream: Path
    output_dir: Path
    driver_library_dir: Path = Path("/root/baselines/LUDVIG/.driver535")
    device: str = "cuda:0"
    prune_gaussians: int = OFFICIAL_PRUNE_GAUSSIANS
    expected_driver_version: str = EXPECTED_NVIDIA_DRIVER_VERSION
    expected_driver_libcuda_sha256: str = EXPECTED_DRIVER_LIBCUDA_SHA256


def _driver_config(config: UpliftConfig) -> PhaseBConfig:
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


def _ply_binding(root: Path, relative: str, gaussian: Any) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    gaussian.save_ply(str(path))
    return {
        "relative_path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "gaussians": int(len(gaussian.get_xyz)),
    }


def run_uplift(config: UpliftConfig, *, argv: Sequence[str] = ()) -> dict[str, Any]:
    """Build one query-free DINO feature field from a frozen UQIS bridge."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigUQISDinoUpliftError(f"refusing to overwrite output: {output}")
    if config.prune_gaussians != OFFICIAL_PRUNE_GAUSSIANS:
        raise LudvigUQISDinoUpliftError("the frozen LUDVIG DINO field keeps exactly 600000 Gaussians")
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise LudvigUQISDinoUpliftError("exact LUDVIG uplifting requires CUDA")
    driver = audit_cuda_driver_binding(_driver_config(config))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    phase_b_root, phase_b = audit_phase_b_attempt(
        type("PhaseCAuditConfig", (), {
            "phase_b_dir": config.phase_b_dir,
            "expected_phase_b_manifest_sha256": config.expected_phase_b_manifest_sha256,
        })()
    )
    phase_a_root = Path(str(phase_b.get("phase_a", {}).get("root", ""))).resolve()
    phase_a_binding = phase_b.get("phase_a", {}).get("manifest", {})
    _validate_file(
        phase_a_root / "run_manifest.json",
        str(phase_a_binding.get("sha256", "")),
        "Phase-A run manifest",
    )
    phase_a = _load_json(phase_a_root / "run_manifest.json", "Phase-A run manifest")
    ledger_binding = phase_a.get("source_adapter_ledger", {})
    ledger_path = Path(str(ledger_binding.get("path", ""))).resolve()
    _validate_file(
        ledger_path,
        str(ledger_binding.get("sha256", "")),
        "UQIS source-adapter ledger",
    )
    ledger = _load_json(ledger_path, "UQIS source-adapter ledger")
    ledger_body = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if (
        ledger.get("benchmark_version") != BENCHMARK_VERSION
        or ledger.get("ledger_sha256") != canonical_json_sha256(ledger_body)
        or ledger.get("query_frames_opened") is not False
        or ledger.get("evaluator_labels_opened") is not False
    ):
        raise LudvigUQISDinoUpliftError("Phase-A bridge benchmark authority changed")
    geometry = phase_a.get("geometry", {})
    geometry_path = Path(str(geometry.get("path", ""))).resolve()
    source_geometry = _validate_file(
        geometry_path, str(geometry.get("sha256", "")), "formal Gaussian geometry"
    )

    GaussianModel, CamScene = _import_ludvig_geometry(config.ludvig_upstream)
    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    cameras = list(CamScene(str(phase_a_root / "staging" / "colmap"), h=480, w=640).cameras)
    views = list(phase_b["views"])
    expected_names = [Path(str(row["source_staged_name"])).stem for row in views]
    if len(cameras) != 120 or [str(camera.image_name) for camera in cameras] != expected_names:
        raise LudvigUQISDinoUpliftError("camera order differs from the frozen Phase-B order")
    source_count = int(len(gaussian.get_xyz))
    if source_count != int(geometry.get("gaussians", -1)):
        raise LudvigUQISDinoUpliftError("source Gaussian count changed")
    if source_count <= OFFICIAL_MIN_GAUSSIANS or source_count < config.prune_gaussians:
        raise LudvigUQISDinoUpliftError("source field is too small for frozen LUDVIG pruning")

    weights = torch.zeros_like(gaussian._opacity, dtype=torch.float32)
    features = torch.zeros((source_count, FEATURE_DIMENSION), dtype=torch.float32, device=device)
    processed: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.no_grad():
        for view, camera in zip(views, cameras):
            tokens = _load_weighted_tokens(phase_b_root, view)
            feature_map = reconstruct_ludvig_feature_map(
                torch.from_numpy(tokens).to(device), phase_b["sliding_window"]
            )
            if tuple(feature_map.shape) != (FEATURE_DIMENSION, 480, 640):
                raise LudvigUQISDinoUpliftError("reconstructed feature-map shape changed")
            gaussian.apply_weights(camera, features, weights, feature_map)
            processed.append({
                "rank": int(view["rank"]),
                "frame_id": int(view["frame_id"]),
                "camera_name": str(camera.image_name),
                "weighted_tokens_sha256": str(view["eigval_weighted_tokens"]["sha256"]),
            })
        features /= weights + 1e-8
        if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
            raise LudvigUQISDinoUpliftError("uplift produced non-finite values")
        # This is the exact integer-pruning branch in upstream utils/solver.py.
        keep = torch.argsort(weights.squeeze())[-config.prune_gaussians:]
        selected_features = features[keep].detach().cpu().numpy().astype(np.float32, copy=False)
        selected_weights = weights[keep].detach().cpu().numpy().astype(np.float32, copy=False)
        selected_indices = keep.detach().cpu().numpy().astype(np.int64, copy=False)
        # Upstream load_ply leaves this non-persistent densification cache on
        # CPU while every persistent Gaussian tensor is on CUDA.  Its pruning
        # helper indexes the cache too, so align the cache device first.
        if int(gaussian.max_radii2D.numel()) != source_count:
            gaussian.max_radii2D = torch.zeros(
                source_count, dtype=torch.float32, device=keep.device
            )
        else:
            gaussian.max_radii2D = gaussian.max_radii2D.to(keep.device)
        gaussian.prune_points_noopt(keep)
    elapsed = time.monotonic() - started
    support = weights.detach().cpu().numpy().reshape(-1)
    peak = {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        artifacts = {
            "gaussian_features": _binding_from_array(
                temporary, "field/gaussian_features.npy", selected_features
            ),
            "inverse_render_weights": _binding_from_array(
                temporary, "field/inverse_render_weights.npy", selected_weights
            ),
            "source_indices": _binding_from_array(
                temporary, "field/source_indices.npy", selected_indices
            ),
            "pruned_geometry": _ply_binding(
                temporary, "field/point_cloud.ply", gaussian
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "status": STATUS,
            "result_eligible": False,
            "field_construction_eligible": True,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "benchmark_local_adapter": True,
            "scene_id": str(phase_b["scene_id"]),
            "field_role": "ludvig_dino_for_image_point_2d_point_3d",
            "attempt_dir": str(output),
            "argv": list(argv),
            "device": str(device),
            "cuda_driver_binding": driver,
            "cuda_peak_memory": peak,
            "phase_b": {
                "root": str(phase_b_root),
                "manifest_sha256": config.expected_phase_b_manifest_sha256,
                "views_sha256": str(phase_b["views_sha256"]),
            },
            "phase_a": {
                "root": str(phase_a_root),
                "manifest_sha256": str(phase_a_binding["sha256"]),
                "source_adapter_ledger_sha256": str(ledger_binding["sha256"]),
                "mapping_observation_receipt_sha256": phase_a.get("mapping_observation_receipt", {}).get("sha256"),
                "geometry_receipt_sha256": phase_a.get("geometry_run_receipt", {}).get("sha256"),
            },
            "source_geometry": source_geometry,
            "camera_order": processed,
            "camera_order_sha256": canonical_json_sha256(processed),
            "protocol": {
                "views": 120,
                "feature_dimension": FEATURE_DIMENSION,
                "feature_reconstruction": "official_LUDVIG_SlidingWindow.fill_then_bilinear_480x640",
                "uplift": "official_LUDVIG_GaussianModel.apply_weights",
                "aggregation": "sum_features_div_inverse_render_weights_plus_1e-8",
                "pruning": "official_LUDVIG_integer_top_inverse_render_weight",
                "source_gaussians": source_count,
                "kept_gaussians": config.prune_gaussians,
                "method_queries_opened": False,
                "evaluator_private_manifest_opened": False,
            },
            "elapsed_seconds": float(elapsed),
            "support": {
                "positive_gaussians": int(np.count_nonzero(support > 0)),
                "total_gaussians": int(support.size),
                "positive_fraction": float(np.mean(support > 0)),
                "minimum": float(support.min()),
                "maximum": float(support.max()),
                "selected_minimum": float(selected_weights.min()),
                "selected_maximum": float(selected_weights.max()),
            },
            "artifacts": artifacts,
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        digest = sha256_file(manifest_path)
        (temporary / "run_manifest.sha256").write_text(digest + "\n", encoding="ascii")
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": digest}
