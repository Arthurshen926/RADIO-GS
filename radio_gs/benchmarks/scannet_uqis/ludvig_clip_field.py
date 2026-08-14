"""Authority-bound LUDVIG OpenCLIP text field for ScanNet-UQIS."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import sha256_file
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_b import (
    PhaseBConfig, audit_cuda_driver_binding,
)
from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_c import _binding_from_array

from .protocol import BENCHMARK_VERSION, canonical_json_sha256


SCHEMA_VERSION = "scannet_uqis_ludvig_clip_field_v1"
STATUS = "clip_text_field_complete_queries_not_run"
CLIP_DIMENSION = 512
OPEN_CLIP_VERSION = "2.29.0"
OPEN_CLIP_PY_TREE_SHA256 = "7ec7016bf7fb0bc5abef81812d82c7f07dca3e9cb1d4cb6581ea6638346ad52a"
OPEN_CLIP_CHECKPOINT_SHA256 = "3f25d29d3cc74e1d25d47e0593b4dd0864ced1e9b4d8e486a247ca4502f227f1"
OPEN_CLIP_CHECKPOINT_BYTES = 598_516_980
LUDVIG_CLIP_FILES = {
    "predictors/clip.py": "072adabf0f25335774990d2bd1f9452a90fabac123edbe6b89633c1bea5fd2c6",
    "clip_utils/openclip_encoder.py": "6fad755b1441672c90aa77d6c64e07a2db895378ad5baa04f59f42b00952c1f5",
    "configs/lerf_clip.yaml": "e0a489c2a4cace2e22db498abea27bfb7b5d783afbaadf0f3cca7a61b6429e48",
}


def _open_clip_tree(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("*.py")):
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return canonical_json_sha256(rows), rows


def run_clip_field(
    phase_a_dir: str | Path,
    *,
    expected_phase_a_manifest_sha256: str,
    ludvig_upstream: str | Path,
    open_clip_site_packages: str | Path,
    open_clip_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the unpruned 512-D field frozen by upstream lerf_clip.yaml."""

    import torch

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    phase_a_root = Path(phase_a_dir).resolve()
    phase_a_path = phase_a_root / "run_manifest.json"
    if sha256_file(phase_a_path) != expected_phase_a_manifest_sha256:
        raise ValueError("Phase-A bridge manifest changed")
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    ledger_path = phase_a_root / "source_adapter_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_body = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if (
        phase_a.get("benchmark_local_adapter") is not True
        or phase_a.get("result_eligible") is not False
        or ledger.get("benchmark_version") != BENCHMARK_VERSION
        or ledger.get("ledger_sha256") != canonical_json_sha256(ledger_body)
        or ledger.get("query_frames_opened") is not False
        or ledger.get("evaluator_labels_opened") is not False
    ):
        raise ValueError("Phase-A bridge authority changed")
    geometry = phase_a.get("geometry", {})
    geometry_path = Path(str(geometry.get("path", ""))).resolve()
    if (
        not geometry_path.is_file()
        or geometry_path.stat().st_size != geometry.get("bytes")
        or sha256_file(geometry_path) != geometry.get("sha256")
    ):
        raise ValueError("formal Gaussian geometry changed")

    upstream = Path(ludvig_upstream).resolve()
    ludvig_sources = []
    for relative, expected in LUDVIG_CLIP_FILES.items():
        path = upstream / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"LUDVIG CLIP source changed: {relative}")
        ludvig_sources.append({"relative_path": relative, "sha256": expected, "bytes": path.stat().st_size})
    site_packages = Path(open_clip_site_packages).resolve()
    open_clip_root = site_packages / "open_clip"
    tree_hash, tree_rows = _open_clip_tree(open_clip_root)
    if tree_hash != OPEN_CLIP_PY_TREE_SHA256:
        raise ValueError("OpenCLIP Python source tree changed")
    checkpoint = Path(open_clip_checkpoint).resolve()
    if (
        not checkpoint.is_file() or checkpoint.stat().st_size != OPEN_CLIP_CHECKPOINT_BYTES
        or sha256_file(checkpoint) != OPEN_CLIP_CHECKPOINT_SHA256
    ):
        raise ValueError("OpenCLIP checkpoint changed")
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))
    import open_clip
    if str(open_clip.__version__) != OPEN_CLIP_VERSION:
        raise ValueError("OpenCLIP runtime version changed")

    driver = audit_cuda_driver_binding(PhaseBConfig(
        phase_a_dir=Path("."), expected_phase_a_manifest_sha256="0" * 64,
        dino_checkpoint=Path("."), ludvig_upstream=upstream,
        source_adapter_ledger=Path("."), dinov2_source=upstream,
        output_dir=output, device=device,
    ))
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("exact CLIP field construction requires CUDA")
    torch.cuda.set_device(cuda_device)
    torch.cuda.reset_peak_memory_stats()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from gaussiansplatting.scene import GaussianModel
    from gaussiansplatting.scene.camera_scene import CamScene
    from predictors.clip import CLIPDataset

    gaussian = GaussianModel(sh_degree=0)
    gaussian.load_ply(str(geometry_path))
    cameras = list(CamScene(str(phase_a_root / "staging/colmap"), h=480, w=640).cameras)
    if len(cameras) != 120:
        raise ValueError("CLIP field requires all 120 legal views")
    gaussian_count = int(len(gaussian.get_xyz))
    if gaussian_count != int(geometry.get("gaussians", -1)):
        raise ValueError("Gaussian count changed")
    dataset = CLIPDataset(
        directory=str(phase_a_root / "staging/colmap/images"),
        scene=str(phase_a["scene_id"]), gaussian=gaussian, cameras=cameras,
        render_fn=None, height=480, width=640,
    )
    weights = torch.zeros_like(gaussian._opacity, dtype=torch.float32)
    features = torch.zeros((gaussian_count, CLIP_DIMENSION), dtype=torch.float32, device=cuda_device)
    processed = []
    started = time.monotonic()
    with torch.no_grad():
        for rank, camera in enumerate(cameras):
            feature_map, loaded_camera = dataset[rank]
            if loaded_camera is not camera or tuple(feature_map.shape) != (CLIP_DIMENSION, 480, 640):
                raise ValueError("OpenCLIP view/camera contract changed")
            if not bool(torch.isfinite(feature_map).all()):
                raise ValueError("OpenCLIP produced a non-finite feature map")
            gaussian.apply_weights(camera, features, weights, feature_map)
            processed.append({"rank": rank, "camera_name": str(camera.image_name)})
        features /= weights + 1e-8
        # Frozen upstream lerf_clip.yaml: normalize=true, prune_gaussians absent.
        features /= torch.linalg.vector_norm(features, dim=1, keepdim=True) + 1e-6
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(weights).all()):
            raise ValueError("CLIP uplift produced non-finite field values")
    elapsed = time.monotonic() - started
    support = weights.detach().cpu().numpy().astype(np.float32, copy=False)
    feature_array = features.detach().cpu().numpy().astype(np.float32, copy=False)
    peak = {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=output.parent))
    try:
        artifacts = {
            "gaussian_features": _binding_from_array(temporary, "field/gaussian_features.npy", feature_array),
            "inverse_render_weights": _binding_from_array(temporary, "field/inverse_render_weights.npy", support),
        }
        support_flat = support.reshape(-1)
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": STATUS,
            "benchmark_version": BENCHMARK_VERSION, "result_eligible": False,
            "field_construction_eligible": True, "official_ludvig_reproduction": False,
            "paper_metric_comparable": False, "benchmark_local_adapter": True,
            "scene_id": phase_a["scene_id"], "field_role": "ludvig_clip_for_text",
            "attempt_dir": str(output), "argv": list(argv), "device": device,
            "phase_a": {"root": str(phase_a_root), "manifest_sha256": expected_phase_a_manifest_sha256,
                        "source_adapter_ledger_sha256": sha256_file(ledger_path),
                        "mapping_observation_receipt_sha256": phase_a["mapping_observation_receipt"]["sha256"],
                        "geometry_receipt_sha256": phase_a["geometry_run_receipt"]["sha256"]},
            "geometry": {"path": str(geometry_path), "sha256": geometry["sha256"],
                         "bytes": geometry["bytes"], "gaussians": gaussian_count},
            "open_clip": {"version": OPEN_CLIP_VERSION, "python_source_root": str(open_clip_root),
                          "python_source_tree_sha256": tree_hash, "python_source_files": tree_rows,
                          "checkpoint": {"path": str(checkpoint), "sha256": OPEN_CLIP_CHECKPOINT_SHA256,
                                         "bytes": OPEN_CLIP_CHECKPOINT_BYTES},
                          "model": "ViT-B-16", "pretrained": "laion2b_s34b_b88k"},
            "ludvig_sources": ludvig_sources, "cuda_driver_binding": driver,
            "cuda_peak_memory": peak, "camera_order": processed,
            "camera_order_sha256": canonical_json_sha256(processed),
            "protocol": {"views": 120, "feature_dimension": CLIP_DIMENSION,
                         "tile_ratios": np.linspace(0.05, 0.5, 7).tolist(),
                         "uplift": "official_LUDVIG_GaussianModel.apply_weights",
                         "normalization": "upstream_l2_plus_1e-6",
                         "pruning": "disabled_exact_lerf_clip_yaml",
                         "method_queries_opened": False, "evaluator_private_manifest_opened": False},
            "elapsed_seconds": float(elapsed),
            "support": {"positive_gaussians": int(np.count_nonzero(support_flat > 0)),
                        "total_gaussians": gaussian_count,
                        "positive_fraction": float(np.mean(support_flat > 0)),
                        "minimum": float(support_flat.min()), "maximum": float(support_flat.max())},
            "artifacts": artifacts,
        }
        path = temporary / "run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = sha256_file(path)
        (temporary / "run_manifest.sha256").write_text(digest + "\n", encoding="ascii")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": digest}
