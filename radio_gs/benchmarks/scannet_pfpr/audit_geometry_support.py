#!/usr/bin/env python3
"""Audit label-free Gaussian coverage of the public PFPR point domain.

This is a field-construction diagnostic, not a PFPR retrieval result.  It
opens the method manifest's public 5 cm candidate coordinates and a frozen
Gaussian geometry, but never evaluator-private anchors, source-frame records,
query crops, masks, instances, semantic labels, ranks, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radio_gs.benchmarks.agile3d_scannet40.audit_geometry_support import (
    all_gaussian_support_fraction,
)
from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    _gaussian_covariances,
    _load_geometry_model,
)
from radio_gs.config import load_config


MODE = "label_free_public_candidate_geometry_support_ceiling"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_scene_domains(benchmark_dir: Path) -> dict[str, Mapping[str, Any]]:
    manifest_path = benchmark_dir / "manifest.method.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("benchmark_version") != "scannet-pfpr-small-v2":
        raise ValueError("PFPR geometry-support audit requires the immutable v2 method manifest")
    rows = payload.get("scene_domains", [])
    if not isinstance(rows, list):
        raise ValueError("PFPR method manifest has malformed scene domains")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PFPR method manifest has malformed scene-domain row")
        scene = str(row.get("scene_id", ""))
        if not scene or scene in result:
            raise ValueError("PFPR method manifest has non-unique scene domains")
        if not bool(row.get("geometry_only", False)):
            raise ValueError(f"{scene}: PFPR support audit requires a public geometry-only domain")
        result[scene] = row
    return result


def _public_candidate_xyz(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("candidate_xyz_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing public PFPR candidate geometry: {path}")
    expected = str(row.get("candidate_xyz_sha256", ""))
    if not expected or _sha256(path) != expected:
        raise ValueError(f"public PFPR candidate geometry digest does not match: {path}")
    xyz = np.asarray(np.load(path), dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f"public PFPR candidate geometry is not finite [N,3]: {path}")
    return np.ascontiguousarray(xyz)


def public_geometry_support_record(
    *,
    scene_id: str,
    candidate_xyz: np.ndarray,
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    gaussian_opacity: torch.Tensor,
    candidate_k: int,
    support_threshold: float,
    minimum_support_fraction: float,
    voxel_size_m: float,
) -> dict[str, object]:
    """Measure the all-Gaussian support ceiling without PFPR query evidence."""

    fraction, quantiles = all_gaussian_support_fraction(
        gaussian_xyz=gaussian_xyz,
        gaussian_covariance=gaussian_covariance,
        gaussian_opacity=gaussian_opacity,
        official_xyz=np.asarray(candidate_xyz, dtype=np.float32),
        candidate_k=int(candidate_k),
        support_threshold=float(support_threshold),
        evaluation_voxel_size_m=float(voxel_size_m),
    )
    return {
        "scene_id": str(scene_id),
        "public_candidate_points": int(len(candidate_xyz)),
        "all_gaussian_count": int(len(gaussian_xyz)),
        "geometry_only_support_fraction": float(fraction),
        "geometry_only_support_quantiles": quantiles,
        "minimum_support_fraction": float(minimum_support_fraction),
        "geometry_ceiling_passes_gate": bool(fraction >= float(minimum_support_fraction)),
        "geometry_rebuild_required": bool(fraction < float(minimum_support_fraction)),
        "readout_candidate_count": int(candidate_k),
        "readout_support_threshold": float(support_threshold),
        "evaluation_voxel_size_m": float(voxel_size_m),
        "readout_kernel": "gaussian_convolved_with_public_candidate_voxel_cell",
        "private_anchors_opened": False,
        "query_crop_pixels_opened": False,
        "instance_or_semantic_labels_opened": False,
        "test_set_calibration": False,
    }


def validate_geometry_support_gate(
    payload: Mapping[str, object],
    *,
    scene_id: str,
    minimum_support_fraction: float,
) -> float:
    """Return support after proving that an audit is safe and passes its gate."""

    if payload.get("mode") != MODE:
        raise ValueError("PFPR geometry support audit has an invalid mode")
    protocol = dict(payload.get("protocol", {}))
    for key in (
        "private_anchors_opened",
        "query_crop_pixels_opened",
        "instance_or_semantic_labels_opened",
        "test_set_calibration",
    ):
        if protocol.get(key) is not False:
            raise ValueError(f"PFPR geometry support audit is not label/query free ({key})")
    rows = [
        row
        for row in payload.get("scene_geometry_support", [])
        if isinstance(row, Mapping) and str(row.get("scene_id", "")) == str(scene_id)
    ]
    if len(rows) != 1:
        raise ValueError("PFPR geometry support audit has no unique scene row")
    fraction = float(rows[0].get("geometry_only_support_fraction", 0.0))
    if fraction < float(minimum_support_fraction):
        raise ValueError(
            f"{scene_id}: rebuilt all-Gaussian geometry support {fraction:.6f} "
            f"< fixed gate {float(minimum_support_fraction):.6f}"
        )
    return fraction


def _scene_report(
    *,
    benchmark_dir: Path,
    field_root: Path,
    scene_id: str,
    candidate_k: int,
    support_threshold: float,
    minimum_support_fraction: float,
    voxel_size_m: float,
    device: torch.device,
) -> dict[str, object]:
    domain = _method_scene_domains(benchmark_dir).get(str(scene_id))
    if domain is None:
        raise ValueError(f"PFPR method manifest has no public domain for {scene_id}")
    candidate_xyz = _public_candidate_xyz(domain)
    config_path = field_root / "render_contracts" / f"{scene_id}.yaml"
    checkpoint_path = field_root / "render_contracts" / f"{scene_id}.geometry_renderer.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing frozen geometry render contract for {scene_id}")
    model = _load_geometry_model(load_config(str(config_path)), str(checkpoint_path), device)
    try:
        gaussian_xyz = model.get_xyz().detach().float().cpu()
        covariance = _gaussian_covariances(model).detach().float().cpu()
        opacity = model.get_opacity().detach().float().reshape(-1).cpu()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return public_geometry_support_record(
        scene_id=scene_id,
        candidate_xyz=candidate_xyz,
        gaussian_xyz=gaussian_xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=opacity,
        candidate_k=int(candidate_k),
        support_threshold=float(support_threshold),
        minimum_support_fraction=float(minimum_support_fraction),
        voxel_size_m=float(voxel_size_m),
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    requested = sorted(set(str(args.scene_names).replace(",", " ").split()))
    if not requested:
        raise ValueError("scene_names must be non-empty")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [
        _scene_report(
            benchmark_dir=Path(args.benchmark_dir),
            field_root=Path(args.field_root),
            scene_id=scene,
            candidate_k=int(args.readout_candidate_k),
            support_threshold=float(args.readout_support_threshold),
            minimum_support_fraction=float(args.minimum_support_fraction),
            voxel_size_m=float(args.candidate_voxel_size_m),
            device=device,
        )
        for scene in requested
    ]
    return {
        "benchmark": "ScanNet-PFPR-Small v2",
        "mode": MODE,
        "protocol": {
            "private_anchors_opened": False,
            "query_crop_pixels_opened": False,
            "instance_or_semantic_labels_opened": False,
            "test_set_calibration": False,
            "candidate_domain": "public_geometry_only_5cm",
            "evaluation_voxel_size_m": float(args.candidate_voxel_size_m),
            "readout_candidate_k": int(args.readout_candidate_k),
            "readout_support_threshold": float(args.readout_support_threshold),
            "minimum_support_fraction": float(args.minimum_support_fraction),
            "all_gaussians_read": True,
        },
        "scene_geometry_support": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--field-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-names", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--readout-candidate-k", type=int, default=64)
    parser.add_argument("--readout-support-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.95)
    parser.add_argument("--candidate-voxel-size-m", type=float, default=0.05)
    args = parser.parse_args()
    payload = audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
