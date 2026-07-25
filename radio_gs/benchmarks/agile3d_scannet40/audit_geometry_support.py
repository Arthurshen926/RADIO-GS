#!/usr/bin/env python3
"""Audit the label-free all-Gaussian support ceiling for AGILE3D fields.

This is a construction diagnostic, not an AGILE interaction result.  It reads
only released geometry/RGB coordinates, a query-free raw MPR cache, and the
field's frozen Gaussian geometry.  The report answers a narrow question before
objects or labels are opened: can this geometry *in principle* cover the
official 5 cm domain if every Gaussian had a semantic descriptor?  If not,
adding more semantic MPR views alone cannot make the fixed support gate pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.querying.query_compilers import continuous_gaussian_readout

from .evaluate_canonical_field import (
    _field_source_metadata,
    _gaussian_covariances,
    _load_geometry_model,
    _nearest_candidate_indices,
    _read_official_geometry,
    observation_source_from_render_contract,
    validate_full_observation_mpr_contract,
)
from .protocol import quantize_scannet_points


def all_gaussian_support_fraction(
    *,
    gaussian_xyz: torch.Tensor,
    gaussian_covariance: torch.Tensor,
    gaussian_opacity: torch.Tensor,
    official_xyz: np.ndarray,
    candidate_k: int,
    support_threshold: float,
    evaluation_voxel_size_m: float,
) -> tuple[float, dict[str, float]]:
    """Return support using every Gaussian, independent of semantic validity."""

    xyz = torch.as_tensor(gaussian_xyz).float()
    covariance = torch.as_tensor(gaussian_covariance).float()
    opacity = torch.as_tensor(gaussian_opacity).float().reshape(-1)
    points = torch.as_tensor(official_xyz).float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must be [N,3]")
    if covariance.shape != (len(xyz), 3, 3) or opacity.shape != (len(xyz),):
        raise ValueError("Gaussian geometry tensors do not align")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("official_xyz must be [P,3]")
    if int(candidate_k) <= 0 or float(support_threshold) < 0:
        raise ValueError("candidate_k/support_threshold are invalid")
    voxel_size = float(evaluation_voxel_size_m)
    if voxel_size < 0:
        raise ValueError("evaluation_voxel_size_m must be non-negative")
    variance = voxel_size**2 / 12.0
    precision = torch.linalg.pinv(
        covariance + variance * torch.eye(3, dtype=covariance.dtype)
    )
    indices = _nearest_candidate_indices(xyz, points.numpy(), count=int(candidate_k))
    _readout, support = continuous_gaussian_readout(
        xyz,
        covariance,
        torch.ones(len(xyz), dtype=torch.float32),
        points,
        gaussian_precision=precision,
        opacity=opacity,
        candidate_k=int(candidate_k),
        candidate_indices=indices,
    )
    fraction = float((support >= float(support_threshold)).float().mean())
    values = torch.quantile(
        support.float(), torch.tensor([0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 1.0])
    )
    return fraction, {
        key: float(value)
        for key, value in zip(
            ("p00", "p01", "p05", "p10", "p50", "p90", "p100"),
            values.tolist(),
        )
    }


def _scene_report(
    *,
    benchmark_root: Path,
    field_root: Path,
    scene_id: str,
    candidate_k: int,
    support_threshold: float,
    minimum_support_fraction: float,
    evaluation_voxel_size_m: float,
    device: torch.device,
) -> dict[str, object]:
    scene_dir = field_root / "canonical_fields" / scene_id
    raw_path = scene_dir / "raw_radio_mpr.pt"
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing raw MPR cache: {raw_path}")
    raw = torch.load(raw_path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError(f"raw MPR cache is malformed: {raw_path}")
    metadata = _field_source_metadata(scene_dir)
    config_path = Path(str(metadata.get("config", "")))
    checkpoint_path = Path(str(metadata.get("checkpoint", "")))
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("raw MPR cache lacks a valid geometry render contract")
    source = observation_source_from_render_contract(config_path)
    validate_full_observation_mpr_contract(
        "scannet_full_observation_pilot",
        metadata,
        expected_source_contract_sha256=str(source["field_source_contract_sha256"]),
        expected_source_contract_version=str(source["field_source_contract_version"]),
    )

    ply_path = benchmark_root / "scans" / f"{scene_id}.ply"
    xyz, rgb = _read_official_geometry(ply_path)
    quantized = quantize_scannet_points(
        xyz,
        rgb,
        np.zeros(len(xyz), dtype=np.int32),
        voxel_size=float(evaluation_voxel_size_m),
    )
    official_xyz = quantized.raw_coordinates + xyz.min(axis=0, keepdims=True)
    config = load_config(str(config_path))
    model = _load_geometry_model(config, str(checkpoint_path), device)
    try:
        gaussian_xyz = model.get_xyz().detach().float().cpu()
        expected_xyz = torch.as_tensor(raw.get("xyz")).float().cpu()
        if gaussian_xyz.shape != expected_xyz.shape or not torch.allclose(
            gaussian_xyz, expected_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("raw MPR and frozen Gaussian geometry do not align")
        covariance = _gaussian_covariances(model).detach().float().cpu()
        opacity = model.get_opacity().detach().float().reshape(-1).cpu()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fraction, quantiles = all_gaussian_support_fraction(
        gaussian_xyz=gaussian_xyz,
        gaussian_covariance=covariance,
        gaussian_opacity=opacity,
        official_xyz=official_xyz,
        candidate_k=int(candidate_k),
        support_threshold=float(support_threshold),
        evaluation_voxel_size_m=float(evaluation_voxel_size_m),
    )
    return {
        "scene_id": scene_id,
        "quantized_points": int(len(official_xyz)),
        "all_gaussian_count": int(len(gaussian_xyz)),
        "geometry_only_support_fraction": fraction,
        "geometry_only_support_quantiles": quantiles,
        "minimum_support_fraction": float(minimum_support_fraction),
        "geometry_ceiling_passes_gate": bool(fraction >= float(minimum_support_fraction)),
        "geometry_rebuild_required": bool(fraction < float(minimum_support_fraction)),
        "readout_candidate_count": int(candidate_k),
        "readout_support_threshold": float(support_threshold),
        "evaluation_voxel_size_m": float(evaluation_voxel_size_m),
        "readout_kernel": "gaussian_convolved_with_evaluator_voxel_cell",
        "raw_mpr_contract": str(
            dict(metadata.get("observation_lifting_contract", {})).get("name", "")
        ),
        "raw_mpr_source_view_count": int(
            metadata.get("full_observation_source_view_count", 0)
        ),
        "labels_opened": False,
        "object_list_opened": False,
        "test_set_calibration": False,
        **source,
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    benchmark_root = Path(args.benchmark_root)
    field_root = Path(args.field_root)
    requested = sorted(set(str(args.scene_names).replace(",", " ").split()))
    if not requested:
        raise ValueError("scene_names must be non-empty")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [
        _scene_report(
            benchmark_root=benchmark_root,
            field_root=field_root,
            scene_id=scene,
            candidate_k=int(args.readout_candidate_k),
            support_threshold=float(args.readout_support_threshold),
            minimum_support_fraction=float(args.minimum_support_fraction),
            evaluation_voxel_size_m=float(args.evaluation_voxel_size_m),
            device=device,
        )
        for scene in requested
    ]
    return {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "mode": "label_free_all_gaussian_geometry_support_ceiling",
        "protocol": {
            "labels_opened": False,
            "object_list_opened": False,
            "test_set_calibration": False,
            "evaluation_voxel_size_m": float(args.evaluation_voxel_size_m),
            "readout_support_threshold": float(args.readout_support_threshold),
            "minimum_support_fraction": float(args.minimum_support_fraction),
            "all_gaussians_read": True,
        },
        "scene_geometry_support": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--field-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-names", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--readout-candidate-k", type=int, default=64)
    parser.add_argument("--readout-support-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.95)
    parser.add_argument("--evaluation-voxel-size-m", type=float, default=0.05)
    args = parser.parse_args()
    report = audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
