#!/usr/bin/env python3
"""Convert an official AGILE3D ScanNet cloud to label-free 5 cm Gaussians."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch

from radio_gs.benchmarks.agile3d_scannet40.protocol import quantize_scannet_points
from radio_gs.scripts.train_colmap_gs import save_ply


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    source = Path(args.input_ply).resolve()
    vertex = PlyData.read(str(source))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "R", "G", "B"}
    if not required.issubset(names):
        raise ValueError(f"{source} lacks AGILE3D geometry/color properties")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float32
    )
    rgb = np.column_stack([vertex[name] for name in ("R", "G", "B")]).astype(
        np.float32
    ) / 255.0
    # Labels are deliberately neither indexed nor used. Zeros only satisfy the
    # generic quantizer's aligned-array contract.
    quantized = quantize_scannet_points(
        xyz,
        rgb,
        np.zeros(len(xyz), dtype=np.int32),
        voxel_size=float(args.voxel_size),
    )
    unique_xyz = xyz[quantized.unique_map]
    unique_rgb = rgb[quantized.unique_map]
    count = len(unique_xyz)
    c0 = 0.28209479177387814
    opacity = min(max(float(args.opacity), 1e-4), 1.0 - 1e-4)
    params = {
        "means": torch.from_numpy(unique_xyz),
        "scales": torch.full(
            (count, 3), math.log(float(args.gaussian_scale)), dtype=torch.float32
        ),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 1),
        "opacities": torch.full(
            (count, 1), math.log(opacity / (1.0 - opacity)), dtype=torch.float32
        ),
        "sh0": torch.from_numpy((unique_rgb - 0.5) / c0).unsqueeze(1),
        "shN": torch.empty(count, 0, 3),
    }
    output = Path(args.output_ply)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_ply(str(output), params, sh_degree=0)
    mapping = Path(args.output_mapping)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        mapping,
        unique_map=quantized.unique_map,
        inverse_map=quantized.inverse_map,
        quantized_xyz=unique_xyz,
    )
    report = {
        "schema_version": 1,
        "source_ply": str(source),
        "source_ply_sha256": _sha256(source),
        "output_ply": str(output.resolve()),
        "output_ply_sha256": _sha256(output),
        "output_mapping": str(mapping.resolve()),
        "input_points": len(xyz),
        "gaussians": count,
        "voxel_size_m": float(args.voxel_size),
        "gaussian_scale_m": float(args.gaussian_scale),
        "opacity": opacity,
        "quantization_order": "MinkowskiEngine_CPU_first_occurrence",
        "label_property_accessed": False,
        "labels_used": False,
        "queries_opened": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--output-mapping", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--gaussian-scale", type=float, default=0.025)
    parser.add_argument("--opacity", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
