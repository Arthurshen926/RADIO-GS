#!/usr/bin/env python3
"""Diagnose row-wise XYZ alignment between ScanNet OpenGaussian PLY files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData


def read_xyz_ply(path: str | Path) -> np.ndarray:
    """Read vertex XYZ coordinates from a PLY file as ``float32 [N, 3]``."""
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    missing = [name for name in ("x", "y", "z") if name not in names]
    if missing:
        raise ValueError(f"PLY is missing XYZ properties {missing}: {path}")
    return np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )


def compare_xyz_rows(
    name: str,
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> dict:
    """Compare two XYZ arrays by row and return count/distance diagnostics."""
    left = np.asarray(left_xyz, dtype=np.float32).reshape(-1, 3)
    right = np.asarray(right_xyz, dtype=np.float32).reshape(-1, 3)
    left_count = int(left.shape[0])
    right_count = int(right.shape[0])
    count = min(left_count, right_count)
    if count == 0:
        distances = np.empty(0, dtype=np.float32)
    else:
        distances = np.linalg.norm(left[:count] - right[:count], axis=1)

    count_match = left_count == right_count
    max_distance = float(distances.max()) if distances.size else 0.0
    mean_distance = float(distances.mean()) if distances.size else 0.0
    p95_distance = float(np.percentile(distances, 95)) if distances.size else 0.0
    within_tolerance_ratio = (
        float(np.mean(distances <= float(tolerance))) if distances.size else float(count_match)
    )
    return {
        "name": str(name),
        "left_count": left_count,
        "right_count": right_count,
        "compared_count": int(count),
        "count_match": bool(count_match),
        "mean_distance": mean_distance,
        "max_distance": max_distance,
        "p95_distance": p95_distance,
        "within_tolerance_ratio": within_tolerance_ratio,
        "row_aligned": bool(count_match and max_distance <= float(tolerance)),
        "tolerance": float(tolerance),
    }


def _default_label_ply(prepared_root: Path, scene: str) -> Path:
    scene_root = prepared_root / scene
    preferred = scene_root / f"{scene}_vh_clean_2.labels.ply"
    if preferred.exists():
        return preferred
    matches = sorted(scene_root.glob("*.labels.ply"))
    if not matches:
        raise FileNotFoundError(f"No *.labels.ply file found in {scene_root}")
    return matches[0]


def _default_geometry_ply(
    geometry_root: Path,
    scene: str,
    geom_tag: str,
    iters: int,
) -> Path:
    return (
        geometry_root
        / scene
        / geom_tag
        / "point_cloud"
        / f"iteration_{int(iters)}"
        / "point_cloud.ply"
    )


def run_diagnostics(args: argparse.Namespace) -> dict:
    prepared_root = Path(args.prepared_root)
    points_ply = Path(args.points_ply) if args.points_ply else prepared_root / args.scene / "points3d.ply"
    label_ply = Path(args.label_ply) if args.label_ply else _default_label_ply(prepared_root, args.scene)
    geometry_ply: Optional[Path]
    if args.geometry_ply:
        geometry_ply = Path(args.geometry_ply)
    else:
        geometry_ply = _default_geometry_ply(
            Path(args.geometry_root),
            args.scene,
            args.geom_tag,
            args.iters,
        )

    points_xyz = read_xyz_ply(points_ply)
    label_xyz = read_xyz_ply(label_ply)
    comparisons = [
        compare_xyz_rows(
            "points3d_vs_label",
            points_xyz,
            label_xyz,
            tolerance=args.tolerance,
        )
    ]

    geometry_exists = bool(geometry_ply.exists()) if geometry_ply is not None else False
    if geometry_ply is not None and geometry_exists:
        geometry_xyz = read_xyz_ply(geometry_ply)
        comparisons.append(
            compare_xyz_rows(
                "points3d_vs_geometry",
                points_xyz,
                geometry_xyz,
                tolerance=args.tolerance,
            )
        )

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene": args.scene,
        "paths": {
            "points_ply": str(points_ply),
            "label_ply": str(label_ply),
            "geometry_ply": str(geometry_ply) if geometry_ply is not None else "",
            "geometry_exists": geometry_exists,
        },
        "comparisons": comparisons,
        "all_row_aligned": bool(all(item["row_aligned"] for item in comparisons)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prepared_root", default="dataset/scannet_og")
    parser.add_argument("--points_ply", default="")
    parser.add_argument("--label_ply", default="")
    parser.add_argument("--geometry_ply", default="")
    parser.add_argument("--geometry_root", default="output/3dgs_models/scannet_og")
    parser.add_argument("--geom_tag", default="og_rgb_3dgs")
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output_json", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_diagnostics(args)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
