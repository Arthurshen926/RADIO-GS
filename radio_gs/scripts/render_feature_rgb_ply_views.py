#!/usr/bin/env python3
"""Render fixed-RGB feature PLY files to simple multi-view PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


DEFAULT_VIEWS = {
    "iso": (28.0, -45.0),
    "front": (8.0, -90.0),
    "side": (8.0, 0.0),
    "top": (88.0, -90.0),
}


def read_colored_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    required = ("x", "y", "z", "red", "green", "blue")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PLY is missing required properties {missing}: {path}")
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    rgb = np.stack(
        [
            np.asarray(vertex["red"], dtype=np.float32),
            np.asarray(vertex["green"], dtype=np.float32),
            np.asarray(vertex["blue"], dtype=np.float32),
        ],
        axis=1,
    )
    return xyz, np.clip(rgb / 255.0, 0.0, 1.0)


def subsample_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
    return xyz[idx], rgb[idx]


def set_equal_axes(ax, xyz: np.ndarray) -> None:
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    radius = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)) * 0.55)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def render_view(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: Path,
    *,
    elev: float,
    azim: float,
    point_size: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 7.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb,
        s=point_size,
        linewidths=0,
        depthshade=False,
    )
    set_equal_axes(ax, xyz)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, transparent=False, facecolor="white")
    plt.close(fig)


def parse_views(raw: str) -> dict[str, tuple[float, float]]:
    if not raw or raw == "default":
        return dict(DEFAULT_VIEWS)
    views: dict[str, tuple[float, float]] = {}
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "--views entries must be name:elev:azim, for example iso:28:-45"
            )
        views[parts[0]] = (float(parts[1]), float(parts[2]))
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, help="Input colored PLY path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_points", type=int, default=120000)
    parser.add_argument("--point_size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument(
        "--views",
        default="default",
        help="Comma-separated name:elev:azim entries, or 'default'",
    )
    args = parser.parse_args()

    xyz, rgb = read_colored_ply(args.ply)
    xyz, rgb = subsample_points(xyz, rgb, args.max_points, args.seed)
    output_dir = Path(args.output_dir)
    for name, (elev, azim) in parse_views(args.views).items():
        render_view(
            xyz,
            rgb,
            output_dir / f"{Path(args.ply).stem}_{name}.png",
            elev=elev,
            azim=azim,
            point_size=args.point_size,
            dpi=args.dpi,
        )
    print(f"Rendered {len(parse_views(args.views))} views to {output_dir}")


if __name__ == "__main__":
    main()
