from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from radio_gs.scripts.render_feature_rgb_ply_views import (
    parse_views,
    read_colored_ply,
    subsample_points,
)


def _write_colored_ply(path: Path) -> None:
    arr = np.empty(
        5,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    arr["x"] = np.arange(5, dtype=np.float32)
    arr["y"] = 1.0
    arr["z"] = 2.0
    arr["red"] = [0, 64, 128, 192, 255]
    arr["green"] = 0
    arr["blue"] = 255
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_read_colored_ply_scales_rgb(tmp_path):
    path = tmp_path / "feature_rgb.ply"
    _write_colored_ply(path)

    xyz, rgb = read_colored_ply(path)

    assert xyz.shape == (5, 3)
    assert rgb.shape == (5, 3)
    assert np.isclose(rgb[0, 0], 0.0)
    assert np.isclose(rgb[-1, 0], 1.0)


def test_parse_views_supports_custom_entries():
    views = parse_views("paper:20:-35,top:90:0")

    assert views == {"paper": (20.0, -35.0), "top": (90.0, 0.0)}


def test_subsample_points_is_deterministic():
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    rgb = np.ones((10, 3), dtype=np.float32)

    xyz_a, rgb_a = subsample_points(xyz, rgb, max_points=4, seed=3)
    xyz_b, rgb_b = subsample_points(xyz, rgb, max_points=4, seed=3)

    assert np.array_equal(xyz_a, xyz_b)
    assert np.array_equal(rgb_a, rgb_b)
