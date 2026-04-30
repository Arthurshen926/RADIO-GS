import numpy as np
from plyfile import PlyData, PlyElement

from radio_gs.scripts.diagnose_scannet_row_alignment import (
    compare_xyz_rows,
    read_xyz_ply,
)


def _write_xyz_ply(path, xyz):
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"] = np.asarray(xyz, dtype=np.float32)[:, 0]
    arr["y"] = np.asarray(xyz, dtype=np.float32)[:, 1]
    arr["z"] = np.asarray(xyz, dtype=np.float32)[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def test_compare_xyz_rows_accepts_row_aligned_points(tmp_path) -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "points.ply"
    _write_xyz_ply(path, points)

    loaded = read_xyz_ply(path)
    stats = compare_xyz_rows("points_vs_label", loaded, points.copy(), tolerance=1e-6)

    assert stats["row_aligned"] is True
    assert stats["count_match"] is True
    assert stats["max_distance"] == 0.0


def test_compare_xyz_rows_flags_same_count_permuted_rows() -> None:
    left = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    right = left[::-1].copy()

    stats = compare_xyz_rows("points_vs_label", left, right, tolerance=1e-6)

    assert stats["count_match"] is True
    assert stats["row_aligned"] is False
    assert stats["max_distance"] > 0.0
