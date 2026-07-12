from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from radio_gs.scripts.train_colmap_gs import _load_colmap_points_binary


def _write_points3d(path: Path) -> None:
    records = [
        # point id, xyz, rgb, reprojection error, [(image id, point2D index)]
        (1, (1.0, 2.0, 3.0), (10, 20, 30), 0.1, [(1, 0), (2, 1)]),
        (2, (4.0, 5.0, 6.0), (40, 50, 60), 0.2, [(1, 2)]),
        (3, (7.0, 8.0, 9.0), (70, 80, 90), 0.3, [(3, 3)]),
    ]
    payload = bytearray(struct.pack("<Q", len(records)))
    for point_id, xyz, rgb, error, track in records:
        payload.extend(struct.pack("<Qddd", point_id, *xyz))
        payload.extend(struct.pack("<BBB", *rgb))
        payload.extend(struct.pack("<d", error))
        payload.extend(struct.pack("<Q", len(track)))
        for image_id, point2d_index in track:
            payload.extend(struct.pack("<II", image_id, point2d_index))
    path.write_bytes(payload)


def test_points_observed_by_excluded_image_are_removed(tmp_path: Path) -> None:
    source = tmp_path / "points3D.bin"
    _write_points3d(source)

    xyz, rgb, metadata = _load_colmap_points_binary(
        source,
        excluded_image_ids=frozenset({2}),
        return_metadata=True,
    )

    np.testing.assert_allclose(xyz, [[4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(rgb * 255.0, [[40, 50, 60], [70, 80, 90]])
    assert metadata["source_point_count"] == 3
    assert metadata["retained_point_count"] == 2
    assert metadata["removed_point_count"] == 1
    assert metadata["excluded_image_ids"] == [2]
    assert metadata["track_filter"] == "drop_any_point_observed_by_excluded_image"


def test_unfiltered_binary_loader_is_backward_compatible(tmp_path: Path) -> None:
    source = tmp_path / "points3D.bin"
    _write_points3d(source)

    xyz, rgb = _load_colmap_points_binary(source)

    assert xyz.shape == (3, 3)
    assert rgb.shape == (3, 3)

