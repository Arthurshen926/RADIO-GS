from __future__ import annotations

import numpy as np

from radio_gs.scripts.audit_scannet_semantic_support_geometry import (
    support_statistics,
)


def test_semantic_support_detects_boundary_mixing() -> None:
    mesh_xyz = np.asarray([[-0.01, 0, 0], [0.01, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    mesh_labels = np.asarray([1, 2, 1], dtype=np.int64)
    result = support_statistics(
        gaussian_xyz=np.asarray([[0, 0, 0], [0.5, 0, 0]], dtype=np.float32),
        gaussian_covariance=np.asarray([np.eye(3) * 0.01**2, np.eye(3) * 0.01**2]),
        gaussian_opacity=np.ones(2, dtype=np.float32),
        mesh_xyz=mesh_xyz,
        mesh_labels=mesh_labels,
        class_ids=[1, 2],
        neighbors=2,
        chunk_size=1,
    )
    assert float(result["boundary_ambiguity"][0]) > 0.45
    assert float(result["boundary_ambiguity"][1]) < 0.01
    assert int(result["dominant_nyu40_id"][1]) == 1


def test_unsupported_gaussian_is_not_misreported_as_boundary() -> None:
    result = support_statistics(
        gaussian_xyz=np.asarray([[2, 0, 0]], dtype=np.float32),
        gaussian_covariance=np.asarray([np.eye(3) * 0.01**2]),
        gaussian_opacity=np.ones(1, dtype=np.float32),
        mesh_xyz=np.asarray([[0, 0, 0]], dtype=np.float32),
        mesh_labels=np.asarray([1], dtype=np.int64),
        class_ids=[1],
        neighbors=1,
    )
    assert not bool(result["support_valid"][0])
    assert float(result["boundary_ambiguity"][0]) == 0.0
    assert float(result["joint_risk"][0]) == 0.0
