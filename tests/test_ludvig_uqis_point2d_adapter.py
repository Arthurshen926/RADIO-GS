from __future__ import annotations

import numpy as np
import pytest

from radio_gs.benchmarks.scannet_uqis.ludvig_point2d_adapter import projection_matrix_from_intrinsics


def test_off_center_projection_maps_camera_axis_to_principal_point() -> None:
    width, height = 1296, 968
    intrinsic = np.array([[1163.45, 0, 653.626], [0, 1164.79, 481.6], [0, 0, 1]])
    projection = projection_matrix_from_intrinsics(intrinsic, width, height)
    camera_point = np.array([0, 0, 2, 1], dtype=np.float32)
    clip = projection @ camera_point
    ndc = clip[:2] / clip[3]
    pixel = ((ndc + 1) * np.array([width, height]) - 1) / 2
    assert pixel == pytest.approx(intrinsic[:2, 2], abs=1e-4)
