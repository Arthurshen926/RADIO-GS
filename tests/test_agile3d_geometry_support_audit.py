import numpy as np
import pytest
import torch

from radio_gs.benchmarks.agile3d_scannet40.audit_geometry_support import (
    all_gaussian_support_fraction,
)


def test_all_gaussian_support_ceiling_is_geometry_only_and_meaningful() -> None:
    fraction, quantiles = all_gaussian_support_fraction(
        gaussian_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        gaussian_covariance=torch.eye(3).unsqueeze(0) * 0.0025,
        gaussian_opacity=torch.tensor([1.0]),
        official_xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        candidate_k=1,
        support_threshold=0.01,
        evaluation_voxel_size_m=0.05,
    )

    assert fraction == pytest.approx(0.5)
    assert quantiles["p00"] < 0.01
    assert quantiles["p100"] > 0.9


def test_all_gaussian_support_ceiling_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="candidate_k/support_threshold"):
        all_gaussian_support_fraction(
            gaussian_xyz=torch.zeros((1, 3)),
            gaussian_covariance=torch.eye(3).unsqueeze(0),
            gaussian_opacity=torch.ones(1),
            official_xyz=np.zeros((1, 3), dtype=np.float32),
            candidate_k=1,
            support_threshold=-1.0,
            evaluation_voxel_size_m=0.05,
        )
