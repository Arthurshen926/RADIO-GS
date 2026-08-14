from __future__ import annotations

import numpy as np
import pytest

from radio_gs.benchmarks.scannet_uqis.ludvig_point3d_adapter import point_descriptor


def test_point_descriptor_selects_local_feature() -> None:
    features = np.zeros((2, 40), dtype=np.float32)
    features[0, 0] = 2.0
    features[1, 1] = 3.0
    xyz = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    covariance = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    opacity = np.ones(2, dtype=np.float32)

    descriptor, indices = point_descriptor(
        features, xyz, covariance, opacity, np.zeros(3, dtype=np.float32), candidate_k=1
    )

    assert indices.tolist() == [0]
    assert descriptor.dtype == np.float32
    assert descriptor[0] == pytest.approx(1.0)
    assert np.count_nonzero(descriptor) == 1


def test_point_descriptor_rejects_unsupported_prompt() -> None:
    with pytest.raises(ValueError, match="support"):
        point_descriptor(
            np.ones((1, 40), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32)[None],
            np.zeros(1, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            candidate_k=1,
        )
