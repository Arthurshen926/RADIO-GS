import numpy as np
import pytest

from radio_gs.scripts.fuse_nvos_box_point_consensus import (
    fuse_box_and_point_consensus,
)


def test_box_is_primary_and_point_supermajority_only_adds_support():
    box = np.array([[0.5, -0.5, -0.5]], dtype=np.float32)
    point = np.array([[-0.5, 0.2, 0.1]], dtype=np.float32)

    fused = fuse_box_and_point_consensus(box, point, point_margin_threshold=0.2)

    assert fused.tolist() == [[0.5, 0.5, -0.5]]


def test_box_point_consensus_requires_aligned_finite_arrays():
    with pytest.raises(ValueError, match="aligned"):
        fuse_box_and_point_consensus(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        fuse_box_and_point_consensus(
            np.array([[np.nan]], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
