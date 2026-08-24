from __future__ import annotations

import numpy as np

from radio_gs.scripts.predict_nvos_sam3_video_from_registered_prompt import (
    sample_signed_points,
    shorter_cyclic_path,
)


def test_shorter_cyclic_path_uses_wrap_and_is_deterministic_on_tie():
    assert shorter_cyclic_path(20, 12, 1) == [12, 13, 14, 15, 16, 17, 18, 19, 0, 1]
    assert shorter_cyclic_path(6, 1, 4) == [1, 2, 3, 4]


def test_signed_point_sampling_preserves_both_labels_and_normalizes_xy():
    positive = np.zeros((5, 7), dtype=np.float32)
    negative = np.zeros_like(positive)
    positive[1, 1] = 1
    positive[4, 6] = 1
    negative[0, 6] = 1
    points, labels = sample_signed_points(positive, negative, maximum_per_sign=4)
    assert labels.tolist() == [1, 1, 0]
    assert np.all((0 <= points) & (points <= 1))
    assert np.allclose(points[0], [1 / 6, 1 / 4])
