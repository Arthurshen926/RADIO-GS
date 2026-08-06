from __future__ import annotations

import numpy as np

from radio_gs.scripts.score_nvos_sealed_prediction_batch import _score_frame


def test_score_frame_matches_frozen_linear_resize_and_ge_threshold() -> None:
    score = np.asarray([[0.5, 0.0], [1.0, 0.5]], dtype=np.float32)
    ground_truth = np.asarray([[1, 0], [1, 1]], dtype=bool)
    metrics = _score_frame(score, ground_truth, 0.5)
    assert metrics == {"foreground_iou": 1.0, "pixel_accuracy": 1.0}


def test_score_frame_empty_union_is_one() -> None:
    metrics = _score_frame(
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((3, 3), dtype=bool),
        0.5,
    )
    assert metrics["foreground_iou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0
