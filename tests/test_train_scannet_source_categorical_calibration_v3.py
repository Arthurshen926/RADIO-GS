from __future__ import annotations

import torch

from radio_gs.scripts.train_scannet_source_categorical_calibration_v3 import (
    calibrated_logits,
    weighted_miou,
)


def test_diagonal_calibration_preserves_rows_and_can_correct_class_bias() -> None:
    scores = torch.tensor([[0.2, 0.3], [0.4, 0.1]])
    output = calibrated_logits(scores, torch.zeros(2), torch.tensor([0.2, -0.2]))
    assert output.shape == scores.shape
    assert output.argmax(1).tolist() == [0, 0]
    assert weighted_miou(output.argmax(1), torch.tensor([0, 0]), torch.ones(2), 2) == 1.0
