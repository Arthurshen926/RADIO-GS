from __future__ import annotations

import torch

from radio_gs.scripts.eval_scannet_source_calibrated_paper8 import (
    calibrated_predictions,
)


def test_calibrated_predictions_restricts_after_shared_19_class_transform() -> None:
    scores = torch.zeros(1, 19)
    scores[0, 0] = 1.0
    scores[0, 2] = 0.9
    prediction = calibrated_predictions(scores, torch.ones(19), torch.zeros(19))
    assert int(prediction["19"][0]) == 1
    assert int(prediction["15"][0]) == 1
    assert int(prediction["10"][0]) == 1
