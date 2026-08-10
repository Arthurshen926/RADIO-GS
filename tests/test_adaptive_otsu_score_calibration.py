from __future__ import annotations

import torch

from radio_gs.querying.adaptive_otsu_score_calibration import (
    FROZEN_THRESHOLD,
    calibrate_to_frozen_threshold,
)


def test_calibration_maps_otsu_membership_to_frozen_threshold() -> None:
    scores = torch.tensor(
        [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8], [1.0, 0.9]]
    )
    valid = torch.ones(6, dtype=torch.bool)
    result = calibrate_to_frozen_threshold(scores, valid, stages=1)
    expected = scores > result.source_thresholds.unsqueeze(0)
    assert torch.equal(result.scores > FROZEN_THRESHOLD, expected)
    assert torch.equal(result.selected_counts, expected.sum(0))
    assert torch.allclose(
        result.scores[torch.arange(6), torch.zeros(6, dtype=torch.long)],
        torch.sort(result.scores[:, 0]).values,
    )


def test_calibration_keeps_invalid_rows_zero() -> None:
    scores = torch.tensor([[0.0], [0.2], [0.8], [1.0]])
    valid = torch.tensor([True, True, True, False])
    result = calibrate_to_frozen_threshold(scores, valid, stages=1)
    assert result.scores[-1, 0] == 0.0
    assert torch.isfinite(result.scores).all()
    assert bool((result.scores >= 0).all())
    assert bool((result.scores <= 1).all())
