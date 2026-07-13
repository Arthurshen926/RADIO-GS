import pytest
import torch

from radio_gs.scripts.build_dense_text_score_maps import (
    normalize_query_heatmaps_by_peak,
)


def test_normalize_query_heatmaps_by_peak_is_per_query():
    scores = torch.tensor(
        [
            [[0.1, 0.4], [0.2, 0.3]],
            [[0.5, 0.25], [0.0, 0.1]],
        ]
    )

    normalized = normalize_query_heatmaps_by_peak(scores)

    assert normalized[0, 0, 1] == pytest.approx(1.0)
    assert normalized[0, 0, 0] == pytest.approx(0.25)
    assert normalized[1, 0, 0] == pytest.approx(1.0)
    assert normalized[1, 0, 1] == pytest.approx(0.5)
