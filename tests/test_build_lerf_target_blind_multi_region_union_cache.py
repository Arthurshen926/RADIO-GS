from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.build_lerf_target_blind_multi_region_union_cache import (
    _scale_major_anchor_probability,
)


def test_scale_major_probability_preserves_each_candidate_scale() -> None:
    probability = torch.tensor(
        [
            [[0.10, 0.11], [0.20, 0.21], [0.30, 0.31]],
            [[0.40, 0.41], [0.50, 0.51], [0.60, 0.61]],
            [[0.70, 0.71], [0.80, 0.81], [0.90, 0.91]],
        ],
        dtype=torch.float32,
    )
    rows = torch.tensor([2, 0], dtype=torch.long)

    aligned = _scale_major_anchor_probability(probability, rows)

    assert torch.equal(
        aligned,
        torch.stack(
            [
                probability[2, 0],
                probability[0, 0],
                probability[2, 1],
                probability[0, 1],
                probability[2, 2],
                probability[0, 2],
            ]
        ),
    )
    assert not torch.equal(aligned[:2], aligned[2:4])
    assert not torch.equal(aligned[:2], aligned[4:6])


def test_scale_major_probability_rejects_wrong_axes() -> None:
    with pytest.raises(ValueError, match=r"\[N,3,Q\]"):
        _scale_major_anchor_probability(torch.ones(3, 2, 1), torch.tensor([0]))
    with pytest.raises(ValueError, match="out of probability range"):
        _scale_major_anchor_probability(torch.ones(3, 3, 1), torch.tensor([3]))
