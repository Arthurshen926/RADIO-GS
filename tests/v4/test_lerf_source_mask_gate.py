import pytest
import torch

from radio_gs.v4.evaluation.lerf_source_mask_gate import _mutually_exclusive_purity_fast
from radio_gs.v4.evaluation.source_mask_ladder import _mutually_exclusive_purity


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_vectorized_purity_matches_reference(device):
    masks = torch.tensor([
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 0], [0, 1]],
    ]).float()
    posterior = torch.tensor([
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.2, 0.0, 0.8, 0.0],
    ])
    expected, expected_pairs = _mutually_exclusive_purity(posterior, masks)
    actual, actual_pairs = _mutually_exclusive_purity_fast(posterior.to(device), masks, pair_chunk_size=2)
    assert actual_pairs == expected_pairs
    assert actual == pytest.approx(expected)
