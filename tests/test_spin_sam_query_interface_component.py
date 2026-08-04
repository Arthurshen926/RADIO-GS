import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_spin_sam_query_interface_component import (
    _exact_subset_mapping,
    _normalize_candidates,
)


def _void_keys(values: list[int]) -> np.ndarray:
    array = np.asarray(values, dtype="<i4")
    return array.view(np.dtype((np.void, 4))).reshape(-1)


def test_exact_subset_mapping_preserves_subset_order() -> None:
    mapping = _exact_subset_mapping(
        _void_keys([30, 10, 20]),
        _void_keys([20, 30]),
    )
    assert mapping.tolist() == [2, 0]


def test_exact_subset_mapping_rejects_missing_row() -> None:
    with pytest.raises(ValueError, match="not an exact carrier subset"):
        _exact_subset_mapping(_void_keys([10, 20]), _void_keys([30]))


def test_candidate_normalization_is_independent_per_channel() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[-4.0, -2.0], [0.0, 4.0]],
        ]
    )
    normalized = _normalize_candidates(values)
    assert torch.allclose(normalized.amin(dim=(1, 2)), torch.zeros(2))
    assert torch.allclose(normalized.amax(dim=(1, 2)), torch.ones(2))
