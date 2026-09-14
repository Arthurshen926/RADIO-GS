from __future__ import annotations

import pytest
import torch

from radio_gs.v4.contracts.build_lerf_object_completion import (
    merge_completion_into_unobserved_rows,
)


def test_completion_fills_only_exactly_unobserved_rows():
    observed = torch.tensor(
        [[0.8, 0.0, 0.0], [0.4, 0.3, 0.0], [0.0, 0.0, 0.0], [1e-4, 0.0, 0.0]]
    )
    active = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.6, 0.1], [0.7, 0.2]])
    completed, writable = merge_completion_into_unobserved_rows(
        observed, active, torch.tensor([0, 2])
    )
    assert writable.tolist() == [False, False, True, False]
    assert torch.equal(completed[0], observed[0])
    assert torch.equal(completed[1], observed[1])
    assert torch.equal(completed[3], observed[3])
    assert torch.allclose(completed[2], torch.tensor([0.6, 0.0, 0.1]))


def test_completion_rejects_duplicate_active_token_ids():
    with pytest.raises(ValueError, match="unique"):
        merge_completion_into_unobserved_rows(
            torch.zeros(2, 3), torch.zeros(2, 2), torch.tensor([1, 1])
        )
