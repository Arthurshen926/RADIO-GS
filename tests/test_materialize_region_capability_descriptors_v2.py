import pytest
import torch

from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    pool_region_capability,
)


def test_pool_region_capability_separates_direction_and_concentration():
    features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    direction, concentration = pool_region_capability(
        compact_features=features,
        active_global_rows=torch.tensor([0, 2, 3]),
        region_rows=torch.tensor([[0, 2], [2, 3]]),
        token_mask=torch.ones(2, 2, dtype=torch.bool),
        batch_size=1,
    )
    torch.testing.assert_close(direction[0].float(), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        direction[1].float(),
        torch.tensor([2**-0.5, 2**-0.5]),
        atol=5e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(concentration, torch.tensor([1.0, 2**-0.5]))
    assert direction.dtype == torch.float16
    assert concentration.dtype == torch.float32


def test_pool_region_capability_ignores_padded_rows():
    direction, concentration = pool_region_capability(
        compact_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        active_global_rows=torch.tensor([1, 3]),
        region_rows=torch.tensor([[3, -1]]),
        token_mask=torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(direction.float(), torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(concentration, torch.ones(1))


def test_pool_region_capability_rejects_noncapability_region_row():
    with pytest.raises(ValueError, match="outside"):
        pool_region_capability(
            compact_features=torch.eye(2),
            active_global_rows=torch.tensor([0, 2]),
            region_rows=torch.tensor([[0, 1]]),
            token_mask=torch.ones(1, 2, dtype=torch.bool),
        )
