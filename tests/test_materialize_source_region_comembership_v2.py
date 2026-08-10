from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.materialize_source_region_comembership_v2 import (
    densify_primitive_instance_mass,
    sparsify_primitive_instance_mass,
)


def test_sparse_primitive_instance_mass_round_trip_is_exact() -> None:
    dense = torch.tensor(
        [[0.0, 1.5, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.25]]
    )
    keys, mass = sparsify_primitive_instance_mass(dense)
    assert keys.tolist() == [1, 3, 5, 8]
    torch.testing.assert_close(
        densify_primitive_instance_mass(
            flat_keys=keys,
            mass=mass,
            primitive_count=3,
            instance_columns_including_zero=3,
        ),
        dense,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "keys,mass",
    [
        (torch.tensor([1, 1]), torch.ones(2)),
        (torch.tensor([2, 1]), torch.ones(2)),
        (torch.tensor([99]), torch.ones(1)),
        (torch.tensor([1]), torch.tensor([0.0])),
    ],
)
def test_dense_reconstruction_rejects_noncanonical_sparse_authority(
    keys: torch.Tensor, mass: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="sparse"):
        densify_primitive_instance_mass(
            flat_keys=keys,
            mass=mass,
            primitive_count=2,
            instance_columns_including_zero=3,
        )


def test_sparsify_rejects_negative_or_empty_mass() -> None:
    with pytest.raises(ValueError, match="dense"):
        sparsify_primitive_instance_mass(torch.tensor([[0.0, -1.0]]))
    with pytest.raises(ValueError, match="empty"):
        sparsify_primitive_instance_mass(torch.zeros(2, 2))
