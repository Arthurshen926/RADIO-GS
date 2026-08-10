from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.audit_region_comembership_v2_scene1_to_scene2 import (
    capability_pair_features,
)


def test_capability_pair_features_have_registered_values_and_order() -> None:
    pairs = torch.tensor([[0, 0], [1, 2]])
    appearance = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    boundary = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    appearance_c = torch.tensor([0.9, 0.4, 0.7])
    boundary_c = torch.tensor([0.8, 0.5, 0.2])
    values = capability_pair_features(
        pair_indices=pairs,
        appearance_direction=appearance,
        boundary_direction=boundary,
        appearance_concentration=appearance_c,
        boundary_concentration=boundary_c,
        chunk_size=1,
    )
    torch.testing.assert_close(
        values,
        torch.tensor(
            [
                [1.0, 0.0, 0.4, 0.5, 0.5, 0.3],
                [0.0, 1.0, 0.7, 0.2, 0.2, 0.6],
            ]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_capability_pair_features_reject_noncanonical_pair_order() -> None:
    with pytest.raises(ValueError, match="inputs differ"):
        capability_pair_features(
            pair_indices=torch.tensor([[1], [0]]),
            appearance_direction=torch.eye(2),
            boundary_direction=torch.eye(2),
            appearance_concentration=torch.ones(2),
            boundary_concentration=torch.ones(2),
        )
