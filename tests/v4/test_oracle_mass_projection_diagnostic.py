from __future__ import annotations

import pytest
import torch

from radio_gs.v4.training.diagnose_scannet_oracle_mass_projection import (
    oracle_mass_project,
)


def test_oracle_mass_projection_matches_pre_cap_support_and_preserves_clamp():
    unary = torch.tensor(
        [
            [1.0, 0.0],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    clamp_mask = torch.tensor([True, False, False, False])
    clamp = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    projected, audit = oracle_mass_project(
        unary,
        clamp_mask,
        clamp,
        torch.tensor([2.5]),
        iteration_count=256,
    )
    assert torch.equal(projected[clamp_mask], clamp[clamp_mask])
    torch.testing.assert_close(projected.sum(-1), torch.ones(4), atol=1e-6, rtol=0)
    assert float(projected[:, 0].sum()) == pytest.approx(2.5, abs=2e-5)
    assert audit["maximum_relative_mass_error"] < 1e-5
