from __future__ import annotations

import torch

from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.querying.source_post_spatial_text_posterior import (
    aggregate_region_reliability,
    build_post_spatial_channels,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import vala_knn_minmax_scores


def _state() -> FactorizedPrimitiveState:
    return FactorizedPrimitiveState(
        xyz=torch.zeros(4, 3),
        valid=torch.tensor([True, True, False, True]),
        global_rows=torch.tensor([0, 1, 3]),
        semantic_direction=torch.ones(3, 1280, dtype=torch.float16),
        predicted_log_amplitude=torch.zeros(3),
        directional_dispersion=torch.tensor([0.1, 0.3, 0.5]),
        log_amplitude_std=torch.tensor([0.2, 0.4, 0.6]),
        observation_evidence=torch.tensor([0.9, 0.7, 0.5]),
        visibility_purity_value=torch.tensor([0.8, 0.6, 0.4]),
        visibility_purity_known=torch.ones(3, dtype=torch.bool),
        metadata={},
    )


def test_region_reliability_uses_exact_member_mean_and_zeros_invalid() -> None:
    accepted = {
        "region_rows": torch.tensor([[0, 1], [1, 2], [2, 3]]),
        "token_mask": torch.ones(3, 2, dtype=torch.bool),
    }
    result = aggregate_region_reliability(
        _state(), accepted, region_valid=torch.tensor([True, True, False])
    )
    assert torch.allclose(
        result[0], torch.tensor([0.8, 0.2, 0.3, 0.8, 0.7])
    )
    assert torch.allclose(
        result[1], torch.tensor([0.7, 0.3, 0.4, 0.7, 0.6])
    )
    assert torch.equal(result[2], torch.zeros(5))


def test_source_post_spatial_base_matches_frozen_vala_implementation() -> None:
    positive_affinity = torch.tensor(
        [[[0.51, 0.55]], [[0.53, 0.54]], [[0.58, 0.52]], [[0.60, 0.51]]]
    )
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    neighbors = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 3, 0], [3, 2, 1, 0]]
    )
    valid = torch.ones(4, dtype=torch.bool)
    base, raw, extent = build_post_spatial_channels(
        positive_affinity, neighbors, xyz, valid=valid
    )
    expected = vala_knn_minmax_scores(raw, xyz, k=10, valid_mask=valid)
    assert torch.equal(base, expected)
    assert extent.shape == (4, 2, 4)
