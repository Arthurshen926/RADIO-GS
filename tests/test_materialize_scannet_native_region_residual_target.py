from __future__ import annotations

import torch

from radio_gs.scripts.materialize_scannet_native_region_residual_target import (
    aggregate_sparse_view_descriptors,
)


def test_equal_view_region_aggregation_and_agreement_gate():
    descriptors, valid, count, agreement = aggregate_sparse_view_descriptors(
        row_indices=torch.tensor([0, 0, 0, 1]),
        proposal_indices=torch.tensor([0, 1, 2, 2]),
        weights=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        proposal_views=torch.tensor([0, 0, 1]),
        proposal_descriptors=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6]]
        ),
        num_rows=2,
        minimum_views=2,
        minimum_view_cosine=0.5,
    )
    assert count.tolist() == [2, 1]
    assert valid.tolist() == [True, False]
    assert torch.allclose(agreement, torch.tensor([0.8, 1.0]))
    assert torch.allclose(descriptors[0].norm(), torch.tensor(1.0))


def test_disagreeing_views_fail_without_turning_absence_negative():
    _descriptor, valid, count, agreement = aggregate_sparse_view_descriptors(
        row_indices=torch.tensor([0, 0]),
        proposal_indices=torch.tensor([0, 1]),
        weights=torch.ones(2),
        proposal_views=torch.tensor([0, 1]),
        proposal_descriptors=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        num_rows=2,
        minimum_views=2,
        minimum_view_cosine=0.5,
    )
    assert count.tolist() == [2, 0]
    assert agreement[0] == -1
    assert valid.tolist() == [False, False]
