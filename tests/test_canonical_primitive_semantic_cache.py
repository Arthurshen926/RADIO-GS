import numpy as np
import pytest
import torch

from radio_gs.scripts.build_canonical_primitive_semantic_cache import (
    build_region_token_mask,
    canonical_reconstruction_confidence,
    query_region_neighbor_rows,
)


def test_valid_neighbor_domain_has_fixed_observed_cardinality() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    valid_rows = torch.tensor([0, 3])

    all_neighbors = query_region_neighbor_rows(
        xyz, valid_rows, 2, domain="all", workers=1
    )
    valid_neighbors = query_region_neighbor_rows(
        xyz, valid_rows, 2, domain="valid", workers=1
    )

    assert all_neighbors.shape == valid_neighbors.shape == (2, 2)
    assert 1 in all_neighbors[0]  # nearest geometry row has no teacher target
    assert set(valid_neighbors[0].tolist()) == {0, 3}
    assert np.isin(valid_neighbors, valid_rows.numpy()).all()


def test_valid_neighbor_domain_caps_k_at_number_of_observed_rows() -> None:
    xyz = torch.eye(3)
    neighbors = query_region_neighbor_rows(
        xyz, torch.tensor([1]), 8, domain="valid", workers=1
    )
    assert neighbors.tolist() == [[1]]


def test_primary_plus_center_preserves_primary_context_and_isolates_fallbacks() -> None:
    neighbors = torch.tensor([[0, 1, 2], [1, 2, 3]])
    valid = torch.tensor([True, True, True, True])
    primary = torch.tensor([True, False, True, False])

    mask = build_region_token_mask(
        neighbors,
        valid,
        torch.tensor([0, 1]),
        policy="primary_plus_center",
        primary_valid=primary,
    )

    assert mask.tolist() == [[True, False, True], [True, True, False]]


def test_all_valid_region_token_policy_matches_legacy_mask() -> None:
    neighbors = torch.tensor([[0, 1], [2, 1]])
    valid = torch.tensor([True, False, True])
    mask = build_region_token_mask(
        neighbors,
        valid,
        torch.tensor([0, 2]),
        policy="all_valid",
    )
    assert mask.tolist() == [[True, False], [True, False]]


def test_reconstruction_confidence_preserves_primary_and_scores_fallback() -> None:
    predicted = torch.tensor([[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    confidence = canonical_reconstruction_confidence(
        predicted,
        teacher,
        valid=torch.tensor([True, True, False]),
        primary_valid=torch.tensor([True, False, False]),
        observation_counts=torch.tensor([1, 2, 0]),
    )

    assert confidence[0] == 1.0  # frozen primary is never weakened
    assert confidence[1] == pytest.approx((2 ** -0.5) * (2.0 / 3.0))
    assert confidence[2] == 0.0
