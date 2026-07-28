import pytest
import torch

from radio_gs.scripts.complete_canonical_field_support import (
    initialize_completion_from_primary_codes,
    initialize_completion_local_codes,
    support_completion_rows,
)
from radio_gs.field import AffineBasisDecoder, CanonicalGaussianField
from radio_gs.field.field_signature import FeatureSpaceSignature


def test_support_completion_selects_only_newly_observed_rows() -> None:
    rows = support_completion_rows(
        torch.tensor([True, False, True, False]),
        torch.tensor([True, True, True, False]),
    )
    assert rows.tolist() == [1]


def test_support_completion_rejects_dropped_primary_rows() -> None:
    with pytest.raises(ValueError, match="cannot drop primary"):
        support_completion_rows(
            torch.tensor([True, True]), torch.tensor([True, False])
        )


def test_completion_initialization_inverts_compact_base_projection() -> None:
    decoder = AffineBasisDecoder(
        feature_dim=5,
        coefficient_dim=5,
        mean=torch.zeros(5),
        scale=torch.ones(5),
        basis=torch.eye(5),
    )
    field = CanonicalGaussianField(
        num_gaussians=3,
        decoder=decoder,
        signature=FeatureSpaceSignature(
            radio_version="test",
            radio_checkpoint_sha256="test",
            raw_feature_dim=5,
        ),
        local_dim=3,
        use_fusion=True,
        reliability=None,
    )
    with torch.no_grad():
        weight = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0],
             [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
        )
        field.fusion.base_projection.weight.copy_(weight)
        field.fusion.base_projection.bias.zero_()
        field.fusion.network[-1].weight.zero_()
        field.fusion.network[-1].bias.zero_()
        primary_before = field.local_codes[0].clone()
    desired_local = torch.tensor([[0.25, -0.5, 0.75]])
    target = desired_local @ weight.T
    mode = initialize_completion_local_codes(field, target, torch.tensor([1]))

    assert mode == "least_squares_base_projection_inverse"
    assert torch.allclose(field.radio_features(torch.tensor([1])), target, atol=1e-5)
    assert torch.equal(field.local_codes[0], primary_before)


def test_completion_initialization_stays_on_primary_code_manifold() -> None:
    decoder = AffineBasisDecoder(
        feature_dim=3,
        coefficient_dim=3,
        mean=torch.zeros(3),
        scale=torch.ones(3),
        basis=torch.eye(3),
    )
    field = CanonicalGaussianField(
        num_gaussians=4,
        decoder=decoder,
        signature=FeatureSpaceSignature(
            radio_version="test",
            radio_checkpoint_sha256="test",
            raw_feature_dim=3,
        ),
        local_dim=3,
        use_fusion=True,
        reliability=None,
    )
    with torch.no_grad():
        field.local_codes.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]
            )
        )
        primary_before = field.local_codes[:2].clone()
    mode = initialize_completion_from_primary_codes(
        field,
        xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
        ),
        target_features=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
        ),
        fallback_rows=torch.tensor([2, 3]),
        primary_rows=torch.tensor([0, 1]),
        spatial_neighbors=2,
    )

    assert mode == "spatial_knn2_target_affinity_primary_code"
    assert torch.equal(field.local_codes[2], primary_before[0])
    assert torch.equal(field.local_codes[3], primary_before[1])
    assert torch.equal(field.local_codes[:2], primary_before)
