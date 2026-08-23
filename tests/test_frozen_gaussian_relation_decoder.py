import torch

from radio_gs.models.frozen_gaussian_relation_decoder import (
    FrozenGaussianRelationDecoder,
)


def test_gaussian_relation_is_symmetric_and_finite():
    torch.manual_seed(3)
    model = FrozenGaussianRelationDecoder(latent_dim=8, hidden_dim=4)
    left = torch.randn(5, 8)
    right = torch.randn(5, 8)
    left_xyz = torch.randn(5, 3)
    right_xyz = torch.randn(5, 3)
    extent = torch.tensor([2.0, 3.0, 4.0])
    forward = model(left, right, left_xyz, right_xyz, extent)
    reverse = model(right, left, right_xyz, left_xyz, extent)
    assert torch.allclose(forward, reverse, atol=1e-6, rtol=1e-6)
    assert bool(torch.isfinite(forward).all())


def test_gaussian_relation_rejects_misaligned_pairs():
    model = FrozenGaussianRelationDecoder(latent_dim=8, hidden_dim=4)
    try:
        model(
            torch.randn(2, 8),
            torch.randn(3, 8),
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.ones(3),
        )
    except ValueError as error:
        assert "axes differ" in str(error)
    else:
        raise AssertionError("misaligned relation pairs must fail closed")
