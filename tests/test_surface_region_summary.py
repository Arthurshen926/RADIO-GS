import torch

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadout,
    surface_region_geometry,
)


def _inputs():
    torch.manual_seed(3)
    features = torch.randn(2, 9, 16)
    xyz = torch.randn(2, 9, 3)
    scales = torch.rand(2, 9, 3) * 0.04 + 0.01
    opacity = torch.rand(2, 9, 1)
    reliability = torch.rand(2, 9, 1) * 0.8 + 0.2
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 7:] = False
    geometry = surface_region_geometry(
        xyz, scales, opacity, reliability, torch.tensor([0.3, 0.6]), token_mask=mask
    )
    return features, geometry, reliability, mask


def test_surface_readout_is_permutation_invariant() -> None:
    features, geometry, reliability, mask = _inputs()
    model = SurfaceRegionSummaryReadout(feature_dim=16, hidden_dim=8).eval()
    expected = model(features, geometry, token_mask=mask, reliability=reliability)
    order = torch.tensor([7, 1, 4, 0, 8, 2, 6, 5, 3])
    actual = model(
        features[:, order], geometry[:, order], token_mask=mask[:, order],
        reliability=reliability[:, order],
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_surface_geometry_is_translation_invariant_and_masks_padding() -> None:
    features, geometry, reliability, mask = _inputs()
    torch.manual_seed(3)
    xyz = torch.randn(2, 9, 3)
    scales = torch.rand(2, 9, 3) * 0.04 + 0.01
    opacity = torch.rand(2, 9, 1)
    shifted = surface_region_geometry(
        xyz + torch.tensor([19.0, -4.0, 7.0]), scales, opacity, reliability,
        torch.tensor([0.3, 0.6]), token_mask=mask,
    )
    original = surface_region_geometry(
        xyz, scales, opacity, reliability, torch.tensor([0.3, 0.6]), token_mask=mask
    )
    torch.testing.assert_close(shifted, original, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(geometry[1, 7:]) == 0


def test_zero_initialized_readout_is_reliability_weighted_raw_mean() -> None:
    features, geometry, reliability, mask = _inputs()
    model = SurfaceRegionSummaryReadout(feature_dim=16, hidden_dim=8).eval()
    output = model(features, geometry, token_mask=mask, reliability=reliability)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
