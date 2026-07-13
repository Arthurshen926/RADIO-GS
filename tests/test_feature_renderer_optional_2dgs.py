import pytest
import torch

import radio_gs.rendering.feature_renderer as feature_renderer


def _make_renderer(*, use_2dgs: bool):
    return feature_renderer.FeatureFieldRenderer(
        image_height=8,
        image_width=8,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=4.0,
        use_2dgs=use_2dgs,
    )


def test_3dgs_renderer_does_not_require_optional_2dgs_api(monkeypatch):
    monkeypatch.setattr(feature_renderer, "rasterization_2dgs", None)

    renderer = _make_renderer(use_2dgs=False)

    assert renderer.use_2dgs is False


def test_2dgs_renderer_fails_at_construction_when_api_is_missing(monkeypatch):
    monkeypatch.setattr(feature_renderer, "rasterization_2dgs", None)

    with pytest.raises(ImportError, match="rasterization_2dgs"):
        _make_renderer(use_2dgs=True)


def test_feature_raster_intrinsics_are_scaled_from_native_resolution():
    renderer = feature_renderer.FeatureFieldRenderer(
        image_height=100,
        image_width=200,
        fx=160.0,
        fy=80.0,
        cx=100.0,
        cy=50.0,
        use_2dgs=False,
    )

    expected = torch.tensor(
        [[40.0, 0.0, 25.0], [0.0, 20.0, 12.5], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(renderer.scaled_intrinsics(50, 25), expected)
    torch.testing.assert_close(renderer.scaled_intrinsics(200, 100), renderer.K)

    with pytest.raises(ValueError, match="positive"):
        renderer.scaled_intrinsics(0, 25)
