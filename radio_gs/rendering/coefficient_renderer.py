"""Renderer adapter for a canonical coefficient field."""

from __future__ import annotations

import torch

from radio_gs.field.canonical_gaussian_field import CanonicalGaussianField
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer


def render_canonical_radio(
    renderer: FeatureFieldRenderer,
    gaussian_geometry,
    field: CanonicalGaussianField,
    viewmat: torch.Tensor,
    *,
    feature_height: int | None = None,
    feature_width: int | None = None,
    alpha_eps: float = 1e-6,
    use_reliability: bool = False,
) -> dict[str, torch.Tensor]:
    """Render coefficients once, normalize, then apply the affine decoder."""

    compact = renderer.render_feature_rows(
        gaussian_geometry,
        viewmat,
        field.coefficients(),
        feature_height=feature_height,
        feature_width=feature_width,
        alpha_normalize=True,
        alpha_eps=alpha_eps,
        row_confidence=field.primitive_confidence() if use_reliability else None,
    )
    return {
        **compact,
        "coefficient_map": compact["feature_map"],
        "feature_map": field.decoder.decode_map(compact["feature_map"]),
    }


def render_scalar_support(
    renderer: FeatureFieldRenderer,
    gaussian_geometry,
    probabilities: torch.Tensor,
    viewmat: torch.Tensor,
    *,
    feature_height: int | None = None,
    feature_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Render only the shared primitive-domain support probability."""

    values = torch.as_tensor(probabilities, device=gaussian_geometry.get_xyz().device)
    if values.ndim != 1 or values.shape[0] != gaussian_geometry.get_xyz().shape[0]:
        raise ValueError("probabilities must be row-aligned [num_gaussians]")
    return renderer.render_feature_rows(
        gaussian_geometry,
        viewmat,
        values[:, None],
        feature_height=feature_height,
        feature_width=feature_width,
        alpha_normalize=True,
    )
