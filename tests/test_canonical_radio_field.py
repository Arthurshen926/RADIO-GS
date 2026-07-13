import pytest
import torch

from radio_gs.field import (
    AffineBasisDecoder,
    CanonicalGaussianField,
    FeatureSpaceSignature,
    fit_affine_basis,
)
from radio_gs.rendering.normalized_splat import affine_commutation_error


def _signature(**overrides) -> FeatureSpaceSignature:
    values = dict(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=8,
        token_type="primitive",
    )
    values.update(overrides)
    return FeatureSpaceSignature(**values)


def test_affine_decoder_commutes_with_alpha_normalized_splat():
    torch.manual_seed(2)
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    coefficients = torch.randn(7, 4)
    weights = torch.rand(11, 7)
    error = affine_commutation_error(weights, coefficients, decoder)
    assert float(error) < 2e-6


def test_canonical_field_is_point_and_batch_invariant_at_initialization():
    torch.manual_seed(3)
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(6, decoder, _signature(), reliability=torch.rand(6, 3))
    one = field.radio_features(torch.tensor([3]))[0]
    batch = field.radio_features(torch.tensor([1, 3, 5]))[1]
    assert torch.allclose(one, batch, atol=1e-7, rtol=1e-6)


def test_stage_one_field_reads_local_coefficients_without_fusion():
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(
        6, decoder, _signature(), reliability=torch.rand(6, 3), use_fusion=False
    )
    rows = torch.tensor([1, 4])
    torch.testing.assert_close(field.coefficients(rows), field.local_codes[rows])


def test_pca_initialization_reconstructs_low_rank_teacher():
    torch.manual_seed(4)
    latent = torch.randn(200, 3)
    teacher = latent @ torch.randn(3, 8) + torch.randn(8)
    decoder, report = fit_affine_basis(teacher, 3, standardize=False)
    reconstruction = decoder(decoder.encode(teacher))
    cosine = torch.nn.functional.cosine_similarity(reconstruction, teacher, dim=-1)
    assert float(cosine.mean()) > 0.999
    assert report.explained_variance_ratio > 0.999


def test_signature_comparison_preserves_but_allows_token_provenance() -> None:
    primitive = _signature(
        adaptor_name="dino_v3_7b.feature_projection",
        adaptor_sha256="radio-hash",
        adaptor_output_dim=8,
        crop_policy="3d_region",
        field_checkpoint_sha256="field-hash",
    )
    spatial = _signature(
        adaptor_name="dino_v3_7b.feature_projection",
        adaptor_sha256="radio-hash",
        adaptor_output_dim=8,
        token_type="spatial",
        crop_policy="official_crop",
        field_checkpoint_sha256="",
    )
    with pytest.raises(ValueError, match="Incompatible"):
        primitive.assert_compatible(spatial)
    primitive.assert_comparable(spatial)
    with pytest.raises(ValueError, match="Incomparable"):
        primitive.assert_comparable(
            _signature(
                adaptor_name="different",
                adaptor_sha256="radio-hash",
                adaptor_output_dim=8,
            )
        )
