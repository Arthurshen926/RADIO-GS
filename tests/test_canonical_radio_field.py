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


def test_compact_spatial_field_is_primitive_and_batch_invariant():
    torch.manual_seed(11)
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    positions = torch.rand(6, 3)
    field = CanonicalGaussianField(
        6,
        decoder,
        _signature(),
        local_dim=2,
        coarse_dim=2,
        primitive_positions=positions,
        spatial_hash={
            "output_dim": 2,
            "num_levels": 2,
            "features_per_level": 2,
            "log2_hashmap_size": 4,
            "base_resolution": 2,
            "max_resolution": 4,
            "hidden_dim": 8,
        },
        reliability=torch.rand(6, 3),
        hidden_dim=8,
        use_fusion=True,
    )
    weight = torch.randn(4, 2)
    bias = torch.randn(4)
    field.fusion.initialize_base_projection(weight, bias)

    one = field.radio_features(torch.tensor([3]))[0]
    batch = field.radio_features(torch.tensor([1, 3, 5]))[1]

    torch.testing.assert_close(one, batch, atol=1e-7, rtol=1e-6)
    assert field.normalized_positions.dtype == torch.float16
    assert field.spatial_encoder.architecture()["output_dim"] == 2


def test_low_dimensional_fusion_starts_from_analytical_projection():
    torch.manual_seed(13)
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(
        5,
        decoder,
        _signature(),
        local_dim=2,
        reliability=torch.rand(5, 3),
        hidden_dim=8,
        use_fusion=True,
    )
    weight = torch.randn(4, 2)
    bias = torch.randn(4)
    field.fusion.initialize_base_projection(weight, bias)

    expected = field.local_codes @ weight.transpose(0, 1) + bias

    torch.testing.assert_close(field.coefficients(), expected)


def test_reliability_can_be_stored_without_entering_fusion():
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(
        5,
        decoder,
        _signature(),
        local_dim=2,
        reliability=torch.rand(5, 3),
        fusion_reliability=False,
        hidden_dim=8,
        use_fusion=True,
    )

    assert field.reliability.shape == (5, 3)
    assert field.fusion.reliability_dim == 0
    assert field.coefficients().shape == (5, 4)


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
