import pytest
import torch

from radio_gs.field import (
    BASIS_CONDITIONING_CONTRACT_VERSION,
    MAXIMUM_BASIS_CONDITION_NUMBER_V1,
    AffineBasisDecoder,
    CanonicalGaussianField,
    FeatureSpaceSignature,
    basis_conditioning_report,
    fit_affine_basis,
    validate_basis_conditioning,
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


def test_query_memory_makes_internal_and_canonical_levels_explicit():
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(6, decoder, _signature(), use_fusion=False)
    rows = torch.tensor([1, 4])
    assert torch.equal(
        field.query_memory(rows, representation="local_codes"), field.local_codes[rows]
    )
    assert torch.equal(
        field.query_memory(rows, representation="coefficients"), field.coefficients(rows)
    )
    projection = torch.randn(decoder.feature_dim, 3)
    assert torch.allclose(
        field.query_memory(
            rows, representation="radio_projected", radio_projection=projection
        ),
        field.radio_features(rows) @ projection,
    )
    contract = field.query_memory_contract(
        representation="coefficients", field_sha256="a" * 64,
        canonicalizer_sha256="b" * 64,
    )
    assert contract["dimension"] == 4
    assert contract["representation"] == "coefficients"
    assert contract["canonicalizer_sha256"] == "b" * 64


def test_half_local_training_table_decodes_in_canonical_float_coordinates():
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(6, decoder, _signature(), use_fusion=False)
    field.local_codes = torch.nn.Parameter(field.local_codes.detach().half())

    loss = field.radio_features(torch.tensor([1, 4])).square().mean()
    loss.backward()

    assert field.coefficients().dtype == torch.float32
    assert field.local_codes.grad is not None
    assert field.local_codes.grad.dtype == torch.float16


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


def test_deep_primitive_fusion_preserves_initial_projection_and_batch_invariance():
    torch.manual_seed(17)
    decoder = AffineBasisDecoder(feature_dim=8, coefficient_dim=4)
    field = CanonicalGaussianField(
        5,
        decoder,
        _signature(),
        local_dim=2,
        reliability=torch.rand(5, 3),
        hidden_dim=8,
        fusion_residual_blocks=2,
        use_fusion=True,
    )
    weight = torch.randn(4, 2)
    bias = torch.randn(4)
    field.fusion.initialize_base_projection(weight, bias)

    expected = field.local_codes @ weight.transpose(0, 1) + bias
    torch.testing.assert_close(field.coefficients(), expected)
    torch.testing.assert_close(
        field.radio_features(torch.tensor([3]))[0],
        field.radio_features(torch.tensor([1, 3, 4]))[1],
        atol=1e-7,
        rtol=1e-6,
    )
    assert len(field.fusion.residual_blocks) == 2


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


def test_encode_is_least_squares_inverse_after_basis_loses_orthogonality():
    basis = torch.tensor(
        [
            [2.0, 0.3],
            [0.0, 0.5],
            [1.0, -0.2],
        ]
    )
    decoder = AffineBasisDecoder(
        feature_dim=3,
        coefficient_dim=2,
        mean=torch.tensor([0.2, -0.1, 0.5]),
        scale=torch.tensor([1.0, 2.0, 0.5]),
        basis=basis,
    )
    coefficients = torch.tensor([[0.7, -1.2], [-0.4, 0.9]])
    features = decoder(coefficients)

    restored = decoder.encode(features)

    torch.testing.assert_close(restored, coefficients, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(decoder(restored), features, atol=2e-6, rtol=2e-6)


def test_basis_conditioning_contract_accepts_well_conditioned_basis():
    basis = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 0.5],
            [1.0, 0.2],
        ]
    )

    report = validate_basis_conditioning(basis)

    assert report.contract_version == BASIS_CONDITIONING_CONTRACT_VERSION
    assert report.numerical_rank == 2
    assert report.condition_number < MAXIMUM_BASIS_CONDITION_NUMBER_V1
    assert report.to_dict()["rank_tolerance_semantics"].startswith("max(")


def test_basis_conditioning_contract_rejects_rank_deficient_basis():
    basis = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    report = basis_conditioning_report(basis)
    assert report.numerical_rank == 1
    with pytest.raises(ValueError, match="rank deficient"):
        validate_basis_conditioning(basis)
    with pytest.raises(ValueError, match="rank deficient"):
        AffineBasisDecoder(feature_dim=3, coefficient_dim=2, basis=basis)


def test_basis_conditioning_contract_rejects_ill_conditioned_basis():
    basis = torch.diag(torch.tensor([1.0, 1e-7]))

    report = basis_conditioning_report(basis)
    assert report.numerical_rank == 2
    assert report.condition_number > MAXIMUM_BASIS_CONDITION_NUMBER_V1
    with pytest.raises(ValueError, match="condition number"):
        validate_basis_conditioning(basis)


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
