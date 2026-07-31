import torch
import torch.nn.functional as F

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    _score_bank,
    registered_seed_unary,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_compilers import (
    compile_world_3d_query,
    world_point_soft_seeds,
)
from radio_gs.querying.query_spec import (
    PrimitiveUnaryEvidence,
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SoftSeedSet,
)
from radio_gs.querying.score_calibration import (
    SceneSpaceCalibration,
    deterministic_sample_rows,
    fit_scene_space_calibration,
    robust_tanh_score_calibration,
)
from radio_gs.querying.support_solver import build_primitive_support_graph


def _signature(dim: int) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="abc",
        raw_feature_dim=dim,
        adaptor_name="unit",
        adaptor_sha256="def",
        adaptor_output_dim=dim,
        token_type="primitive",
    )


def test_deterministic_sample_rows_includes_fixed_evenly_spaced_rows():
    values = torch.arange(20).reshape(10, 2)
    sampled = deterministic_sample_rows(values, 3)
    torch.testing.assert_close(sampled, values[[0, 4, 9]].float())


def test_robust_scene_calibration_is_deterministic_and_whitens_shape():
    torch.manual_seed(3)
    features = torch.randn(100, 6) * torch.arange(1, 7) + torch.arange(6)
    first = fit_scene_space_calibration(
        features, sample_size=50, background_centroids=3
    )
    second = fit_scene_space_calibration(
        features, sample_size=50, background_centroids=3
    )
    assert first.background_centroids is not None
    assert first.background_centroids.shape == (3, 6)
    torch.testing.assert_close(first.center, second.center)
    torch.testing.assert_close(first.scale, second.scale)
    torch.testing.assert_close(first.background_centroids, second.background_centroids)
    norms = first.transform(features[:5]).norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_scene_calibration_preserves_nonzero_direction_at_exact_robust_center():
    features = F.normalize(
        torch.tensor([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]]), dim=-1
    )
    calibration = fit_scene_space_calibration(
        features, method="diagonal_robust", sample_size=3
    )
    transformed = calibration.transform(features)
    torch.testing.assert_close(
        transformed.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6
    )


def test_robust_tanh_score_calibration_centers_scene_median():
    scores = torch.tensor([-10.0, -1.0, 0.0, 1.0, 50.0])
    calibrated = robust_tanh_score_calibration(scores)
    assert calibrated[2] == 0
    assert bool((calibrated.abs() <= 1).all())


def test_centered_score_calibration_is_explicit_optional_diagnostic():
    scores = torch.tensor([-2.0, 1.0, 2.0, 3.0, 20.0])
    zero_preserving = robust_tanh_score_calibration(scores)
    centered = robust_tanh_score_calibration(scores, preserve_zero=False)
    assert zero_preserving[2] > 0
    assert centered[2] == 0


def test_registered_seed_unary_preserves_soft_signed_responsibility() -> None:
    actual = registered_seed_unary(
        torch.tensor([1.0, 0.5, 0.0]),
        torch.tensor([0.0, 0.1, 0.5]),
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.4, -0.5]))


def test_registered_seed_unary_is_not_erased_by_field_reliability() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    features = F.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=-1
    )
    signature = _signature(2)
    query = QuerySpec(
        modality=QueryModality.REGISTERED_2D,
        intent=QueryIntent.REGION,
        registration=RegistrationMode.CAMERA,
        appearance_evidence=PrototypeSet(features[0], signature),
        positive_seeds=SoftSeedSet(torch.tensor([1.0, 0.4, 0.0]), "unit"),
        negative_seeds=SoftSeedSet(torch.tensor([0.0, 0.0, 1.0]), "unit"),
        primitive_unary_evidence=PrimitiveUnaryEvidence(
            torch.tensor([1.0, 0.4, -1.0]),
            "unit_joint_mass",
        ),
    )
    engine = CanonicalQueryEngine(
        build_primitive_support_graph(xyz),
        scoring_config=EvidenceScoringConfig(
            registered_seed_unary_weight=1.0
        ),
        node_reliability=torch.zeros(3),
    )
    result = engine.execute(
        query,
        {"appearance": features},
        feature_signatures={"appearance": signature},
    )

    torch.testing.assert_close(result.unary, torch.tensor([1.0, 0.4, -1.0]))
    torch.testing.assert_close(
        result.evidence_components["appearance"], torch.zeros(3)
    )
    torch.testing.assert_close(
        result.evidence_components["registered_seed"],
        torch.tensor([1.0, 0.4, -1.0]),
    )


def test_uncalibrated_score_bank_preserves_original_explicit_negative_rule():
    field = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    evidence = PrototypeSet(
        torch.tensor([[1.0, 0.0]]),
        _signature(2),
        negatives=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
    )
    actual = _score_bank(field, evidence, temperature=0.07, chunk_size=1)
    expected = field @ evidence.features.T
    expected = expected[:, 0] - (field @ evidence.negatives.T).amax(dim=1)
    torch.testing.assert_close(actual, expected)


def test_background_prior_does_not_dilute_an_explicit_negative_click():
    """Scene modes are a prior, while an interaction negative stays hard evidence."""

    field = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=-1
    )
    evidence = PrototypeSet(
        torch.tensor([[1.0, 0.0]]),
        _signature(2),
        negatives=torch.tensor([[0.0, 1.0]]),
    )
    calibration = SceneSpaceCalibration(
        center=torch.zeros(2),
        scale=torch.ones(2),
        background_centroids=torch.tensor([[-1.0, 0.0]]),
        method="none",
        sample_count=3,
    )
    actual = _score_bank(
        field,
        evidence,
        temperature=0.07,
        calibration=calibration,
        background_negative_policy="explicit_hard_max",
    )
    expected = (field @ evidence.features.T)[:, 0] - torch.maximum(
        (field @ evidence.negatives.T).amax(dim=1),
        (field @ calibration.background_centroids.T)[:, 0],
    )
    torch.testing.assert_close(actual, expected)


def test_calibrated_background_scoring_is_chunk_size_invariant() -> None:
    generator = torch.Generator().manual_seed(19)
    field = F.normalize(torch.randn(23, 7, generator=generator), dim=-1)
    evidence = PrototypeSet(
        F.normalize(torch.randn(3, 7, generator=generator), dim=-1),
        _signature(7),
        negatives=F.normalize(torch.randn(2, 7, generator=generator), dim=-1),
    )
    calibration = fit_scene_space_calibration(
        field,
        method="diagonal_robust",
        sample_size=17,
        background_centroids=4,
    )
    reference = _score_bank(
        field,
        evidence,
        temperature=0.07,
        calibration=calibration,
        background_negative_policy="pooled_mean",
        chunk_size=23,
    )
    memory_bounded = _score_bank(
        field,
        evidence,
        temperature=0.07,
        calibration=calibration,
        background_negative_policy="pooled_mean",
        chunk_size=3,
    )

    torch.testing.assert_close(memory_bounded, reference, atol=1e-6, rtol=1e-6)


def test_engine_fits_label_free_calibration_once_and_returns_finite_scores():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    features = F.normalize(
        torch.tensor([[1.0, 0.1], [0.9, 0.2], [0.1, 1.0]]), dim=-1
    )
    graph = build_primitive_support_graph(xyz)
    signature = _signature(2)
    query = QuerySpec(
        modality=QueryModality.TEXT,
        intent=QueryIntent.CATEGORY,
        registration=RegistrationMode.NONE,
        appearance_evidence=PrototypeSet(features[0], signature),
    )
    engine = CanonicalQueryEngine(
        graph,
        scoring_config=EvidenceScoringConfig(
            feature_calibration="diagonal_robust",
            background_centroids=2,
            calibration_sample_size=3,
            score_calibration="robust_tanh",
        ),
    )
    result = engine.execute(
        query,
        {"appearance": features},
        feature_signatures={"appearance": signature},
    )
    assert bool(torch.isfinite(result.unary).all())
    assert set(engine._calibrations) == {"appearance"}
    assert result.score_calibration == "robust_tanh"


def test_explicit_negative_evidence_can_be_localized_without_moving_positive_evidence():
    signature = _signature(2)
    evidence = PrototypeSet(
        torch.tensor([1.0, 0.0]),
        signature,
        negatives=torch.tensor([[0.0, 1.0]]),
    )
    field = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    localized = _score_bank(
        field,
        evidence,
        temperature=0.07,
        explicit_negative_influence=torch.tensor([1.0, 0.0]),
    )
    assert localized[0] < -0.99
    assert abs(float(localized[1])) < 1e-6


def test_signed_spatial_evidence_keeps_each_negative_coupled_to_its_click():
    signature = _signature(2)
    evidence = PrototypeSet(
        torch.tensor([1.0, 0.0]),
        signature,
        negatives=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    field = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    # Both nodes match the first negative descriptor. Only node zero is
    # geodesically close to the click that produced that descriptor.
    localized = _score_bank(
        field,
        evidence,
        temperature=0.07,
        explicit_negative_spatial=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]]
        ),
        spatial_log_weight=1.0,
        spatial_floor=1e-6,
    )
    assert localized[0] < -0.99
    assert abs(float(localized[1])) < 1e-4


def test_signed_spatial_evidence_keeps_each_positive_coupled_to_its_click():
    signature = _signature(2)
    evidence = PrototypeSet(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        signature,
    )
    field = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    localized = _score_bank(
        field,
        evidence,
        temperature=0.07,
        positive_spatial_influence=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]]
        ),
        spatial_log_weight=1.0,
        spatial_floor=1e-6,
    )
    assert localized[0] > 0.95
    # The only nearby descriptor is orthogonal. The small negative offset is
    # the existing equal-prototype log weight, not leakage from the far click.
    assert -0.06 < float(localized[1]) < -0.04


def test_engine_applies_explicit_modality_score_policy_without_changing_default():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    features = F.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]), dim=-1
    )
    signature = _signature(2)
    evidence = PrototypeSet(features[0], signature)
    registered = QuerySpec(
        modality=QueryModality.REGISTERED_2D,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.CAMERA,
        appearance_evidence=evidence,
    )
    world = QuerySpec(
        modality=QueryModality.WORLD_3D,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.WORLD,
        appearance_evidence=evidence,
    )
    engine = CanonicalQueryEngine(
        build_primitive_support_graph(xyz),
        score_calibration_by_modality={
            QueryModality.WORLD_3D: "robust_tanh_zero"
        },
    )
    kwargs = {
        "feature_signatures": {"appearance": signature},
    }
    registered_result = engine.execute(
        registered, {"appearance": features}, **kwargs
    )
    world_result = engine.execute(world, {"appearance": features}, **kwargs)
    assert registered_result.score_calibration == "none"
    assert world_result.score_calibration == "robust_tanh_zero"
    torch.testing.assert_close(registered_result.unary, features @ features[0])
    assert not torch.allclose(world_result.unary, registered_result.unary)


def test_engine_rejects_invalid_modality_score_policy_at_construction():
    graph = build_primitive_support_graph(torch.zeros(1, 3))
    try:
        CanonicalQueryEngine(
            graph,
            score_calibration_by_modality={"world_3d": "not-a-calibration"},
        )
    except ValueError as error:
        assert "score_calibration" in str(error)
    else:
        raise AssertionError("invalid score calibration should fail closed")


def test_precomputed_gaussian_precision_preserves_world_point_seeds():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    covariance = torch.eye(3)[None].repeat(2, 1, 1) * torch.tensor([1.0, 2.0])[:, None, None]
    identity = torch.eye(3)
    precision = torch.linalg.pinv(covariance + 1e-6 * identity)
    expected = world_point_soft_seeds(xyz, covariance, torch.tensor([0.2, 0.0, 0.0]))
    actual = world_point_soft_seeds(
        xyz,
        covariance,
        torch.tensor([0.2, 0.0, 0.0]),
        gaussian_precision=precision,
    )
    torch.testing.assert_close(actual, expected)


def test_world_point_seed_topk_sparsifies_without_labels():
    xyz = torch.stack([torch.arange(5).float(), torch.zeros(5), torch.zeros(5)], dim=1)
    covariance = torch.eye(3)[None].repeat(5, 1, 1)
    features = F.normalize(torch.randn(5, 3, generator=torch.Generator().manual_seed(4)), dim=-1)
    query = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([0.1, 0.0, 0.0]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature(3),
        boundary_signature=_signature(3),
        seed_topk=2,
    )
    assert query.positive_seeds is not None
    assert int((query.positive_seeds.weights > 0).sum()) == 2
    assert query.metadata["seed_topk"] == 2


def test_world_point_euclidean_candidates_block_distant_large_gaussian() -> None:
    xyz = torch.tensor([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    covariance = torch.stack([torch.eye(3) * 0.01, torch.eye(3) * 100.0])
    point = torch.zeros(3)
    unrestricted = world_point_soft_seeds(xyz, covariance, point)
    local = world_point_soft_seeds(
        xyz, covariance, point, euclidean_candidate_k=1
    )
    assert int(unrestricted.argmax()) == 1
    assert int(local.argmax()) == 0
    assert local[1] == 0


def test_world_point_candidate_mask_enforces_surface_support() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(3, 1, 1)
    weights = world_point_soft_seeds(
        xyz,
        covariance,
        torch.zeros(3),
        candidate_mask=torch.tensor([True, False, True]),
    )
    assert weights[0] > 0 and weights[2] > 0
    assert weights[1] == 0


def test_world_point_seeds_are_stable_when_gaussian_density_underflows() -> None:
    """World clicks use relative kernel responsibility, not raw density scale."""

    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    covariance = torch.eye(3)[None].repeat(2, 1, 1) * 1e-8
    weights = world_point_soft_seeds(
        xyz,
        covariance,
        torch.tensor([10.0, 0.0, 0.0]),
        euclidean_candidate_k=2,
    )
    assert torch.isfinite(weights).all()
    assert float(weights.max()) == 1.0
    assert int((weights > 0).sum()) == 1


def test_world_point_seed_temperature_sharpens_relative_responsibility():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    covariance = torch.eye(3)[None].repeat(2, 1, 1)
    features = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    kwargs = dict(
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature(2),
        boundary_signature=_signature(2),
    )
    base = compile_world_3d_query(
        xyz, covariance, torch.tensor([0.0, 0.0, 0.0]), **kwargs
    )
    sharp = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([0.0, 0.0, 0.0]),
        seed_temperature=0.5,
        **kwargs,
    )
    assert base.positive_seeds is not None and sharp.positive_seeds is not None
    assert sharp.positive_seeds.weights[1] < base.positive_seeds.weights[1]
    assert sharp.metadata["seed_temperature"] == 0.5


def test_world_point_per_click_prototypes_preserve_multiple_interaction_modes():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    covariance = torch.eye(3)[None].repeat(2, 1, 1) * 0.001
    features = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    query = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature(2),
        boundary_signature=_signature(2),
        world_point_prototype_mode="per_click_local",
    )
    assert query.appearance_evidence is not None
    assert query.appearance_evidence.features.shape == (2, 2)
    assert query.metadata["world_point_prototype_mode"] == "per_click_local"


def test_world_point_equal_click_weighting_prevents_kernel_scale_dominance():
    xyz = torch.tensor(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [1.00, 0.0, 0.0], [1.05, 0.0, 0.0]]
    )
    covariance = torch.diag_embed(
        torch.tensor(
            [[1e-2, 1e-2, 1e-2], [1e-2, 1e-2, 1e-2], [1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6]]
        )
    )
    features = F.normalize(torch.eye(4), dim=-1)
    kwargs = dict(
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature(4),
        boundary_signature=_signature(4),
        world_point_prototype_mode="per_click_local",
    )
    historical = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        world_point_prototype_weighting="support_mass",
        **kwargs,
    )
    equal = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        world_point_prototype_weighting="equal_click",
        **kwargs,
    )
    assert historical.appearance_evidence is not None
    assert equal.appearance_evidence is not None
    assert not torch.allclose(
        historical.appearance_evidence.weights,
        torch.full((2,), 0.5),
    )
    torch.testing.assert_close(
        equal.appearance_evidence.weights, torch.full((2,), 0.5)
    )
    assert equal.metadata["world_point_prototype_weighting"] == "equal_click"
