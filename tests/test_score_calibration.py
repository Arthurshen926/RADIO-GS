import torch
import torch.nn.functional as F

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    _score_bank,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_compilers import (
    compile_world_3d_query,
    world_point_soft_seeds,
)
from radio_gs.querying.query_spec import (
    PrototypeSet,
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
)
from radio_gs.querying.score_calibration import (
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
