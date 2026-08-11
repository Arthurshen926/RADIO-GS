import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.query_compilers import (
    compile_image_query,
    compile_registered_primitive_seeds,
    compile_world_3d_query,
    continuous_gaussian_readout,
)
from radio_gs.querying.query_spec import (
    PrimitiveUnaryEvidence,
    QueryIntent,
    SelectionMode,
)


def test_continuous_gaussian_readout_normalizes_local_support() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(2, 1, 1) * 0.01
    values = torch.tensor([0.0, 1.0])
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    readout, support = continuous_gaussian_readout(
        xyz, covariance, values, points, candidate_k=2
    )

    assert readout.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)
    assert support.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def _signature(name: str, dim: int) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=1280,
        adaptor_name=name,
        adaptor_sha256="radio-hash",
        adaptor_output_dim=dim,
        token_type="primitive",
        field_checkpoint_sha256="field-hash",
    )


def test_posefree_image_query_selects_one_instance_component() -> None:
    signature = _signature("image", 2)
    query = compile_image_query(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        semantic_signature=signature,
        appearance_signature=signature,
        prototype_count=2,
    )

    assert query.intent is QueryIntent.INSTANCE
    assert query.selection_mode is SelectionMode.TOP_COMPONENT


def test_registered_query_allows_explicit_full_region_selection() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    common = dict(
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
        prototype_count=1,
    )
    frozen = compile_registered_primitive_seeds(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        **common,
    )
    full_region = compile_registered_primitive_seeds(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        positive_prompt_mass=torch.tensor([1.0, 0.0]),
        negative_prompt_mass=torch.tensor([0.0, 0.1]),
        selection_mode=SelectionMode.ALL_COMPONENTS,
        **common,
    )

    assert frozen.selection_mode is SelectionMode.SEEDED_COMPONENT
    assert full_region.selection_mode is SelectionMode.ALL_COMPONENTS
    assert full_region.negative_seeds is not None
    torch.testing.assert_close(
        full_region.negative_seeds.weights, torch.tensor([0.0, 1.0])
    )
    assert full_region.primitive_unary_evidence is not None
    torch.testing.assert_close(
        full_region.primitive_unary_evidence.values,
        torch.tensor([1.0, -0.1]),
    )
    torch.testing.assert_close(
        full_region.primitive_unary_evidence.confidence,
        torch.tensor([1.0, 0.1]),
    )


def test_registered_query_can_preserve_one_shared_observation_scale() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    evidence = PrimitiveUnaryEvidence(
        torch.tensor([0.8, -0.001]),
        "poisson_adjoint",
        confidence=torch.tensor([0.8, 0.001]),
    )

    query = compile_registered_primitive_seeds(
        torch.tensor([0.8, 0.0]),
        torch.tensor([0.0, 0.001]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
        primitive_unary_evidence=evidence,
        seed_normalization="none",
    )

    assert query.positive_seeds is not None
    assert query.negative_seeds is not None
    torch.testing.assert_close(
        query.positive_seeds.weights, torch.tensor([0.8, 0.0])
    )
    torch.testing.assert_close(
        query.negative_seeds.weights, torch.tensor([0.0, 0.001])
    )


def test_world_point_hard_seed_topk_keeps_continuous_descriptor_support() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.08, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(3, 1, 1) * 0.001
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([[0.01, 0.0, 0.0]]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
        euclidean_candidate_k=3,
        seed_topk=1,
        world_point_prototype_mode="per_click_local",
    )
    assert query.positive_seeds is not None
    assert int((query.positive_seeds.weights > 0).sum()) == 1
    assert query.appearance_evidence is not None
    # The local prototype remains a covariance-weighted descriptor, not a
    # copied single primitive, so hard constraints and evidence readout have
    # distinct, explicit roles.
    assert not torch.allclose(query.appearance_evidence.features[0], features[0])
    assert query.metadata["seed_topk"] == 1


def test_world_point_query_accepts_shared_primitive_likelihood_contract() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(2, 1, 1) * 0.001
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    observation = PrimitiveUnaryEvidence.from_probability(
        torch.tensor([0.9, 0.2]),
        confidence=torch.tensor([1.0, 0.5]),
        source="learned_world_interaction_head_v1",
    )

    query = compile_world_3d_query(
        xyz,
        covariance,
        xyz[:1],
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
        primitive_unary_evidence=observation,
    )

    assert query.primitive_unary_evidence is observation
    torch.testing.assert_close(
        observation.foreground_probability, torch.tensor([0.9, 0.2])
    )


def test_world_point_query_preserves_per_click_seed_groups_for_signed_readout() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    covariance = torch.eye(3).repeat(3, 1, 1) * 0.001
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = compile_world_3d_query(
        xyz,
        covariance,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
        negative_points=torch.tensor([[0.02, 0.0, 0.0]]),
        world_point_prototype_mode="per_click_local",
    )
    assert query.positive_seed_groups is not None
    assert query.negative_seed_groups is not None
    assert query.positive_seed_groups.weights.shape == (3, 2)
    assert query.negative_seed_groups.weights.shape == (3, 1)
    assert query.appearance_evidence is not None
    assert query.appearance_evidence.features.shape[0] == 2
    assert query.appearance_evidence.negatives is not None
    assert query.appearance_evidence.negatives.shape[0] == 1


def test_world_point_query_allows_explicit_selection_override_without_changing_default() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(2, 1, 1) * 0.001
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    common = dict(
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_signature("dino", 2),
        boundary_signature=_signature("sam3", 2),
    )
    frozen = compile_world_3d_query(
        xyz, covariance, xyz[:1], **common
    )
    variant = compile_world_3d_query(
        xyz,
        covariance,
        xyz[:1],
        selection_mode=SelectionMode.MIN_SEED_COVER,
        **common,
    )
    assert frozen.selection_mode is SelectionMode.SEEDED_COMPONENT
    assert variant.selection_mode is SelectionMode.MIN_SEED_COVER
