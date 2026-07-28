import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.query_compilers import (
    compile_image_query,
    compile_world_3d_query,
    continuous_gaussian_readout,
)
from radio_gs.querying.query_spec import QueryIntent, SelectionMode


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
