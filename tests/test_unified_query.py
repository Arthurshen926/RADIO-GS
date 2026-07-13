import numpy as np
import pytest

from radio_gs.querying.unified_query import (
    QueryKind,
    QuerySpace,
    QuerySpec,
    SupportPropagationConfig,
    binary_mask,
    build_support_graph,
    cosine_bank_torch,
    mean_prototype,
    propagate_support,
    score_feature_map,
    score_features,
    seed_connected_component,
)


def test_registered_prompt_matches_cosine_margin() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    query = QuerySpec(
        kind=QueryKind.REGISTERED_2D,
        space=QuerySpace.REGION,
        positive_prototypes=np.array([1.0, 0.0]),
        negative_prototypes=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(score_features(features, query), [1.0, -1.0, 0.0], atol=1e-6)


def test_feature_map_and_mean_prototype_use_same_interface() -> None:
    feature_map = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    flattened = np.moveaxis(feature_map, 0, -1).reshape(-1, 2)
    positive = mean_prototype(flattened, [0])
    negative = mean_prototype(flattened, [1])
    query = QuerySpec(QueryKind.TEXT, QuerySpace.SEMANTIC, positive, negative)
    np.testing.assert_allclose(score_feature_map(feature_map, query), [[1.0, -1.0]])


def test_support_propagation_is_label_free_and_clamps_declared_seeds() -> None:
    xyz = np.stack([np.arange(5, dtype=np.float32) * 0.01, np.zeros(5), np.zeros(5)], axis=1)
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1))
    query = QuerySpec(
        QueryKind.POINT_3D,
        QuerySpace.REGION,
        np.array([1.0, 0.0]),
        positive_seed_indices=(0,),
    )
    scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    config = SupportPropagationConfig(neighbors=2, spatial_sigma=0.05, iterations=3)
    graph = build_support_graph(xyz, features, config)
    output = propagate_support(
        xyz,
        features,
        scores,
        query,
        config,
        graph=graph,
    )
    assert output[0] == pytest.approx(1.0)
    assert output[1] > 0.0

    component = seed_connected_component(output > 0, 0, graph, max_edge_distance=0.03)
    assert component[0]
    assert component[1]


def test_symmetric_adaptive_support_graph_has_reciprocal_edges() -> None:
    xyz = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.03, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
    graph = build_support_graph(
        xyz,
        features,
        SupportPropagationConfig(
            neighbors=1,
            graph_mode="symmetric_union",
            adaptive_spatial=True,
        ),
    )
    for row in range(xyz.shape[0]):
        valid = np.isfinite(graph.distances[row])
        for neighbor in graph.indices[row, valid]:
            reverse_valid = np.isfinite(graph.distances[int(neighbor)])
            assert row in graph.indices[int(neighbor), reverse_valid]
    np.testing.assert_allclose(graph.weights.sum(axis=1), 1.0, atol=1e-6)


def test_fail_closed_for_invalid_prototype_and_scores() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        QuerySpec(QueryKind.TEXT, QuerySpace.SEMANTIC, np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        binary_mask(np.array([np.nan]))


def test_torch_semantic_query_bank_uses_cosine() -> None:
    import torch

    logits = cosine_bank_torch(
        torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    np.testing.assert_allclose(logits.numpy(), np.eye(2), atol=1e-6)
