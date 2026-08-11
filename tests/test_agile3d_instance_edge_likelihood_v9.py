from __future__ import annotations

import torch

from radio_gs.querying.instance_edge_likelihood import (
    InstanceEdgeFeatures,
    MonotoneInstanceEdgeLikelihood,
    absorbing_component_seed_support,
    gate_graph_by_instance_edge_likelihood,
    instance_edge_features_from_graph,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph() -> PrimitiveSupportGraph:
    # 0-1-2 is one long instance; 2-3 is an appearance-similar boundary edge.
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    weight = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
    geometry = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.2, 0.2])
    appearance = torch.tensor([0.9, 0.9, 0.85, 0.85, 0.95, 0.95])
    boundary = torch.tensor([0.9, 0.9, 0.85, 0.85, 0.05, 0.05])
    return PrimitiveSupportGraph(
        edge_index=edges,
        edge_weight=weight,
        raw_affinity=weight,
        local_sigma=torch.ones(4),
        num_nodes=4,
        edge_channels={
            "geometry": geometry,
            "appearance": appearance,
            "boundary": boundary,
        },
    )


def _head() -> MonotoneInstanceEdgeLikelihood:
    head = MonotoneInstanceEdgeLikelihood()
    with torch.no_grad():
        head.bias.fill_(-1.0)
        head.raw_appearance_weight.fill_(0.0)
        head.raw_boundary_weight.fill_(2.0)
        head.raw_distance_weight.fill_(2.0)
        head.raw_reliability_weight.fill_(0.0)
        head.raw_coverage_weight.fill_(0.0)
    return head


def test_same_appearance_cross_instance_boundary_is_pruned() -> None:
    graph = _graph()
    features = instance_edge_features_from_graph(
        graph, reliability=torch.ones(4), coverage=torch.ones(4)
    )
    gated = gate_graph_by_instance_edge_likelihood(
        graph, features, _head(), apply_edge_likelihood=True
    )
    assert gated.edge_weight[4].item() == 0.0
    assert gated.edge_weight[5].item() == 0.0


def test_long_object_chain_edges_remain_connected() -> None:
    graph = _graph()
    features = instance_edge_features_from_graph(
        graph, reliability=torch.ones(4), coverage=torch.ones(4)
    )
    gated = gate_graph_by_instance_edge_likelihood(
        graph, features, _head(), apply_edge_likelihood=True
    )
    assert bool((gated.edge_weight[:4] > 0).all())
    support = absorbing_component_seed_support(
        gated, torch.tensor([True, False, False, False])
    )
    assert torch.equal(support, torch.tensor([True, True, True, False]))


def test_edge_derivative_signs_are_auditable() -> None:
    head = _head()
    base = InstanceEdgeFeatures(
        appearance_similarity=torch.tensor([0.5]),
        boundary_similarity=torch.tensor([0.5]),
        scaled_distance=torch.tensor([0.5]),
        endpoint_reliability=torch.tensor([0.5]),
        endpoint_coverage=torch.tensor([0.5]),
    )
    score = head.log_likelihood_ratio(base)
    fields = list(vars(base))
    for index, name in enumerate(fields):
        changed = dict(vars(base))
        changed[name] = torch.tensor([0.6])
        delta = head.log_likelihood_ratio(InstanceEdgeFeatures(**changed)) - score
        assert (delta < 0).item() if name == "scaled_distance" else (delta > 0).item()
    weights = head.signed_weights
    assert bool((weights[[0, 1, 3, 4]] >= 0).all())
    assert weights[2] <= 0


def test_default_off_returns_exact_graph_object() -> None:
    graph = _graph()
    features = instance_edge_features_from_graph(
        graph, reliability=torch.ones(4), coverage=torch.ones(4)
    )
    actual = gate_graph_by_instance_edge_likelihood(graph, features, _head())
    assert actual is graph


def test_edge_features_are_query_independent_bounded_capabilities() -> None:
    graph = _graph()
    features = instance_edge_features_from_graph(
        graph,
        reliability=torch.tensor([0.7, 0.8, 0.9, 1.0]),
        coverage=torch.tensor([0.6, 0.7, 0.8, 0.9]),
    )
    matrix = features.matrix()
    assert matrix.shape == (6, 5)
    assert bool(((matrix >= 0) & (matrix <= 1)).all())
