import torch

from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import (
    QueryIntent,
    QueryModality,
    QuerySpec,
    RegistrationMode,
    SelectionMode,
    SoftSeedSet,
)
from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    SupportSolverConfig,
    _component_labels,
    build_primitive_support_graph,
    graph_for_query_intent,
    mix_support_graph_channels,
    select_support_components,
)


def _two_clusters():
    return torch.tensor(
        [
            [0.00, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [2.00, 0.0, 0.0],
            [2.02, 0.0, 0.0],
            [2.04, 0.0, 0.0],
        ]
    )


def test_surface_graph_is_symmetric_and_row_normalized():
    graph = build_primitive_support_graph(
        _two_clusters(),
        appearance_features=torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3),
        config=SupportGraphConfig(neighbors=2),
    )
    edges = {tuple(pair) for pair in graph.edge_index.T.tolist()}
    assert all((right, left) in edges for left, right in edges)
    row_sum = torch.zeros(graph.num_nodes)
    row_sum.index_add_(0, graph.edge_index[0], graph.edge_weight)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6)
    assert set(graph.edge_channels) == {"geometry", "appearance"}
    torch.testing.assert_close(
        graph.raw_affinity,
        graph.edge_channels["geometry"] * graph.edge_channels["appearance"],
    )


def test_multichannel_mixture_is_normalized_and_typed():
    appearance = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    boundary = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] * 2
    )
    graph = build_primitive_support_graph(
        _two_clusters(),
        appearance_features=appearance,
        boundary_features=boundary,
        config=SupportGraphConfig(neighbors=2),
    )
    mixed = mix_support_graph_channels(
        graph, {"geometry": 0.2, "appearance": 0.4, "boundary": 0.4}
    )
    row_sum = torch.zeros(mixed.num_nodes)
    row_sum.index_add_(0, mixed.edge_index[0], mixed.edge_weight)
    torch.testing.assert_close(row_sum, torch.ones_like(row_sum), atol=1e-6, rtol=0)
    typed = graph_for_query_intent(graph, QueryIntent.INSTANCE, policy="typed")
    torch.testing.assert_close(typed.edge_weight, mixed.edge_weight)
    legacy_endpoint = mix_support_graph_channels(
        graph,
        {"geometry": 1.0},
        legacy_residual=1.0,
    )
    torch.testing.assert_close(legacy_endpoint.edge_weight, graph.edge_weight)


def test_mutual_knn_topology_is_a_symmetric_subset_of_union():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    union = build_primitive_support_graph(
        points,
        config=SupportGraphConfig(neighbors=1, topology_mode="symmetric_union"),
    )
    mutual = build_primitive_support_graph(
        points,
        config=SupportGraphConfig(neighbors=1, topology_mode="mutual_knn"),
    )
    union_edges = {tuple(edge) for edge in union.edge_index.T.tolist()}
    mutual_edges = {tuple(edge) for edge in mutual.edge_index.T.tolist()}
    assert mutual_edges
    assert mutual_edges < union_edges
    assert all((right, left) in mutual_edges for left, right in mutual_edges)


def test_seed_only_query_uses_shared_graph_and_keeps_seeded_component():
    graph = build_primitive_support_graph(
        _two_clusters(), config=SupportGraphConfig(neighbors=2)
    )
    seeds = torch.zeros(6)
    seeds[0] = 1.0
    query = QuerySpec(
        modality=QueryModality.WORLD_3D,
        intent=QueryIntent.INSTANCE,
        registration=RegistrationMode.WORLD,
        positive_seeds=SoftSeedSet(seeds, "unit_test"),
        selection_mode=SelectionMode.SEEDED_COMPONENT,
    )
    engine = CanonicalQueryEngine(
        graph,
        solver_config=SupportSolverConfig(
            iterations=20, residual=0.1, support_threshold=0.1
        ),
    )
    result = engine.execute(query, {})
    assert result.selected_support[:3].any()
    assert not result.selected_support[3:].any()
    assert result.probabilities[0] == 1.0


def test_seeded_bfs_fast_path_matches_full_component_labels():
    graph = build_primitive_support_graph(
        _two_clusters(), config=SupportGraphConfig(neighbors=2)
    )
    probabilities = torch.tensor([0.9, 0.8, 0.7, 0.9, 0.8, 0.7])
    seeds = torch.zeros(6)
    seeds[0] = 1.0
    seeds[3] = 0.3
    positive = SoftSeedSet(seeds, "unit_test")
    config = SupportSolverConfig(support_threshold=0.5)
    actual = select_support_components(
        graph,
        probabilities,
        SelectionMode.SEEDED_COMPONENT,
        positive_seeds=positive,
        config=config,
    )
    active = probabilities >= config.support_threshold
    labels = _component_labels(graph, active, config.component_edge_threshold)
    expected = torch.zeros(6, dtype=torch.bool)
    for component in labels[labels >= 0].unique():
        if bool((positive.weights[labels == component] >= config.seeded_component_min_weight).any()):
            expected |= labels == component
    torch.testing.assert_close(actual.cpu(), expected)
