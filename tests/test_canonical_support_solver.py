import pytest
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
    _hard_seed_masks,
    build_primitive_support_graph,
    graph_for_query_intent,
    mix_support_graph_channels,
    normalized_laplacian_affinity,
    query_conditioned_laplacian_affinity,
    select_support_components,
    solve_primitive_support,
)


def test_confidence_aware_weight_uses_evidence_not_task_name():
    from radio_gs.querying.support_solver import confidence_aware_laplacian_weight

    positive = torch.tensor([1.0, 0.0, 0.0, 0.0])
    negative = torch.tensor([0.0, 0.0, 0.0, 1.0])
    confident = torch.tensor([0.99, 0.95, 0.05, 0.01])
    uncertain = torch.tensor([0.55, 0.52, 0.48, 0.45])
    assert confidence_aware_laplacian_weight(
        uncertain, positive, negative, base_weight=1.0
    ) > confidence_aware_laplacian_weight(
        confident, positive, negative, base_weight=1.0
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


def test_typed_instance_graph_uses_unoriented_surface_normals_when_available():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    )
    appearance = torch.tensor([[1.0, 0.0]] * len(points))
    boundary = torch.tensor([[1.0, 0.0]] * len(points))
    # The first three lie on one unoriented plane; the fourth has a perpendicular
    # local relation.  Reversing a normal must not alter a surface relation.
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    graph = build_primitive_support_graph(
        points,
        appearance_features=appearance,
        boundary_features=boundary,
        normals=normals,
        normal_reliability=torch.ones(len(points)),
        config=SupportGraphConfig(
            neighbors=3,
            normal_temperature=0.10,
            surface_tangent_relation=True,
        ),
    )

    assert {"normal", "surface_tangent"}.issubset(graph.edge_channels)
    index = {tuple(edge): row for row, edge in enumerate(graph.edge_index.T.tolist())}
    assert graph.edge_channels["normal"][index[(0, 1)]] > 0.99
    assert graph.edge_channels["normal"][index[(0, 3)]] < 0.01
    typed = graph_for_query_intent(graph, QueryIntent.INSTANCE, policy="typed")
    expected = mix_support_graph_channels(
        graph,
        {
            "geometry": 0.20,
            "appearance": 0.30,
            "boundary": 0.30,
            "normal": 0.10,
            "surface_tangent": 0.10,
        },
    )
    torch.testing.assert_close(typed.edge_weight, expected.edge_weight)


def test_surface_tangent_relation_rejects_parallel_layer_shortcuts():
    """A normal match alone must not join two nearby parallel surfaces.

    This is a purely local geometry invariant: a displacement within a plane
    is tangent to its normal, whereas a displacement between parallel layers
    has a large normal component.  No query, class, mask, or benchmark label
    participates in this edge relation.
    """

    points = torch.tensor(
        [
            [0.00, 0.0, 0.00],  # reference surface
            [0.04, 0.0, 0.00],  # tangent neighbour on that surface
            [0.00, 0.0, 0.04],  # close but parallel second layer
        ]
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]] * len(points))
    graph = build_primitive_support_graph(
        points,
        normals=normals,
        normal_reliability=torch.ones(len(points)),
        config=SupportGraphConfig(
            neighbors=2,
            surface_tangent_temperature=0.20,
            surface_tangent_relation=True,
        ),
    )
    index = {tuple(edge): row for row, edge in enumerate(graph.edge_index.T.tolist())}
    tangent = graph.edge_channels["surface_tangent"]
    assert tangent[index[(0, 1)]] > 0.99
    assert tangent[index[(0, 2)]] < 0.01


def test_surface_tangent_becomes_neutral_when_local_normals_are_unreliable():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.04]])
    graph = build_primitive_support_graph(
        points,
        normals=torch.tensor([[0.0, 0.0, 1.0]] * len(points)),
        normal_reliability=torch.zeros(len(points)),
        config=SupportGraphConfig(neighbors=1, surface_tangent_relation=True),
    )
    torch.testing.assert_close(
        graph.edge_channels["surface_tangent"],
        torch.ones_like(graph.edge_channels["surface_tangent"]),
    )


def test_typed_instance_graph_uses_query_free_covisibility_when_available():
    points = torch.tensor(
        [[0.00, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0]]
    )
    features = torch.tensor([[1.0, 0.0]] * len(points))
    # The first two primitives share a registered training view; the third
    # does not.  This is source-observation evidence, not a query-side cue.
    visibility = torch.tensor(
        [[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=torch.bool
    )
    graph = build_primitive_support_graph(
        points,
        appearance_features=features,
        boundary_features=features,
        view_observations=visibility,
        config=SupportGraphConfig(neighbors=2, covisibility_weight=1.0),
    )
    assert "covisibility" in graph.edge_channels
    index = {tuple(edge): row for row, edge in enumerate(graph.edge_index.T.tolist())}
    assert graph.edge_channels["covisibility"][index[(0, 1)]] > graph.edge_channels[
        "covisibility"
    ][index[(0, 2)]]
    typed = graph_for_query_intent(graph, QueryIntent.INSTANCE, policy="typed")
    expected = mix_support_graph_channels(
        graph,
        {
            "geometry": 0.20,
            "appearance": 0.35,
            "boundary": 0.35,
            "covisibility": 0.10,
        },
    )
    torch.testing.assert_close(typed.edge_weight, expected.edge_weight)


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


def test_random_walker_enforces_positive_and_negative_seeds_exactly():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    graph = build_primitive_support_graph(points, config=SupportGraphConfig(neighbors=2))
    positive = SoftSeedSet(torch.tensor([1.0, 0.0, 0.0, 0.0]), "positive")
    negative = SoftSeedSet(torch.tensor([0.0, 0.0, 0.0, 1.0]), "negative")
    probability = solve_primitive_support(
        graph,
        torch.zeros(4),
        positive_seeds=positive,
        negative_seeds=negative,
        config=SupportSolverConfig(
            solver_type="random_walker", laplacian_weight=1.0, cg_iterations=128
        ),
    )
    assert probability[0] == 1.0
    assert probability[3] == 0.0
    assert 0.0 < probability[2] < probability[1] < 1.0


def test_relative_hard_seed_policy_resolves_opposite_click_overlap_without_positive_bias():
    """One broad primitive must not silently discard a later negative click.

    The historical policy is retained verbatim for reproducible baselines.
    The generic relative policy only makes a primitive hard when its local
    Gaussian responsibility is larger for one click sign than the other; a
    tie stays soft and is decided by the same feature unary/graph solver.
    """

    positive = torch.tensor([1.0, 0.70, 0.60, 0.0])
    negative = torch.tensor([0.0, 0.80, 0.60, 1.0])
    historic_positive, historic_negative = _hard_seed_masks(
        positive,
        negative,
        SupportSolverConfig(hard_seed_conflict_policy="positive_priority"),
    )
    relative_positive, relative_negative = _hard_seed_masks(
        positive,
        negative,
        SupportSolverConfig(hard_seed_conflict_policy="exclusive_relative"),
    )

    # The historic positive-first tie-break keeps row 1 positive even though
    # the negative click has stronger local support there.
    assert historic_positive.tolist() == [True, True, True, False]
    assert historic_negative.tolist() == [False, False, False, True]
    # Relative conflict resolution assigns row 1 to the stronger negative
    # click and leaves the exact responsibility tie at row 2 soft.
    assert relative_positive.tolist() == [True, False, False, False]
    assert relative_negative.tolist() == [False, True, False, True]


def test_hard_seed_conflict_policy_is_explicitly_validated():
    with pytest.raises(ValueError, match="hard-seed"):
        SupportSolverConfig(hard_seed_conflict_policy="target_aware")
    with pytest.raises(ValueError, match="hard-seed"):
        SupportSolverConfig(hard_seed_conflict_margin=-1e-3)


def test_random_walker_without_hard_seeds_preserves_constant_prior():
    graph = build_primitive_support_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        config=SupportGraphConfig(neighbors=1),
    )
    probability = solve_primitive_support(
        graph,
        torch.zeros(2),
        config=SupportSolverConfig(solver_type="random_walker"),
    )
    torch.testing.assert_close(probability, torch.full((2,), 0.5), atol=1e-5, rtol=0)


def test_cached_normalized_affinity_is_exactly_equivalent_to_in_solver_build():
    graph = build_primitive_support_graph(
        _two_clusters(), config=SupportGraphConfig(neighbors=2)
    )
    positive = SoftSeedSet(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "positive")
    negative = SoftSeedSet(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), "negative")
    config = SupportSolverConfig(
        solver_type="confidence_random_walker", cg_iterations=64
    )
    kwargs = {
        "positive_seeds": positive,
        "negative_seeds": negative,
        "config": config,
    }
    expected = solve_primitive_support(graph, torch.linspace(-0.2, 0.3, 6), **kwargs)
    actual = solve_primitive_support(
        graph,
        torch.linspace(-0.2, 0.3, 6),
        normalized_affinity=normalized_laplacian_affinity(graph),
        **kwargs,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_query_conditioned_affinity_preserves_smooth_edges_and_gates_unary_boundary():
    graph = build_primitive_support_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        config=SupportGraphConfig(neighbors=1),
    )
    prior = torch.tensor([0.95, 0.92, 0.05])
    baseline = normalized_laplacian_affinity(graph)
    gated = query_conditioned_laplacian_affinity(graph, prior, contrast=4.0)
    row, col = graph.edge_index
    smooth = (row == 0) & (col == 1)
    boundary = (row == 1) & (col == 2)
    assert gated[smooth].item() / baseline[smooth].item() > gated[boundary].item() / baseline[boundary].item()
    torch.testing.assert_close(
        query_conditioned_laplacian_affinity(graph, prior, contrast=0.0), baseline
    )
