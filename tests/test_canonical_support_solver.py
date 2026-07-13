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
    build_primitive_support_graph,
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
