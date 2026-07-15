import torch

from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    solve_primitive_support,
)
from radio_gs.scripts.augment_primary_support_graph import (
    build_primary_anchored_completion_graph,
)


def test_primary_anchored_completion_preserves_primary_solution() -> None:
    primary_graph = PrimitiveSupportGraph(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.ones(2),
        raw_affinity=torch.ones(2),
        local_sigma=torch.ones(2),
        num_nodes=2,
        edge_channels={
            "geometry": torch.ones(2),
            "appearance": torch.ones(2),
            "boundary": torch.ones(2),
        },
    )
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.0, 0.0]]
    )
    valid = torch.ones(4, dtype=torch.bool)
    primary_valid = torch.tensor([True, False, True, False])
    graph, global_rows, stats = build_primary_anchored_completion_graph(
        xyz=xyz,
        valid=valid,
        primary_valid=primary_valid,
        primary_global_rows=torch.tensor([0, 2]),
        primary_graph=primary_graph,
        appearance_features=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        boundary_features=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        neighbors=1,
        spatial_scale=2.0,
        appearance_temperature=0.1,
        boundary_temperature=0.1,
    )

    assert torch.equal(global_rows, torch.arange(4))
    assert stats["primary_transition_exact"] is True
    assert graph.edge_index.shape[1] == 4
    # The copied primary rows (global/local 0 and 2) never receive fallback edges.
    assert set(graph.edge_index[0, 2:].tolist()) == {1, 3}
    torch.testing.assert_close(
        graph.edge_weight[:2], primary_graph.edge_weight, atol=0, rtol=0
    )

    config = SupportSolverConfig(iterations=4, residual=0.3, unary_temperature=0.1)
    old_unary = torch.tensor([0.8, -0.8])
    completed_unary = torch.tensor([0.8, -0.2, -0.8, -0.2])
    old_probability = solve_primitive_support(
        primary_graph, old_unary, config=config
    )
    completed_probability = solve_primitive_support(
        graph, completed_unary, config=config
    )
    torch.testing.assert_close(
        completed_probability[primary_valid], old_probability, atol=0, rtol=0
    )
    assert torch.equal(
        graph.edge_index[:, 2:], torch.tensor([[1, 3], [0, 2]])
    )
