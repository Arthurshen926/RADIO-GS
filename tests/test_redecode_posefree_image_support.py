import torch

from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import PrimitiveSupportGraph, SupportSolverConfig
from radio_gs.scripts.redecode_posefree_image_support import (
    decode_posefree_image_unary,
)


def _two_component_graph() -> PrimitiveSupportGraph:
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )
    values = torch.ones(4)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=values,
        raw_affinity=values,
        local_sigma=torch.ones(4),
        num_nodes=4,
        edge_channels={"geometry": values},
    )


def test_posefree_redecode_keeps_only_strongest_instance_component() -> None:
    probabilities, selected = decode_posefree_image_unary(
        _two_component_graph(),
        torch.tensor([1.0, 1.0, 0.2, 0.2]),
        solver_config=SupportSolverConfig(iterations=0, support_threshold=0.5),
        graph_policy="geometry",
        selection_mode=SelectionMode.TOP_COMPONENT,
    )

    assert bool((probabilities >= 0.5).all())
    assert selected.tolist() == [True, True, False, False]
