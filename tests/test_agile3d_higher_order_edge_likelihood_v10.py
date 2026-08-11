from __future__ import annotations

import torch

from radio_gs.querying.higher_order_edge_likelihood import SymmetricHigherOrderEdgeLikelihood
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.materialize_agile3d_higher_order_edge_features_v10 import (
    FEATURE_NAMES,
    build_symmetric_features,
)


def _graph() -> PrimitiveSupportGraph:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    geometry = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.2, 0.2])
    appearance = torch.tensor([0.9, 0.9, 0.85, 0.85, 0.95, 0.95])
    boundary = torch.tensor([0.9, 0.9, 0.85, 0.85, 0.05, 0.05])
    weight = geometry / torch.tensor([0.9, 0.9, 1.0, 1.0, 0.2, 0.2])
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=weight,
        raw_affinity=geometry,
        local_sigma=torch.ones(4),
        num_nodes=4,
        edge_channels={"geometry": geometry, "appearance": appearance, "boundary": boundary},
    )


def _bundle() -> dict[str, torch.Tensor]:
    return {
        "primitive_xyz": torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [2.0, 1, 0]]),
        "primitive_covariance": torch.stack(
            [torch.diag(torch.tensor([0.04, 0.2, 0.3])) for _ in range(4)]
        ),
        "primitive_opacity": torch.tensor([0.8, 0.82, 0.78, 0.3]),
        "reliability": torch.tensor([0.9, 0.9, 0.85, 0.7]),
        "coverage": torch.tensor([0.8, 0.8, 0.75, 0.6]),
    }


def test_higher_order_features_are_symmetric_bounded_and_label_free() -> None:
    edges, features, inventory = build_symmetric_features(
        graph=_graph(), bundle=_bundle(), device=torch.device("cpu"), chunk_size=2
    )
    assert edges.shape == (2, 3)
    assert features.shape == (3, len(FEATURE_NAMES))
    assert bool(((features >= 0) & (features <= 1)).all())
    assert inventory["co_visibility_feature_included"] is False


def test_symmetric_mlp_is_deterministic_for_pre_symmetrized_pair() -> None:
    torch.manual_seed(0)
    head = SymmetricHigherOrderEdgeLikelihood(feature_count=len(FEATURE_NAMES))
    pair = torch.linspace(0.05, 0.95, len(FEATURE_NAMES))[None]
    assert torch.equal(head.log_likelihood_ratio(pair), head.log_likelihood_ratio(pair.clone()))


def test_higher_order_head_rejects_unbounded_or_wrong_shape_features() -> None:
    head = SymmetricHigherOrderEdgeLikelihood(feature_count=len(FEATURE_NAMES))
    try:
        head.log_likelihood_ratio(torch.ones(2, len(FEATURE_NAMES) - 1))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong feature width must fail closed")
