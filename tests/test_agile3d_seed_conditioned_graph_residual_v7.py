from __future__ import annotations

import torch

from radio_gs.querying.query_likelihood_head import (
    MonotoneOneSidedDensityRatioHead,
    QueryLikelihoodInputs,
)
from radio_gs.querying.seed_conditioned_graph_residual import (
    SeedConditionedGraphResidualHead,
    nonnegative_seed_hop_stack,
    reliability_weighted_support_graph,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph(
    num_nodes: int,
    directed_edges: list[tuple[int, int]],
    weights: list[float],
) -> PrimitiveSupportGraph:
    edge_index = torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
    weight = torch.tensor(weights, dtype=torch.float32)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=weight,
        raw_affinity=weight,
        local_sigma=torch.ones(num_nodes),
        num_nodes=num_nodes,
        edge_channels={
            "geometry": weight.clone(),
            "appearance": weight.clone(),
            "boundary": weight.clone(),
        },
    )


def _base_head() -> MonotoneOneSidedDensityRatioHead:
    head = MonotoneOneSidedDensityRatioHead(affinity_channel_count=2)
    with torch.no_grad():
        head.raw_slopes.fill_(0.7)
        head.intercepts.fill_(-0.1)
    return head


def _inputs(rows: int, *, coverage: float = 1.0) -> QueryLikelihoodInputs:
    return QueryLikelihoodInputs(
        positive_affinity=torch.full((rows, 1, 2), 0.65),
        negative_affinity=torch.empty((rows, 0, 2)),
        prior_probability=torch.full((rows,), 0.5),
        coverage=torch.full((rows,), coverage),
        reliability=torch.ones(rows),
    )


def _head(base: MonotoneOneSidedDensityRatioHead | None = None) -> SeedConditionedGraphResidualHead:
    head = SeedConditionedGraphResidualHead(
        _base_head() if base is None else base,
        propagation_steps=3,
        propagation_decay=1.0,
        max_logit_residual=4.0,
        hard_seed_threshold=0.2,
    )
    with torch.no_grad():
        head.raw_residual_gate.fill_(0.0)
        head.raw_hop_weights.zero_()
    return head


def test_disconnected_instance_receives_no_seed_residual() -> None:
    graph = _graph(
        4,
        [(0, 1), (1, 0), (2, 3), (3, 2)],
        [1.0, 1.0, 1.0, 1.0],
    )
    head = _head()
    inputs = _inputs(4)
    base = head.base_head.log_likelihood_ratio(inputs)
    structured = head.structured_log_likelihood_ratio(
        inputs,
        graph=graph,
        positive_seeds=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        negative_seeds=torch.zeros(4),
    )
    assert structured[1] > base[1]
    assert torch.equal(structured[2:], base[2:])


def test_zero_boundary_conductance_cuts_propagation() -> None:
    graph = _graph(
        3,
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [1.0, 1.0, 0.0, 0.0],
    )
    hops = nonnegative_seed_hop_stack(
        graph,
        torch.tensor([1.0, 0.0, 0.0]),
        steps=3,
        decay=1.0,
    )
    assert bool((hops[1] > 0).any())
    assert torch.equal(hops[2], torch.zeros_like(hops[2]))


def test_negative_lookalike_seed_strictly_suppresses_and_never_increases() -> None:
    graph = _graph(
        3,
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [1.0, 0.5, 0.5, 1.0],
    )
    head = _head()
    inputs = _inputs(3)
    without = head.structured_log_likelihood_ratio(
        inputs,
        graph=graph,
        positive_seeds=torch.tensor([1.0, 0.0, 0.0]),
        negative_seeds=torch.zeros(3),
    )
    with_negative = head.structured_log_likelihood_ratio(
        inputs,
        graph=graph,
        positive_seeds=torch.tensor([1.0, 0.0, 0.0]),
        negative_seeds=torch.tensor([0.0, 0.0, 1.0]),
    )
    assert with_negative[2] < without[2]
    assert bool((with_negative <= without).all())


def test_hard_anchor_probability_identity_is_bitwise_exact() -> None:
    graph = _graph(2, [(0, 1), (1, 0)], [1.0, 1.0])
    output = _head()(
        _inputs(2, coverage=0.0),
        graph=graph,
        positive_seeds=torch.tensor([1.0, 0.0]),
        negative_seeds=torch.tensor([0.0, 1.0]),
        source="synthetic",
        apply_residual=True,
    )
    assert torch.equal(output.foreground_probability, torch.tensor([1.0, 0.0]))
    assert torch.equal(output.confidence, torch.ones(2))


def test_default_off_is_exact_base_head_regression() -> None:
    graph = _graph(2, [(0, 1), (1, 0)], [1.0, 1.0])
    base = _base_head()
    head = _head(base)
    inputs = _inputs(2)
    expected = base(inputs, source="default")
    actual = head(
        inputs,
        graph=graph,
        positive_seeds=torch.tensor([1.0, 0.0]),
        negative_seeds=torch.tensor([0.0, 1.0]),
        source="default",
    )
    assert actual.source == expected.source
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.confidence, expected.confidence)


def test_reliability_edge_gate_is_nonnegative_and_blocks_zero_endpoint() -> None:
    graph = _graph(
        3,
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [1.0, 0.5, 0.5, 1.0],
    )
    gated = reliability_weighted_support_graph(
        graph, torch.tensor([1.0, 1.0, 0.0])
    )
    assert bool((gated.edge_weight >= 0).all())
    assert gated.edge_weight[2].item() == 0.0
    assert gated.edge_weight[3].item() == 0.0
