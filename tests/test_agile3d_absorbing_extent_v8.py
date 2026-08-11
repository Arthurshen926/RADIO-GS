from __future__ import annotations

import torch

from radio_gs.querying.absorbing_extent_head import (
    AbsorbingExtentInteractionHead,
    StructuredFinalProbability,
    finite_absorbing_seed_reach,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneOneSidedDensityRatioHead,
    QueryLikelihoodInputs,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph() -> PrimitiveSupportGraph:
    # Two appearance-identical but topologically disconnected instances.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    weight = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0])
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=weight,
        raw_affinity=weight,
        local_sigma=torch.ones(6),
        num_nodes=6,
        edge_channels={name: weight.clone() for name in ("geometry", "appearance", "boundary")},
    )


def _base() -> MonotoneOneSidedDensityRatioHead:
    head = MonotoneOneSidedDensityRatioHead(affinity_channel_count=2)
    with torch.no_grad():
        head.raw_slopes.fill_(0.8)
        head.intercepts.fill_(-0.1)
    return head


def _inputs() -> QueryLikelihoodInputs:
    return QueryLikelihoodInputs(
        positive_affinity=torch.full((6, 1, 2), 0.9),
        negative_affinity=torch.empty((6, 0, 2)),
        prior_probability=torch.full((6,), 0.5),
        coverage=torch.ones(6),
        reliability=torch.ones(6),
    )


def test_single_positive_click_reaches_only_connected_instance() -> None:
    reach = finite_absorbing_seed_reach(
        _graph(), torch.tensor([1.0, 0, 0, 0, 0, 0]), steps=3
    )
    assert bool((reach[:3] > 0).all())
    assert torch.equal(reach[3:], torch.zeros(3))


def test_unreached_instance_has_exact_c0_and_zero_selection_probability() -> None:
    result = AbsorbingExtentInteractionHead(_base(), absorbing_steps=3)(
        _inputs(),
        graph=_graph(),
        positive_seeds=torch.tensor([1.0, 0, 0, 0, 0, 0]),
        negative_seeds=torch.zeros(6),
        source="synthetic",
        apply_extent=True,
    )
    assert isinstance(result, StructuredFinalProbability)
    assert torch.equal(result.extent_confidence[3:], torch.zeros(3))
    assert torch.equal(result.foreground_probability[3:], torch.full((3,), 0.5))
    assert torch.equal(result.selection_probability[3:], torch.zeros(3))


def test_negative_click_only_suppresses_reached_lookalike() -> None:
    head = AbsorbingExtentInteractionHead(_base(), absorbing_steps=3)
    positive = torch.tensor([1.0, 0, 0, 1.0, 0, 0])
    before = head(
        _inputs(),
        graph=_graph(),
        positive_seeds=positive,
        negative_seeds=torch.zeros(6),
        source="before",
        apply_extent=True,
    )
    after = head(
        _inputs(),
        graph=_graph(),
        positive_seeds=positive,
        negative_seeds=torch.tensor([0.0, 0, 0, 0, 0, 1.0]),
        source="after",
        apply_extent=True,
    )
    assert bool((after.selection_probability <= before.selection_probability).all())
    assert after.selection_probability[4] < before.selection_probability[4]
    assert torch.equal(after.selection_probability[:3], before.selection_probability[:3])


def test_hard_anchors_are_exact_even_outside_positive_extent() -> None:
    result = AbsorbingExtentInteractionHead(_base(), absorbing_steps=0)(
        _inputs(),
        graph=_graph(),
        positive_seeds=torch.tensor([1.0, 0, 0, 0, 0, 0]),
        negative_seeds=torch.tensor([0.0, 0, 0, 0, 0, 1.0]),
        source="anchor",
        apply_extent=True,
    )
    assert torch.equal(result.selection_probability[[0, 5]], torch.tensor([1.0, 0.0]))
    assert torch.equal(result.extent_confidence[[0, 5]], torch.ones(2))


def test_default_off_is_bitwise_base_identity_and_does_not_claim_bypass() -> None:
    base = _base()
    head = AbsorbingExtentInteractionHead(base, absorbing_steps=3)
    expected = base(_inputs(), source="default")
    actual = head(
        _inputs(),
        graph=_graph(),
        positive_seeds=torch.tensor([1.0, 0, 0, 0, 0, 0]),
        negative_seeds=torch.zeros(6),
        source="default",
    )
    assert type(actual) is type(expected)
    assert actual.source == expected.source
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.confidence, expected.confidence)


def test_structured_result_explicitly_forbids_second_graph_solve() -> None:
    result = AbsorbingExtentInteractionHead(_base(), absorbing_steps=1)(
        _inputs(),
        graph=_graph(),
        positive_seeds=torch.tensor([1.0, 0, 0, 0, 0, 0]),
        negative_seeds=torch.zeros(6),
        source="bypass",
        apply_extent=True,
    )
    assert result.solver_policy == "bypass_existing_graph_solver_with_anchor_and_extent_contract"
    assert not isinstance(result, type(_base()(_inputs(), source="base")))
