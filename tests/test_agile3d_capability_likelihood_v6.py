from __future__ import annotations

import torch

from radio_gs.querying.query_likelihood_head import (
    MonotoneOneSidedDensityRatioHead,
    QueryLikelihoodInputs,
)


def _inputs(positive: torch.Tensor, negative: torch.Tensor) -> QueryLikelihoodInputs:
    rows = positive.shape[0]
    return QueryLikelihoodInputs(
        positive_affinity=positive,
        negative_affinity=negative,
        prior_probability=torch.full((rows,), 0.5),
        coverage=torch.ones(rows),
        reliability=torch.ones(rows),
    )


def _head() -> MonotoneOneSidedDensityRatioHead:
    head = MonotoneOneSidedDensityRatioHead(affinity_channel_count=2)
    with torch.no_grad():
        head.raw_slopes.fill_(1.0)
        head.intercepts.fill_(-0.2)
    return head


def test_negative_lookalike_strictly_decreases() -> None:
    head = _head()
    positive = torch.full((2, 1, 2), 0.9)
    before = head.log_likelihood_ratio(_inputs(positive, torch.empty((2, 0, 2))))
    negative = torch.tensor([[[0.2, 0.2]], [[0.95, 0.95]]])
    after = head.log_likelihood_ratio(_inputs(positive, negative))
    assert after[1] < before[1]


def test_unrelated_to_negative_is_exactly_unchanged() -> None:
    head = _head()
    positive = torch.full((3, 1, 2), 0.85)
    before = head.log_likelihood_ratio(_inputs(positive, torch.empty((3, 0, 2))))
    unrelated = torch.full((3, 1, 2), 0.1)
    after = head.log_likelihood_ratio(_inputs(positive, unrelated))
    assert torch.equal(after, before)


def test_adding_negative_click_never_increases_any_unary() -> None:
    head = _head()
    positive = torch.tensor(
        [[[0.9, 0.8]], [[0.6, 0.7]], [[0.2, 0.3]], [[0.95, 0.95]]]
    )
    one_negative = torch.tensor(
        [[[0.1, 0.2]], [[0.8, 0.7]], [[0.4, 0.4]], [[0.9, 0.9]]]
    )
    two_negative = torch.cat((one_negative, torch.full((4, 1, 2), 0.75)), dim=1)
    first = head.log_likelihood_ratio(_inputs(positive, one_negative))
    second = head.log_likelihood_ratio(_inputs(positive, two_negative))
    assert bool((second <= first).all())
