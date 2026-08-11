from __future__ import annotations

import torch

from radio_gs.querying.query_likelihood_head import (
    MonotoneChannelDensityRatioHead,
    QueryLikelihoodInputs,
)
from radio_gs.scripts.train_capability_density_ratio_head import (
    density_ratio_posterior_loss,
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


def test_source_posterior_fit_learns_a_negative_null_ratio() -> None:
    head = MonotoneChannelDensityRatioHead(affinity_channel_count=2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    positive = torch.tensor([[0.85, 0.75], [0.80, 0.70], [0.90, 0.80]])
    negative = torch.tensor(
        [[-0.10, 0.00], [0.05, -0.05], [0.00, 0.10], [-0.05, 0.00]]
    )
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = density_ratio_posterior_loss(
            positive,
            negative,
            prevalence=0.05,
            raw_slopes=head.raw_slopes,
            intercepts=head.intercepts,
        )
        loss.backward()
        optimizer.step()
    null_ratio = head.per_observation_log_likelihood_ratio(
        torch.full((1, 1, 2), 0.5)
    )
    match_ratio = head.per_observation_log_likelihood_ratio(
        torch.tensor([[[0.925, 0.875]]])
    )
    assert bool((null_ratio < 0).all())
    assert bool((match_ratio > 0).all())


def test_empty_observations_have_exact_zero_aggregate_ratio() -> None:
    head = MonotoneChannelDensityRatioHead(affinity_channel_count=2)
    with torch.no_grad():
        head.intercepts.fill_(-2.0)
    empty = torch.empty((5, 0, 2))
    ratio = head.log_likelihood_ratio(_inputs(empty, empty))
    assert torch.equal(ratio, torch.zeros(5))


def test_density_ratio_is_monotone_and_negative_click_suppresses() -> None:
    head = MonotoneChannelDensityRatioHead(affinity_channel_count=2)
    with torch.no_grad():
        head.intercepts.fill_(-0.4)
    low = torch.full((2, 1, 2), 0.55)
    high = low.clone()
    high[1] = 0.95
    positive_ratio = head.log_likelihood_ratio(
        _inputs(high, torch.empty((2, 0, 2)))
    )
    with_negative = head.log_likelihood_ratio(_inputs(high, high))
    assert positive_ratio[1] > positive_ratio[0]
    assert with_negative[1] < positive_ratio[1]
