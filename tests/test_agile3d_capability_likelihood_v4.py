from __future__ import annotations

import torch

from radio_gs.querying.query_likelihood_head import (
    MonotoneSignedLikelihoodRatioHead,
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


def test_signed_null_cosine_is_strictly_neutral_without_trainable_bias() -> None:
    head = MonotoneSignedLikelihoodRatioHead(affinity_channel_count=2)
    null = torch.full((11, 1, 2), 0.5)
    likelihood = head.log_likelihood_ratio(
        _inputs(null, torch.full((11, 1, 2), 0.5))
    )
    assert torch.equal(likelihood, torch.zeros(11))
    assert "bias" not in dict(head.named_parameters())
    probability = head(
        _inputs(torch.empty((11, 0, 2)), torch.empty((11, 0, 2))),
        source="synthetic_null",
    ).foreground_probability
    assert torch.equal(probability, torch.full((11,), 0.5))


def test_far_capability_lookalike_receives_same_positive_likelihood() -> None:
    head = MonotoneSignedLikelihoodRatioHead(affinity_channel_count=2)
    positive = torch.tensor(
        [
            [[0.99, 0.95]],
            [[0.99, 0.95]],
            [[0.10, 0.20]],
        ]
    )
    probability = head(
        _inputs(positive, torch.empty((3, 0, 2))), source="synthetic_far"
    ).foreground_probability
    assert torch.equal(probability[0], probability[1])
    assert probability[1] > 0.5
    assert probability[2] < 0.5


def test_negative_click_monotonically_suppresses_a_lookalike() -> None:
    head = MonotoneSignedLikelihoodRatioHead(affinity_channel_count=2)
    positive = torch.tensor([[[0.95, 0.90]], [[0.85, 0.80]]])
    before = head(
        _inputs(positive, torch.empty((2, 0, 2))), source="synthetic_negative"
    ).foreground_probability
    after = head(
        _inputs(positive, torch.tensor([[[0.20, 0.20]], [[0.95, 0.90]]])),
        source="synthetic_negative",
    ).foreground_probability
    assert after[1] < before[1]
    assert after[0] > after[1]
