from __future__ import annotations

import torch

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    click_gaussian_mixture_weights,
    scene_centered_capability_affinity,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneQueryLikelihoodHead,
    QueryLikelihoodInputs,
)
from radio_gs.scripts.train_capability_query_likelihood_head import (
    balanced_bce_soft_dice_loss,
)


def _observations(positive: torch.Tensor, negative: torch.Tensor) -> QueryLikelihoodInputs:
    rows = positive.shape[0]
    return QueryLikelihoodInputs(
        positive_affinity=positive,
        negative_affinity=negative,
        prior_probability=torch.full((rows,), 0.5),
        coverage=torch.ones(rows),
        reliability=torch.ones(rows),
    )


def test_far_primitive_with_same_capability_gets_positive_global_affinity() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    affinity, report = scene_centered_capability_affinity(
        features,
        click_candidate_indices=torch.tensor([[0]]),
        click_mixture_weights=torch.ones((1, 1)),
        chunk_size=2,
    )
    assert float(affinity[1, 0]) > 0.99
    assert float(affinity[1, 0]) > float(affinity[2, 0])
    assert report["uses_labels"] is False
    assert report["uses_query_threshold"] is False


def test_negative_capability_click_suppresses_an_appearance_lookalike() -> None:
    appearance = torch.tensor(
        [[1.0, 0.0], [1.0, 0.01], [-1.0, 0.0], [0.0, 1.0]]
    )
    boundary = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    candidates = torch.tensor([[0], [1]])
    weights = torch.ones((2, 1))
    appearance_affinity, _ = scene_centered_capability_affinity(
        appearance,
        click_candidate_indices=candidates,
        click_mixture_weights=weights,
    )
    boundary_affinity, _ = scene_centered_capability_affinity(
        boundary,
        click_candidate_indices=candidates,
        click_mixture_weights=weights,
    )
    affinity = torch.stack((appearance_affinity, boundary_affinity), dim=-1)
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=2)
    positive_only = head(
        _observations(affinity[:, :1], torch.empty((4, 0, 2))),
        source="synthetic",
    ).foreground_probability
    with_negative = head(
        _observations(affinity[:, :1], affinity[:, 1:2]),
        source="synthetic",
    ).foreground_probability
    assert with_negative[1] < positive_only[1]


def test_balanced_bce_dice_does_not_collapse_one_percent_foreground() -> None:
    rows = 100
    positive = torch.full((rows, 1, 2), 0.05)
    positive[0, 0] = torch.tensor([0.99, 0.99])
    target = torch.zeros(rows)
    target[0] = 1
    observations = _observations(positive, torch.empty((rows, 0, 2)))
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        probability = head(observations, source="synthetic").foreground_probability
        loss, _details = balanced_bce_soft_dice_loss(probability, target)
        loss.backward()
        optimizer.step()
    probability = head(observations, source="synthetic").foreground_probability
    prediction = probability >= 0.5
    assert bool(prediction[0])
    assert not bool(prediction[1:].any())


def test_click_gaussian_mixture_log_normalization_survives_underflow() -> None:
    weights = click_gaussian_mixture_weights(
        primitive_xyz=torch.tensor([[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]]),
        primitive_covariance=torch.eye(3).mul(1e-4).repeat(2, 1, 1),
        primitive_opacity=torch.tensor([0.7, 0.4]),
        click_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        click_candidate_indices=torch.tensor([[0, 1]]),
    )
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert float(weights[0, 0]) > float(weights[0, 1])
