import torch

from radio_gs.scripts.evaluate_lerf_source_conservative_gaussian_core import (
    reciprocal_core_score,
    triangle_core_score,
    wilson_lower_bound,
)


def test_wilson_lower_bound_rewards_repeated_support() -> None:
    value = wilson_lower_bound(torch.tensor([1.0, 10.0]), torch.tensor([1.0, 10.0]))
    assert 0 < value[0] < value[1] < 1


def test_triangle_core_keeps_complete_three_view_cycle_only() -> None:
    views = torch.tensor([0, 1, 2, 3])
    probability = torch.full((4, 4), 0.1); probability.fill_diagonal_(1)
    geometry = torch.ones(4, 4)
    for left, right in ((0, 1), (1, 2), (2, 0)):
        probability[left, right] = probability[right, left] = 0.9
    score = triangle_core_score(probability, geometry, views)
    assert score[0, 1] > 0
    assert score[0, 3] == 0


def test_reciprocal_core_keeps_pair_without_fabricating_third_view() -> None:
    views = torch.tensor([0, 1, 2])
    probability = torch.tensor([[1.0, 0.8, 0.1], [0.8, 1.0, 0.1], [0.1, 0.1, 1.0]])
    geometry = torch.full((3, 3), 0.6)
    score = reciprocal_core_score(probability, geometry, views)
    assert torch.isclose(score[0, 1], torch.tensor(0.6))
    assert score[0, 2] == 0
