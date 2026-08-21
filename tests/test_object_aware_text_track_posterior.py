import torch

from radio_gs.querying.object_aware_text_track_posterior import (
    object_aware_text_track_posterior,
)


def _run(seed_probability: torch.Tensor, seed_valid: torch.Tensor, affinity: float = 0.9):
    base = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.1, 0.1]])
    language = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    return object_aware_text_track_posterior(
        base, seed_probability, seed_valid, language,
        torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 1, 1]),
        torch.ones(4), torch.tensor([0, 1]), None,
        torch.tensor([0]), torch.tensor([1]), torch.tensor([affinity]), None,
        same_threshold=0.5,
    )


def test_query_permutation_does_not_change_affinity_membership() -> None:
    probability = torch.tensor([[0.9, 0.1], [0.4, 0.8]])
    valid = torch.tensor([[True, False], [False, True]])
    original = _run(probability, valid)
    permuted = _run(probability[:, [1, 0]], valid[:, [1, 0]])
    assert torch.equal(original.selected_membership[:, [1, 0]], permuted.selected_membership)


def test_no_seed_replays_v1_bitwise() -> None:
    result = _run(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool))
    expected = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.1, 0.1]])
    assert torch.equal(result.probability, expected)
    assert bool(result.fallback.all())


def test_isolated_seed_replays_v1_bitwise() -> None:
    result = _run(torch.tensor([[0.9, 0.1], [0.1, 0.9]]), torch.ones(2, 2, dtype=torch.bool), affinity=0.1)
    expected = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.1, 0.1]])
    assert torch.equal(result.probability, expected)
    assert bool(result.fallback.all())


def test_track_expansion_is_seed_direct_one_hop_only() -> None:
    base = torch.tensor([[0.8], [0.2], [0.1]])
    result = object_aware_text_track_posterior(
        base, torch.tensor([[0.9], [0.1], [0.1]]),
        torch.tensor([[True], [False], [False]]), torch.full((3, 1), 0.8),
        torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 1, 2]), torch.ones(4),
        torch.tensor([0, 1, 2]), None, torch.tensor([0, 1]), torch.tensor([1, 2]),
        torch.tensor([0.9, 0.9]), None, same_threshold=0.5,
    )
    assert result.selected_membership[:, 0].tolist() == [True, True, False]


def test_single_view_positive_is_unknown_and_replays_v1() -> None:
    base = torch.tensor([[0.8], [0.2]])
    result = object_aware_text_track_posterior(
        base, torch.tensor([[0.9], [0.1]]), torch.tensor([[True], [False]]),
        torch.full((2, 1), 0.8), torch.tensor([0, 1]), torch.tensor([0, 1]),
        torch.ones(2), torch.tensor([0, 1]), None,
        torch.tensor([0]), torch.tensor([1]), torch.tensor([0.9]), None,
        same_threshold=0.5,
    )
    assert torch.equal(result.probability, base)
