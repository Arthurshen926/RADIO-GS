from __future__ import annotations

import torch

from radio_gs.v4.completion.fragment_noise import fragment_set_observation_noise
from radio_gs.v4.completion.oracle import PartialObjectMembership


def _partial() -> PartialObjectMembership:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, -1])
    return PartialObjectMembership.from_oracle_visibility(
        labels, torch.ones(9, dtype=torch.bool), token_count=2
    )


def test_fragment_noise_is_deterministic_and_only_drops_observed_positives():
    centres = torch.arange(27).reshape(9, 3).float()
    first, receipt = fragment_set_observation_noise(
        _partial(), centres, scene_id="scene", seed=7,
        minimum_keep_fraction=0.5, maximum_keep_fraction=0.5,
        maximum_fragments=2,
    )
    second, _ = fragment_set_observation_noise(
        _partial(), centres, scene_id="scene", seed=7,
        minimum_keep_fraction=0.5, maximum_keep_fraction=0.5,
        maximum_fragments=2,
    )
    assert torch.equal(first.positive, second.positive)
    assert bool((first.positive <= _partial().positive).all())
    assert torch.equal(first.negative, _partial().negative)
    assert first.positive.sum(0).tolist() == [2, 2]
    assert receipt["positive_invented"] is False
    assert receipt["negative_changed"] is False


def test_fragment_noise_moves_only_dropped_positive_pairs_to_unknown():
    partial = _partial()
    noisy, _ = fragment_set_observation_noise(
        partial, torch.arange(27).reshape(9, 3).float(),
        scene_id="scene", seed=3, minimum_keep_fraction=0.5,
        maximum_keep_fraction=0.5, maximum_fragments=1,
    )
    dropped = partial.positive & ~noisy.positive
    assert torch.equal(noisy.unknown, partial.unknown | dropped)
    assert not bool((noisy.positive & noisy.unknown).any())
