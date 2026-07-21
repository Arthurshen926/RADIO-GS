import math

import torch

from radio_gs.interfaces.scale_ordered_relation import (
    accumulate_scale_ordered_votes,
    logarithmic_scale_bin_edges,
    merge_scale_intervals,
    robust_mask_physical_radius,
)


def test_physical_mask_radius_is_robust_and_metric() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radius = robust_mask_physical_radius(xyz, torch.ones(3), minimum_primitives=3)
    assert radius == 1.0
    assert math.isnan(robust_mask_physical_radius(xyz, torch.tensor([1.0, 0.0, 0.0])))


def test_scale_ordered_votes_preserve_small_separate_and_large_same() -> None:
    edge = torch.tensor([[0, 1], [1, 2]])
    # At 0.1 m, primitive 0 separates from 1.  At 1 m they share a broader
    # object mask, yielding a valid ordered interval rather than a conflict.
    small = torch.tensor([[1.0, 0.0, 0.0]])
    large = torch.tensor([[1.0, 1.0, 0.0]])
    bins = torch.log(torch.tensor([0.05, 0.20, 0.50, 2.0]))
    votes = accumulate_scale_ordered_votes(
        [small, large], [torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        [torch.tensor([0.10]), torch.tensor([1.0])], edge, bins,
    )
    result = merge_scale_intervals(votes)
    assert votes["separate_votes"][0].sum() > 0
    assert votes["same_votes"][0].sum() > 0
    assert bool(result["has_lower"][0]) and bool(result["has_upper"][0])
    assert bool(result["interval_consistent"][0])
    assert result["lower_log_radius"][0] < result["upper_log_radius"][0]
    # No positive can erase the second edge's small-scale separating evidence.
    assert votes["separate_votes"][1].sum() > 0


def test_logarithmic_scale_edges_are_valid() -> None:
    edges = logarithmic_scale_bin_edges(minimum_radius_m=0.1, maximum_radius_m=1.6, bins=4)
    assert tuple(edges.shape) == (5,)
    assert bool((edges[1:] > edges[:-1]).all())


def test_same_only_votes_do_not_fabricate_track_exterior_constraints() -> None:
    votes = accumulate_scale_ordered_votes(
        [torch.tensor([[1.0, 0.0]])],
        [torch.ones(2, dtype=torch.bool)],
        [torch.tensor([0.5])],
        torch.tensor([[0], [1]]),
        torch.log(torch.tensor([0.1, 1.0])),
        include_separate=False,
    )
    assert votes["same_votes"].sum() == 0
    assert votes["separate_votes"].sum() == 0


def test_per_mask_observation_batch_is_exactly_equivalent_to_independent_tracks() -> None:
    edge = torch.tensor([[0, 1], [1, 2]])
    bins = torch.log(torch.tensor([0.1, 1.0]))
    first = torch.tensor([[1.0, 1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0, 1.0]])
    first_observed = torch.tensor([True, True, False])
    second_observed = torch.tensor([False, True, True])
    independent = accumulate_scale_ordered_votes(
        [first, second], [first_observed, second_observed],
        [torch.tensor([0.5]), torch.tensor([0.5])], edge, bins, mask_chunk=1,
    )
    packed = accumulate_scale_ordered_votes(
        [torch.cat([first, second], dim=0)],
        [torch.stack([first_observed, second_observed], dim=0)],
        [torch.tensor([0.5, 0.5])], edge, bins, mask_chunk=1,
    )
    for key in ("same_votes", "separate_votes", "observed_votes", "same_events", "separate_events"):
        assert torch.equal(independent[key], packed[key])
