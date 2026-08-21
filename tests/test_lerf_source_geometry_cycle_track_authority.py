import torch

from radio_gs.scripts.build_lerf_source_geometry_cycle_track_authority import (
    reciprocal_cycle_labels,
)


def test_reciprocal_positive_cycle_is_same() -> None:
    strength = torch.tensor([[0.0, 0.8, 0.2], [0.8, 0.0, 0.0], [0.2, 0.0, 0.0]])
    labels = reciprocal_cycle_labels(
        strength, torch.tensor([0, 1, 1]), (strength > 0).int(), torch.ones(3, 3)
    )
    assert labels[0, 1] == labels[1, 0] == 1
    assert labels[0, 2] == -1


def test_visible_zero_overlap_is_different_but_occluded_is_unknown() -> None:
    strength = torch.zeros(2, 2); intersection = torch.zeros(2, 2, dtype=torch.int32)
    visible = reciprocal_cycle_labels(strength, torch.tensor([0, 1]), intersection, torch.ones(2, 2))
    occluded_visibility = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    occluded = reciprocal_cycle_labels(strength, torch.tensor([0, 1]), intersection, occluded_visibility)
    assert visible[0, 1] == 0
    assert occluded[0, 1] == -1
