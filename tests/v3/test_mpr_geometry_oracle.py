import torch

from radio_gs.v3.evaluation.mpr_geometry_oracle import (
    exact_mpr_membership,
    optimize_same_view_posterior,
    render_exact_membership,
    union_memberships,
)
from radio_gs.v3.training.instance_upper_bound import MaskEpisode


def _episode(target: list[bool]) -> MaskEpisode:
    return MaskEpisode(
        proposal_index=0,
        view_index=0,
        gaussian_ids=torch.tensor([0, 1, 2]),
        pixel_ids=torch.tensor([0, 0, 1]),
        contribution_weights=torch.tensor([0.6, 0.4, 1.0]),
        target=torch.tensor([target]),
        known=torch.ones(1, 2, dtype=torch.bool),
        boundary=torch.ones(1, 2, dtype=torch.bool),
        unknown=torch.zeros(1, 2, dtype=torch.bool),
        scale=0.5,
    )


def test_exact_mpr_roundtrip_uses_continuous_unthresholded_membership():
    episode = _episode([True, False])
    lifted = exact_mpr_membership(episode, 3)
    torch.testing.assert_close(lifted.probability, torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(render_exact_membership(lifted, episode), torch.tensor([1.0, 0.0]))


def test_positive_union_does_not_turn_unobserved_rows_into_negative_evidence():
    left = exact_mpr_membership(_episode([True, False]), 3)
    right = exact_mpr_membership(_episode([False, True]), 3)
    merged = union_memberships([left, right])
    torch.testing.assert_close(merged.probability, torch.ones(3))
    assert merged.observed.all()


def test_target_driven_posterior_oracle_improves_a_misaligned_initialization():
    episode = _episode([True, False])
    initial = exact_mpr_membership(_episode([False, True]), 3)
    before = render_exact_membership(initial, episode)
    dense, _loss = optimize_same_view_posterior(
        initial, episode, device=torch.device("cpu"), steps=40, learning_rate=0.2
    )
    after = render_exact_membership(
        type(initial)(dense, initial.observed, initial.semantic_mass), episode
    )
    target = episode.target.flatten().float()
    assert torch.mean((after - target) ** 2) < torch.mean(
        (before - target) ** 2
    )
