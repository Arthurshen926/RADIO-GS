import pytest
import torch

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _calibrate_threshold,
    build_training_pairs,
    track_augmented_training_targets,
    visible_membership_target,
)


def test_training_pairs_treat_invisible_rows_as_unknown():
    supports = [torch.tensor([0, 2]), torch.tensor([3])]
    proposal_views = torch.tensor([0, 1])
    visible = torch.tensor(
        [[True, True, True, False, False], [False, False, False, True, True]]
    )
    proposal, row, label, weight = build_training_pairs(
        supports=supports,
        proposal_views=proposal_views,
        view_observed=visible,
        selected=torch.tensor([True, True]),
        positive_cap=8,
        negative_cap=8,
        seed=1,
    )
    pairs = set(zip(proposal.tolist(), row.tolist(), label.tolist()))
    assert pairs == {(0, 0, 1.0), (0, 2, 1.0), (0, 1, 0.0), (1, 3, 1.0), (1, 4, 0.0)}
    for item in (0, 1):
        selected = proposal == item
        assert torch.isclose(weight[selected & (label == 1)].sum(), torch.tensor(0.5))
        assert torch.isclose(weight[selected & (label == 0)].sum(), torch.tensor(0.5))


def test_training_pairs_preserve_soft_exact_mpr_targets():
    proposal, row, label, _weight = build_training_pairs(
        supports=[torch.tensor([0, 2])],
        proposal_views=torch.tensor([0]),
        view_observed=torch.tensor([[True, True, True]]),
        selected=torch.tensor([True]),
        positive_cap=8,
        negative_cap=8,
        seed=1,
        support_probabilities=[torch.tensor([0.25, 0.8])],
    )
    labels = {(int(p), int(r)): float(y) for p, r, y in zip(proposal, row, label)}
    assert labels[(0, 0)] == 0.25
    assert labels[(0, 2)] == pytest.approx(0.8)
    assert labels[(0, 1)] == 0.0


def test_threshold_calibration_maximizes_macro_iou():
    values = [
        (torch.tensor([0.9, 0.8, 0.1]), torch.tensor([True, True, False])),
        (torch.tensor([0.7, 0.2]), torch.tensor([True, False])),
    ]
    threshold, iou = _calibrate_threshold(values, 32)
    assert 0.2 < threshold <= 0.7
    assert iou == 1.0


def test_visible_membership_target_uses_the_dense_carrier_domain():
    visible = torch.tensor([7, 1, 9, 4, 2])
    support = torch.tensor([2, 7, 8])
    assert torch.equal(
        visible_membership_target(visible, support, num_rows=10),
        torch.tensor([True, False, False, False, True]),
    )


def test_track_augmentation_uses_training_views_only():
    rows = torch.tensor([0, 1, 0, 1, 2])
    proposals = torch.tensor([0, 0, 1, 1, 2])
    supports = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2])]
    augmented, observed, stats = track_augmented_training_targets(
        rows=rows,
        proposals=proposals,
        weights=torch.ones(rows.numel()),
        supports=supports,
        proposal_views=torch.tensor([0, 1, 2]),
        view_observed=torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, False, True, True],
            ]
        ),
        training=torch.tensor([True, True, False]),
        minimum_soft_cosine=0.5,
    )
    assert stats["num_tracks"] == 1
    assert torch.equal(augmented[0], torch.tensor([0, 1]))
    assert torch.equal(augmented[1], torch.tensor([0, 1]))
    assert torch.equal(augmented[2], torch.tensor([2]))
    assert torch.equal(observed[0], torch.tensor([True, True, True, False]))
    assert torch.equal(observed[2], torch.tensor([False, False, True, True]))
