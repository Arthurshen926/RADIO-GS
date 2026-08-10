import torch

from radio_gs.querying.source_heldout_missing_support import (
    FEATURE_NAMES,
    calibration_free_maximin_selection,
    heldout_missing_support_label,
    proposal_feature_vector,
)


def _mass() -> tuple[torch.Tensor, torch.Tensor]:
    training = torch.zeros(6, 3)
    heldout = torch.zeros(6, 3)
    training[0:2, 1] = 2.0
    heldout[2:4, 1] = torch.tensor([3.0, 2.0])
    heldout[2:4, 2] = torch.tensor([0.25, 0.50])
    heldout[4:6, 2] = 2.0
    return training, heldout


def test_heldout_label_uses_training_seed_and_missing_target_only():
    training, heldout = _mass()
    label = heldout_missing_support_label(
        target_selected_scale_scores=torch.tensor([0.2, 0.6, 0.9, 0.8]),
        target_rows=torch.tensor([2, 3, 4, 5]),
        seed_rows=torch.tensor([0, 1]),
        training_primitive_instance_mass=training,
        heldout_primitive_instance_mass=heldout,
    )
    assert label.evaluable
    assert label.seed_instance_id == 1
    assert label.missing_primitive_count == 2
    assert label.hard_positive
    assert label.heldout_target_mass == 5.0
    assert label.heldout_visible_mass == 5.75
    assert label.signed_utility > 0.0


def test_heldout_label_is_negative_for_other_instance_and_fail_closed_when_empty():
    training, heldout = _mass()
    negative = heldout_missing_support_label(
        target_selected_scale_scores=torch.tensor([0.9, 0.8, 0.1, 0.2]),
        target_rows=torch.tensor([2, 3, 4, 5]),
        seed_rows=torch.tensor([0, 1]),
        training_primitive_instance_mass=training,
        heldout_primitive_instance_mass=heldout,
    )
    assert negative.evaluable
    assert not negative.hard_positive
    assert negative.signed_utility == -1.0
    empty = heldout_missing_support_label(
        target_selected_scale_scores=torch.ones(4),
        target_rows=torch.tensor([2, 3, 4, 5]),
        seed_rows=torch.tensor([0, 1]),
        training_primitive_instance_mass=training,
        heldout_primitive_instance_mass=heldout,
    )
    assert not empty.evaluable


def test_proposal_feature_vector_exposes_required_axes():
    feature = proposal_feature_vector(
        edge_comembership_reliability=0.9,
        source_observation_count=3.0,
        source_observation_agreement=0.8,
        target_selected_scale_scores=torch.tensor([0.9, 0.7, 0.4, 0.2]),
        target_anchor_score=0.9,
        seed_median_score=0.8,
        target_visibility=0.75,
    )
    assert feature.shape == (len(FEATURE_NAMES),)
    assert torch.isfinite(feature).all()
    assert feature[4] == 0.9
    assert feature[7] == 0.5
    assert feature[11] > 0.0


def test_calibration_free_rank_is_group_local_and_tie_deterministic():
    base = torch.tensor(
        [0.9, 4.0, 0.9, 0.8, 0.9, 0.6, 0.5, 0.5, 0.6, 4.0, 0.8, 0.6]
    )
    weak = base.clone()
    weak[[0, 1, 2, 3, 10]] = torch.tensor([0.2, 1.0, 0.2, 0.2, 0.2])
    features = torch.stack((weak, base, base, weak))
    selected, score = calibration_free_maximin_selection(
        features,
        group_ids=torch.tensor([0, 0, 1, 1]),
        target_region_indices=torch.tensor([5, 7, 3, 4]),
    )
    assert selected.tolist() == [False, True, True, False]
    assert score[1] > score[0]
    assert score[2] > score[3]

