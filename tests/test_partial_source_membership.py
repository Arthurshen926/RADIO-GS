import torch
from radio_gs.training.partial_source_membership import corroborated_membership_masks


def test_conflict_never_becomes_negative_by_majority_vote():
    p = torch.tensor([[1., 1., 0.], [0., 1., 0.], [0., .4, .4]])
    pos, neg, conflict = corroborated_membership_masks(p, torch.ones_like(p))
    assert pos.tolist() == [False, True, False]
    assert neg.tolist() == [False, False, True]
    assert conflict.tolist() == [True, False, False]


def test_unseen_and_single_view_are_not_corroboration():
    p = torch.tensor([[1., 0.], [float('nan'), float('nan')]])
    mass = torch.tensor([[1., 1.], [0., 0.]])
    pos, neg, conflict = corroborated_membership_masks(p, mass)
    assert not (pos | neg | conflict).any()
