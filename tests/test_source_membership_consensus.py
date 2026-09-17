import torch
from radio_gs.training.source_membership_consensus import source_membership_consensus


def test_weak_contradictions_do_not_outvote_clear_observation():
    p = torch.tensor([[1.], [0.], [0.], [0.]])
    mass = torch.tensor([[1.], [.001], [.001], [.001]])
    result, known = source_membership_consensus(p, mass)
    assert known.all() and result.item() > .99
    assert p.mean().item() < .5


def test_splitting_observation_preserves_evidence_and_missing_is_unknown():
    p = torch.tensor([[.8, float('nan')], [.2, float('nan')]])
    mass = torch.tensor([[2., 0.], [1., 0.]])
    expected, known = source_membership_consensus(p, mass)
    divided = p[[0, 0, 1]]
    splitmass = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    actual, actual_known = source_membership_consensus(divided, splitmass)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(known, actual_known) and not known[1] and actual[1] == 0


def test_equal_evidence_matches_view_mean():
    p = torch.tensor([[.1, .6], [.7, .8]])
    result, _ = source_membership_consensus(p, torch.ones_like(p))
    torch.testing.assert_close(result, p.mean(0))
