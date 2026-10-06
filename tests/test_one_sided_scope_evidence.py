import torch
from radio_gs.training.one_sided_scope_evidence import positive_scope_evidence


def test_weak_foreground_is_unknown_not_certain_or_negative():
    n = torch.tensor([0., .01, .5, .74, .8, 1.])
    num, den = positive_scope_evidence(n, torch.ones_like(n))
    assert torch.equal(den[:4], torch.zeros(4))
    assert torch.allclose(num[4:] / den[4:], n[4:])
    assert torch.equal(n, torch.tensor([0., .01, .5, .74, .8, 1.]))


def test_preserves_fraction_under_visibility_scaling_without_fake_confidence():
    n = torch.tensor([.8, .008, 0.])
    d = torch.tensor([1., .01, 0.])
    num, den = positive_scope_evidence(n, d)
    assert torch.allclose(num[:2]/den[:2], torch.tensor([.8, .8]))
    assert torch.allclose(den[0]/den[1], torch.tensor(100.))
    assert num[2] == 0 and den[2] == 0
