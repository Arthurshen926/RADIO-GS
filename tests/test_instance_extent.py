import numpy as np
import pytest
import torch
from radio_gs.querying.instance_extent import scribble_fields, admit_component_additions


def test_unmarked_reference_mass_is_unknown_and_contribution_order_is_irrelevant():
    pid = torch.tensor([0, 1, 2, 2]); gid = torch.tensor([0, 0, 0, 1])
    w = torch.tensor([.2, .3, .5, .7]); p = torch.tensor([True, False, False]); n = torch.tensor([False, True, False])
    value, receipt = scribble_fields(pid, gid, w, p, n, 3)
    torch.testing.assert_close(value, torch.tensor([[.2, .3], [0., 0.], [0., 0.]]))
    order = torch.tensor([3, 0, 2, 1])
    other, _ = scribble_fields(pid[order], gid[order], w[order], p, n, 3)
    torch.testing.assert_close(value, other)
    assert not receipt['unknown_is_negative']
    with pytest.raises(ValueError): scribble_fields(pid, gid, w, p, p, 3)


def test_identity_vote_recovers_extent_and_rejects_another_instance():
    base = np.zeros((5, 8), np.float32); base[1, 1] = .9
    proposal = np.zeros_like(base); proposal[1:4, 1:3] = .8; proposal[1:4, 5:7] = .9
    positive = np.zeros_like(base); negative = np.zeros_like(base)
    positive[1, 1] = .2; negative[2, 5] = .3
    value, receipt = admit_component_additions(base, proposal, positive, negative)
    assert np.all(value[1:4, 1:3] >= .8) and not value[:, 5:].any()
    assert value[1, 1] == base[1, 1] and receipt['admitted_components'] == 1
    assert receipt['added_foreground_pixels'] == 5


def test_unknown_or_conflicting_identity_preserves_incumbent():
    base = np.array([[.9, .2], [.1, .3]], np.float32); proposal = np.ones_like(base)
    zero = np.zeros_like(base)
    for positive, negative in [(zero, zero), (proposal, proposal), (zero, proposal)]:
        value, receipt = admit_component_additions(base, proposal, positive, negative)
        np.testing.assert_array_equal(value, base)
        assert receipt['admitted_components'] == 0
