import pytest
import torch
from radio_gs.training.geometry_relation_weight import geometry_negative_weights


def test_geometry_does_not_invent_negatives_and_is_coordinate_invariant():
    xyz = torch.tensor([[0., 0, 0], [.01, 0, 0], [.01, 0, 0], [10., 0, 0]])
    normals = torch.tensor([[0., 0, 1], [1., 0, 0], [1., 0, 0], [1., 0, 0]])
    rows = torch.arange(4)
    negative = torch.tensor([False, False, True, True])
    w = geometry_negative_weights(xyz, normals, rows, 0, negative, 1.)
    assert w[0] == w[1] == 1
    assert w[2] > 2 and w[3] == 1
    transformed = geometry_negative_weights(xyz * 7 + 12, -normals, rows, 0, negative, 7.)
    torch.testing.assert_close(w, transformed)


def test_unknown_normals_do_not_add_evidence():
    xyz = torch.zeros(2, 3)
    normals = torch.tensor([[0., 0, 1], [float('nan'), 0, 0]])
    assert torch.equal(geometry_negative_weights(xyz, normals, torch.arange(2), 0, torch.ones(2, dtype=torch.bool), 1.), torch.ones(2))
    with pytest.raises(ValueError):
        geometry_negative_weights(xyz, normals, torch.arange(2), 0, torch.ones(2, dtype=torch.bool), 0.)
