import numpy as np
import pytest
import torch
from radio_gs.geometry.registered_mesh_carrier import build_registered_mesh_carrier


def test_area_controls_tangent_footprint_and_vertex_identity():
    vertices = np.array([[0, 0, 0], [.03, 0, 0], [0, .03, 0], [9, 9, 9]])
    model, ids = build_registered_mesh_carrier(vertices, np.array([[0, 1, 2]]))
    assert ids.tolist() == [0, 1, 2]
    expected = np.sqrt((.03*.03/2/3)/np.pi)
    assert torch.allclose(model.scales[:, :2], torch.full((3, 2), expected))
    assert torch.allclose(model.normals, torch.tensor([[0., 0., 1.]]).expand(3, 3))
    assert not any(x.requires_grad for x in model.buffers())


def test_rigid_translation_does_not_change_surface_footprint():
    v = np.array([[0, 0, 0], [.04, 0, 0], [0, .04, 0]])
    f = np.array([[0, 1, 2]])
    a, _ = build_registered_mesh_carrier(v, f)
    b, _ = build_registered_mesh_carrier(v+[1, -2, 3], f)
    assert torch.allclose(a.scales, b.scales, atol=1e-7)
    assert torch.allclose(a.quaternions, b.quaternions)


@pytest.mark.parametrize('faces', [np.array([[0, 1, 9]]), np.array([[0., 1., 2.]]), np.array([[0, 0, 0]])])
def test_invalid_or_degenerate_surface_fails_closed(faces):
    with pytest.raises(ValueError):
        build_registered_mesh_carrier(np.eye(3), faces)
