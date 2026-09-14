import torch
from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.evaluation.selected_surface_render import render_selected_surface_support


def test_selected_only_removes_unselected_occluders_without_mutating_scene():
    carrier = SurfaceVoxelCarrier(torch.tensor([[0., 0., 2.], [0., 0., 4.]]), .1,
        maximum_splat_radius=0, surface_band_voxels=0, maximum_contributors_per_pixel=1)
    camera = Camera("test", torch.tensor([[10., 0., 2.], [0., 10., 2.], [0., 0., 1.]]), torch.eye(4), 5, 5)
    posterior = torch.tensor([0., 1.])
    assert not (carrier.render_posterior(posterior, camera) >= .5).any()
    selected = render_selected_surface_support(carrier, posterior, camera, threshold=.5)
    assert selected[2, 2] and selected.sum() == 1
    assert carrier.num_elements == 2
    assert not render_selected_surface_support(carrier, torch.zeros(2), camera, threshold=.5).any()
