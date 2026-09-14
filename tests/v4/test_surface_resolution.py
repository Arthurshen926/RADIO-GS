import torch
from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.contracts.surface_scene_bundle import SurfaceCarrierConfiguration


def make_carrier(reference=None):
    return SurfaceVoxelCarrier(torch.tensor([[0., 0., 2.], [0., 0., 4.]]), .2,
        maximum_splat_radius=1, surface_band_voxels=.5, maximum_contributors_per_pixel=8,
        reference_raster_shape=reference)


def camera(scale_x=1, scale_y=1):
    return Camera("same", torch.tensor([[10.*scale_x, 0., 16.*scale_x],
        [0., 10.*scale_y, 16.*scale_y], [0., 0., 1.]]), torch.eye(4), 32*scale_y, 32*scale_x)


def test_reference_projection_preserves_source_and_depth_occlusion():
    old, fixed = make_carrier(), make_carrier((32, 32))
    a, b = old.project(camera()), fixed.project(camera())
    for key in ("element_ids", "pixel_ids", "weights", "depths"):
        assert torch.equal(getattr(a, key), getattr(b, key))
    large = fixed.project(camera(4, 4))
    assert set(large.element_ids.tolist()) == {0}
    assert large.num_pixels == 128*128
    assert len(large.pixel_ids) >= 45
    assert len(old.project(camera(4, 4)).pixel_ids) == 5


def test_nonuniform_resize_scales_footprint_axes_separately():
    projection = make_carrier((32, 32)).project(camera(4, 2))
    x, y = projection.pixel_ids % 128, projection.pixel_ids // 128
    assert int(x.max() - x.min()) == 8
    assert int(y.max() - y.min()) == 4


def test_serialized_reference_scale_survives_reload_and_legacy_identity():
    old = dict(voxel_size=.2, maximum_splat_radius=1, surface_band_voxels=.5,
               maximum_contributors_per_pixel=8, camera_convention="colmap_world_opencv_camera_feature_raster")
    assert SurfaceCarrierConfiguration.from_dict(old).to_dict() == old
    config = SurfaceCarrierConfiguration.from_dict({**old, "reference_raster_shape": [32, 32]})
    restored = SurfaceCarrierConfiguration.from_dict(config.to_dict())
    carrier = restored.build_carrier(torch.tensor([[0., 0., 2.]]), normals=None, confidence=torch.ones(1))
    assert len(carrier.project(camera(4, 4)).pixel_ids) >= 45
