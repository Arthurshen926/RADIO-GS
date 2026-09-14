from types import SimpleNamespace
import pytest
import torch
from radio_gs.v4.geometry.fuse_lerf_moge3 import _scaled_camera


@pytest.mark.parametrize("stride", [1, 2, 4, 7])
def test_sliced_depth_camera_preserves_original_pixel_rays(stride):
    view = SimpleNamespace(intrinsic=torch.tensor([[30., 0., 10.], [0., 40., 8.], [0., 0., 1.]]),
                           width=23, height=19, frame_index=1, camera_to_world=torch.eye(4))
    full = _scaled_camera(view, 19, 23, 1)
    sampled = _scaled_camera(view, 19, 23, stride)
    for axis, length in ((0, 23), (1, 19)):
        original_pixels = torch.arange(0, length, stride).double()
        sample_pixels = torch.arange(original_pixels.numel()).double()
        expected = (original_pixels + .5 - full.intrinsic[axis, 2]) / full.intrinsic[axis, axis]
        actual = (sample_pixels + .5 - sampled.intrinsic[axis, 2]) / sampled.intrinsic[axis, axis]
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
