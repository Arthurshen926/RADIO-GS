import numpy as np
import pytest
import torch
from PIL import Image

from scripts.research.compile_query_field_native_region_appearance import bound_masks


def test_two_stage_proposal_binding_and_permutation_guard():
    masks = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=np.uint8)
    sam = dict(packed_masks=np.packbits(masks, axis=-1, bitorder='little'), mask_shape=[2, 2])
    region = dict(native_proposal_indices=torch.tensor([2, 0, 1]))
    raw = torch.from_numpy(masks[[2, 0, 1]]).bool()
    assert torch.equal(bound_masks(region, sam, [1, 0], raw), torch.from_numpy(masks[[0, 2]]))
    with pytest.raises(ValueError, match='binding changed'):
        bound_masks(region, sam, [1, 0], raw.flip(0))
    with pytest.raises(ValueError, match='region IDs'):
        bound_masks(region, sam, [0, 0], raw)


def test_native_pooling_preserves_token_magnitude_and_excludes_padding():
    from radio_gs.models.native_region_appearance import NativeRegionAppearance
    teacher = NativeRegionAppearance.__new__(NativeRegionAppearance)
    teacher.device = torch.device('cpu'); teacher.patch = 14
    # Last patch has one real column: its large token magnitude must count
    # with 1/14 support. Per-token L2 or padded-area pooling both fail here.
    teacher.patches = lambda image: (torch.tensor([[10., 0.], [0., 14.]]), None)
    image = Image.new('RGB', (15, 14))
    masks = np.ones((2, 14, 15), dtype=np.float32); masks[1, :, :14] = 0
    expected = torch.nn.functional.normalize(torch.tensor([[10., 1.], [0., 1.]]), dim=1)
    torch.testing.assert_close(teacher.encode(image, masks), expected)
