import numpy as np
import pytest
import torch

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
