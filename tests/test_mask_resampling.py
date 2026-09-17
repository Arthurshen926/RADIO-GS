import pytest
import torch
from torch.nn import functional as F
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact


@pytest.mark.parametrize('source,target', [((19,31),(7,12)),((7,12),(19,31)),((16,24),(8,12)),((1,3),(8,2)),((11,13),(11,13))])
def test_boolean_sampling_matches_torch_nearest_exact(source,target):
    mask=torch.rand(3,*source)>.5
    reference=F.interpolate(mask[:,None].float(),size=target,mode='nearest-exact')[:,0].bool()
    assert torch.equal(resize_boolean_masks_nearest_exact(mask,target),reference)


def test_empty_region_axis_and_bad_dtype():
    assert resize_boolean_masks_nearest_exact(torch.empty(0,10,20,dtype=torch.bool),(3,7)).shape==(0,3,7)
    with pytest.raises(ValueError):resize_boolean_masks_nearest_exact(torch.ones(1,3,4),(2,2))
