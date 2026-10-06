import numpy as np
import pytest
import torch
from radio_gs.training.small_region_inventory import discarded_small_region_ids,original_native_gray_crop


def test_area_cutoff_restores_only_nonempty_old_exclusions():
    masks=torch.zeros((5,96,128),dtype=torch.bool)
    for i,n in enumerate([0,1,24,25,8000]):masks[i].flatten()[:n]=True
    assert discarded_small_region_ids(masks).tolist()==[1,2]
    assert [i for i,m in enumerate(masks) if .002<=float(m.float().mean())<=.6]==[3]


def test_native_crop_uses_native_detail_and_gray_background_without_mutation():
    rgb=np.zeros((9,9,3),np.uint8);rgb[4,4]=[240,30,20]
    mask=np.zeros((9,9),bool);mask[4,4]=True;old=rgb.copy()
    crop=np.asarray(original_native_gray_crop(rgb,mask))
    assert crop.shape==(384,384,3)
    assert np.all(crop[0,0]==127)
    assert crop[192,192,0]>200 and crop[192,192,1]<60
    assert np.array_equal(rgb,old)


def test_empty_and_misaligned_native_masks_rejected():
    rgb=np.zeros((4,4,3),np.uint8)
    with pytest.raises(ValueError):original_native_gray_crop(rgb,np.zeros((4,4),bool))
    with pytest.raises(ValueError):original_native_gray_crop(rgb,np.ones((3,4),bool))
    with pytest.raises(ValueError):discarded_small_region_ids(torch.ones(3,4,4))
