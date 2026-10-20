import numpy as np
import pytest
from radio_gs.querying.response_crop import response_crop_box, restore_response_crop


def test_non_square_window_contains_extent_with_context_and_clips_edges():
    prior=np.zeros((8,20),np.float32);prior[2:6,13:19]=.8
    assert response_crop_box(prior)==(11,1,20,7)
    prior[:]=0;prior[0,0]=1
    assert response_crop_box(prior)==(0,0,2,2)
    prior[:]=0
    assert response_crop_box(prior) is None


def test_local_restore_preserves_every_outside_probability():
    prior=np.linspace(0,1,80,dtype=np.float32).reshape(8,10)
    box=(2,1,7,4);local=np.full((3,5),.75,np.float32)
    result=restore_response_crop(prior,local,box)
    inside=np.zeros_like(prior,dtype=bool);inside[1:4,2:7]=True
    assert np.array_equal(result[~inside],prior[~inside])
    assert np.array_equal(result[inside],local.ravel())
    assert np.array_equal(prior,np.linspace(0,1,80,dtype=np.float32).reshape(8,10))


def test_bad_coordinates_and_nonfinite_evidence_are_rejected():
    with pytest.raises(ValueError): response_crop_box(np.full((2,4),np.nan))
    with pytest.raises(ValueError): restore_response_crop(np.zeros((3,8)),np.ones((2,4)),(1,1,3,5))
