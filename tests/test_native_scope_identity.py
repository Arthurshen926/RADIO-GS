import numpy as np
import pytest
from radio_gs.utils.native_scope_identity import native_scope_identity


def key(coarse,native):
    return native_scope_identity(coarse,np.packbits(native,axis=-1),native.shape)


def test_equal_coarse_masks_do_not_merge_different_native_scopes():
    coarse=np.array([[True,False]])
    a=np.array([[1,1,0,0],[1,1,0,0]],dtype=bool);b=a.copy();b[0,2]=1
    assert key(coarse,a)!=key(coarse,b)
    assert key(coarse,a)==key(coarse,a.copy())
    assert key(coarse,a)!=key(~coarse,a)


def test_pixel_identity_ignores_padding_and_validates_shape():
    coarse=np.array([[True]]);packed=np.array([[128]],dtype=np.uint8)
    assert native_scope_identity(coarse,packed,(1,1))==native_scope_identity(coarse,packed|127,(1,1))
    with pytest.raises(ValueError):native_scope_identity(coarse,packed,(2,1))
