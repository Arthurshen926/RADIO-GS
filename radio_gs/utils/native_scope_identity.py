"""A coarse-mask collision is not proof that two native teacher scopes agree."""
import hashlib
import numpy as np


def native_scope_identity(coarse_mask, packed_native, native_shape):
    coarse=np.asarray(coarse_mask,dtype=np.bool_)
    packed=np.asarray(packed_native,dtype=np.uint8)
    height,width=map(int,native_shape)
    if coarse.ndim!=2 or min(height,width)<=0 or packed.shape!=(height,(width+7)//8):
        raise ValueError('Native mask shape and packed representation differ')
    # Ignore unused padding bits so identity describes pixels, not storage junk.
    packed=packed.copy()
    if width%8:packed[:,-1]&=np.uint8((255<<(8-width%8))&255)
    digest=hashlib.sha256(coarse.tobytes()+packed.tobytes()).hexdigest()
    return (tuple(coarse.shape),(height,width),digest)
