"""Separate disconnected appearance supports; never assert instance identity."""
import numpy as np
import torch
from scipy.ndimage import label


def connected_source_regions(mask, minimum_pixels=8):
    if mask.ndim!=2 or mask.dtype!=torch.bool or minimum_pixels<1:
        raise ValueError('source component mask must be 2D bool')
    labels,count=label(mask.cpu().numpy(),structure=np.ones((3,3),dtype=np.uint8))
    return [torch.from_numpy(labels==i) for i in range(1,count+1) if int((labels==i).sum())>=minimum_pixels]
