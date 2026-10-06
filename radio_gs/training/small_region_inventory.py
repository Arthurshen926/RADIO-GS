"""Restore existing query-free SAM regions lost only to the coarse area cutoff."""
import numpy as np
from PIL import Image
import torch


def discarded_small_region_ids(resized_masks):
    masks=torch.as_tensor(resized_masks)
    if masks.dtype!=torch.bool or masks.ndim!=3 or min(masks.shape[1:])<1:
        raise ValueError('Boolean nonempty raster stack required')
    area=masks.float().flatten(1).mean(1)
    return torch.where((area>0)&(area<.002))[0]


def original_native_gray_crop(rgb,mask):
    """Exact native bbox±3, gray127,384BICUBIC recipe used by old teachers."""
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or mask.dtype!=bool or mask.shape!=rgb.shape[:2] or not mask.any():
        raise ValueError('Native uint8 RGB and aligned nonempty Boolean mask required')
    ys,xs=np.where(mask);h,w=mask.shape
    x0,x1=max(0,int(xs.min())-3),min(w,int(xs.max())+4)
    y0,y1=max(0,int(ys.min())-3),min(h,int(ys.max())+4)
    values=rgb[y0:y1,x0:x1].copy();values[~mask[y0:y1,x0:x1]]=127
    return Image.fromarray(values).resize((384,384),Image.Resampling.BICUBIC)
