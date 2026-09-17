"""Exact nearest pixel-centre sampling without expanding Boolean masks to fp32."""
import torch


def resize_boolean_masks_nearest_exact(masks, size):
    if masks.ndim != 3 or masks.dtype != torch.bool:
        raise ValueError('Boolean [N,H,W] masks required')
    height, width = size
    if height < 1 or width < 1 or min(masks.shape[-2:]) < 1:
        raise ValueError('positive raster dimensions required')
    # Integer form of floor((destination_index + .5) * source / destination).
    # It agrees with nearest-exact without allocating N*H*W float32 values.
    y = ((2*torch.arange(height, device=masks.device)+1)*masks.shape[-2])//(2*height)
    x = ((2*torch.arange(width, device=masks.device)+1)*masks.shape[-1])//(2*width)
    return masks[:, y[:, None], x[None, :]]
