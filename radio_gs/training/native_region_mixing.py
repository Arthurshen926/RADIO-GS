"""Explicit mixing of overlapping source mask descriptors."""
import torch


def mix_region_descriptors(masks, descriptors, *, inverse_area=False):
    if masks.ndim!=2 or descriptors.ndim!=2 or len(masks)!=len(descriptors):
        raise ValueError('region descriptor axes differ')
    weights=masks.float()
    if not torch.isfinite(descriptors).all() or not torch.isfinite(weights).all() or ((weights<0)|(weights>1)).any():raise ValueError('invalid region evidence')
    if inverse_area:weights=weights/weights.sum(1,keepdim=True).clamp_min(1)
    mass=weights.sum(0)
    return weights.T@descriptors/mass[:,None].clamp_min(1e-8),mass>0
