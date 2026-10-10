"""Construction-time exclusion permission for an accepted region union.

This is a stronger, explicit hypothesis than intersecting partial negatives:
interiors of comparable regions disjoint from the entire accepted union may
exclude it. Missing/background pixels remain unknown. It requires correctly
accepted scopes, not merely semantic resemblance. Not a deployed field decoder.
"""
import torch
from .observations import erode


def union_exclusion(masks, scales, accepted, visible, previous):
    if masks.ndim != 3 or masks.dtype != torch.bool:
        raise ValueError('expected boolean region rasters')
    if scales.shape != (len(masks),) or accepted.shape != scales.shape or accepted.dtype != torch.bool:
        raise ValueError('region metadata must align')
    if visible.shape != masks.shape[1:] or previous.shape != visible.shape:
        raise ValueError('pixel permissions must align')
    if visible.dtype != torch.bool or previous.dtype != torch.bool:
        raise ValueError('pixel permissions must be boolean')
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError('region scales must be finite and positive')
    if not bool(accepted.any()):
        if bool(previous.any()):raise ValueError('empty union cannot authorize exclusion')
        return torch.zeros_like(previous), torch.zeros_like(accepted)
    union=masks[accepted].any(0)
    if bool((previous & (union | ~visible)).any()):
        raise ValueError('previous exclusion conflicts with accepted scope or visibility')
    disjoint=~(masks & union).flatten(1).any(1)
    ratio=scales[:,None]/scales[accepted][None,:].clamp_min(1e-6)
    comparable=((ratio>=.5)&(ratio<=2)).any(1)
    permitted=disjoint & comparable
    extra=erode(masks[permitted]).any(0) if bool(permitted.any()) else torch.zeros_like(union)
    return previous | (extra & visible & ~union), permitted
