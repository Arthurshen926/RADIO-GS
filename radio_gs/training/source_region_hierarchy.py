"""Source-view mask containment, without inventing cross-view instance IDs.

These relations are observational: containment does not establish part identity.
The two exclusive masks are diagnostic controls, not replacement teacher masks.
"""
import torch


def source_region_hierarchy(masks, views, *, containment=0.95, max_area_ratio=0.8):
    """Return directed child->parent relations only within the same source view.

    Equal-size/duplicate masks and merely intersecting masks are not a hierarchy.
    All strict ancestors are retained, so deleting overlaps handles deep nesting.
    """
    if masks.ndim != 3 or masks.dtype != torch.bool or views.shape != (len(masks),):
        raise ValueError('expected boolean R,H,W masks and R view identities')
    if not 0 < containment <= 1 or not 0 < max_area_ratio < 1:
        raise ValueError('invalid containment or area ratio')
    flat = masks.flatten(1).float()
    area = flat.sum(1)
    intersection = flat @ flat.T
    same_view = views.to(masks.device)[:, None] == views.to(masks.device)[None, :]
    child_parent = (same_view & (area[:, None] > 0)
                    & (area[:, None] <= max_area_ratio * area[None, :])
                    & (intersection >= containment * area[:, None]))
    return child_parent


def exclusive_scope_masks(masks, child_parent, *, scope):
    """Remove only overlapping pixels of established ancestors/descendants.

    A fine control retains the smallest supported region at nested pixels; a
    coarse control retains the largest. Partial overlaps with no containment
    remain independent. No descriptor is recomputed on these synthetic masks:
    that deliberate limitation isolates assignment from teacher recognition.
    """
    if scope not in ('fine', 'coarse'):
        raise ValueError('scope must be fine or coarse')
    if masks.dtype != torch.bool or child_parent.dtype != torch.bool:
        raise ValueError('masks and relations must be boolean')
    if masks.ndim != 3 or child_parent.shape != (len(masks), len(masks)):
        raise ValueError('mask/relation axes differ')
    result = masks.clone()
    suppressors = child_parent.T if scope == 'fine' else child_parent
    for i in range(len(masks)):
        selected = suppressors[i]
        if selected.any():
            result[i] &= ~masks[selected].any(0)
    return result
