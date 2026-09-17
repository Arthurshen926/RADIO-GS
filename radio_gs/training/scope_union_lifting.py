"""Lift a same-view pixel union; max after projection loses shared support."""
import torch
from radio_gs.training.region_visibility import normalize_visible_region_mass


def lift_mask_union(pixel_to_element, masks):
    if masks.ndim < 2 or masks.shape[0] == 0:
        raise ValueError('A nonempty mask group is required')
    union = masks.bool().flatten(1).any(0).float()
    if pixel_to_element.shape[0] != union.numel():
        raise ValueError('Mask pixels and projection rows differ')
    mass = torch.sparse.sum(pixel_to_element, dim=0).to_dense()
    probability, seen = normalize_visible_region_mass(
        torch.sparse.mm(pixel_to_element.T, union[:, None]), mass)
    return probability[:, 0], seen
