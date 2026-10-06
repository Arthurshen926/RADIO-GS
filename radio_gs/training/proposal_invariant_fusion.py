"""Count identical per-view mask/feature observations once during semantic writes.

Pixel mass conservation alone does not prevent a repeated whole-object proposal
from gaining extra semantic votes against an overlapping part proposal. Only
exact duplicate evidence is collapsed here; distinct crops or views remain
independent observations.
"""
import numpy as np
import torch

from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights


def inverse_evidence_multiplicity(masks, descriptors):
    if masks.ndim != 2 or descriptors.ndim != 2 or len(masks) != len(descriptors):
        raise ValueError('Mask and native descriptor rows must match')
    if not torch.isfinite(masks).all() or not ((masks == 0) | (masks == 1)).all():
        raise ValueError('Binary finite source masks required')
    if not torch.isfinite(descriptors).all() or (descriptors.norm(dim=1) == 0).any():
        raise ValueError('Finite nonzero native descriptors required')
    bits = np.packbits(masks.detach().bool().cpu().numpy(), axis=1)
    values = descriptors.detach().contiguous().cpu().numpy()
    groups, identifiers = {}, []
    for mask, descriptor in zip(bits, values):
        key = (mask.tobytes(), descriptor.tobytes())
        identifiers.append(key)
        groups[key] = groups.get(key, 0) + 1
    return torch.tensor([1. / groups[key] for key in identifiers], device=masks.device, dtype=torch.float32)


def proposal_invariant_pixel_weights(masks, descriptors, pixel_ids, element_ids, contributions, num_elements):
    multiplicity = inverse_evidence_multiplicity(masks, descriptors)
    weights, mass = source_pixel_evidence_weights(masks.float()*multiplicity[:, None],
        pixel_ids, element_ids, contributions, num_elements, partition_overlap=True)
    return weights, mass, multiplicity
