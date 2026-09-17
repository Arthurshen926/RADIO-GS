"""Lift query-independent source evidence without counting overlapping masks twice."""
import torch


def source_pixel_evidence_weights(masks, pixel_ids, element_ids, contributions,
                                  num_elements, *, partition_overlap):
    if masks.ndim != 2 or not torch.isfinite(contributions).all() or (contributions < 0).any():
        raise ValueError('Invalid source mask/contribution axes or values')
    pixels = masks.shape[1]
    mass = contributions.new_zeros(pixels).scatter_add_(0, pixel_ids, contributions)
    values = contributions / mass[pixel_ids].clamp_min(1e-8)
    operator = torch.sparse_coo_tensor(torch.stack([element_ids, pixel_ids]), values,
                                       (num_elements, pixels)).coalesce()
    visible_mass = torch.sparse.sum(operator, dim=1).to_dense()
    targets = masks.float()
    if partition_overlap:
        targets = targets / targets.sum(0).clamp_min(1)[None]
    lifted = torch.sparse.mm(operator, targets.T).T
    lifted[:, visible_mass <= 1e-6] = 0
    return lifted, visible_mass
