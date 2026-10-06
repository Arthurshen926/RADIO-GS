"""Regional summaries of deployed, visibility-normalized pixel responses."""
import torch


def region_render_weights(masks, pixel_ids, element_ids, contributions,
                          observed, *, minimum_coverage=0.5):
    """Adjoint weights for mean rendered cosine, never P(mask|element).

    The returned R x N matrix pools element scalars exactly as rendering
    those scalars then averaging covered pixels within each source mask.
    Unsupported regions have zero weights and are explicitly reported.
    """
    if masks.ndim != 2 or observed.ndim != 1:
        raise ValueError('Expected region/pixel masks and element support')
    if not torch.isfinite(contributions).all() or (contributions < 0).any():
        raise ValueError('Nonnegative finite rendering contributions required')
    pixels = masks.shape[1]
    pixel_mass = contributions.new_zeros(pixels).scatter_add_(0, pixel_ids, contributions)
    observed_values = contributions * observed[element_ids]
    observed_mass = contributions.new_zeros(pixels).scatter_add_(0, pixel_ids, observed_values)
    coverage = observed_mass / pixel_mass.clamp_min(1e-8)
    valid = (coverage > minimum_coverage) & (pixel_mass > 1e-8)
    values = observed_values / observed_mass[pixel_ids].clamp_min(1e-8)
    operator = torch.sparse_coo_tensor(torch.stack([pixel_ids, element_ids]), values,
                                      (pixels, len(observed))).coalesce()
    supported_masks = masks.float() * valid[None]
    count = supported_masks.sum(1)
    pixel_weights = supported_masks / count[:, None].clamp_min(1)
    weights = torch.sparse.mm(operator.T, pixel_weights.T).T
    return weights, count > 0, valid
