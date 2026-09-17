"""Exact L1 feasibility diagnostic for a shared scalar element posterior."""
import torch


def mixture_error_floor(positive_mass, negative_mass):
    """Minimize sum a_i(1-z_i)+b_i*z_i over independent z_i in [0,1].

    Masses aggregate normalized rendering contributions from explicitly labelled
    pixels, optionally across views of the same query. This is a relaxation of
    a learned cosine field, not an IoU ceiling or an attainable feature model.
    """
    if positive_mass.shape != negative_mass.shape or positive_mass.ndim != 1:
        raise ValueError('element mass axes differ')
    if any(not torch.isfinite(x).all() or (x < 0).any() for x in (positive_mass, negative_mass)):
        raise ValueError('label masses must be finite and nonnegative')
    total = float((positive_mass + negative_mass).sum())
    unavoidable = float(torch.minimum(positive_mass, negative_mass).sum())
    return {'minimum_absolute_error_sum': unavoidable,
            'label_contribution_mass': total,
            'minimum_mean_absolute_error': unavoidable / total if total else None,
            'conflicting_elements': int(((positive_mass > 0) & (negative_mass > 0)).sum())}
