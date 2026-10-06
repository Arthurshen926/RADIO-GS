"""Preserve absolute rendered evidence mass when writing source semantics."""
import torch


def alpha_weighted_source_evidence(masks,pixel_ids,element_ids,contributions,num_elements):
    """Lift overlap-partitioned masks with raw T*alpha instead of T*alpha/A.

    The original normalized-ray visibility predicate remains fixed to isolate
    only absolute source evidence weighting. These are writing weights, not
    conditional membership labels or a changed deployment renderer.
    """
    if masks.ndim!=2 or contributions.ndim!=1 or pixel_ids.shape!=contributions.shape or element_ids.shape!=contributions.shape:
        raise ValueError('Source evidence axes differ')
    if not torch.isfinite(contributions).all() or (contributions<0).any():
        raise ValueError('Nonnegative finite rendering contribution required')
    if not torch.isfinite(masks).all() or not ((masks==0)|(masks==1)).all():
        raise ValueError('Binary finite source masks required')
    pixels=masks.shape[1]
    pixel_mass=contributions.new_zeros(pixels).scatter_add_(0,pixel_ids,contributions)
    normalized=contributions/pixel_mass[pixel_ids].clamp_min(1e-8)
    seen=contributions.new_zeros(num_elements).scatter_add_(0,element_ids,normalized)>1e-6
    operator=torch.sparse_coo_tensor(torch.stack([element_ids,pixel_ids]),contributions,(num_elements,pixels)).coalesce()
    partition=masks.float()/masks.sum(0).clamp_min(1)[None]
    lifted=torch.sparse.mm(operator,partition.T).T
    lifted[:,~seen]=0
    return lifted,pixel_mass
