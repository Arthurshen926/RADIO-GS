"""Fixed-iteration source-view geometric median; no query-dependent weights."""
import torch


def robust_view_fusion(vectors, masses, *, excluded_view=None, iterations=5, epsilon=1e-4):
    if vectors.ndim != 3 or masses.shape != vectors.shape[:2]:
        raise ValueError('Expected view, element, feature axes')
    if not torch.isfinite(vectors).all() or not torch.isfinite(masses).all() or (masses < 0).any():
        raise ValueError('Finite vectors and nonnegative observation masses required')
    if iterations < 0 or epsilon <= 0:
        raise ValueError('Invalid bounded solver configuration')
    weights=masses.clone()
    if excluded_view is not None:weights[excluded_view]=0
    mass=weights.sum(0)
    estimate=(vectors*weights[...,None]).sum(0)/mass[:,None].clamp_min(1e-8)
    for _ in range(iterations):
        distance=(vectors-estimate[None]).square().sum(-1).sqrt().clamp_min(epsilon)
        adjusted=weights/distance
        estimate=(vectors*adjusted[...,None]).sum(0)/adjusted.sum(0)[:,None].clamp_min(1e-8)
    return estimate,mass>1e-8
