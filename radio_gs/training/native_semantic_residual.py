"""An orthogonal native semantic extension that never rotates the base field."""
import math

import torch
from torch.nn import functional as F


def fit_native_residual(descriptors, queries, base, dimensions, query_weight=0.):
    if not math.isfinite(query_weight) or query_weight < 0:
        raise ValueError('Invalid query weight')
    for value in (descriptors, queries, base):
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError('Finite matrices required')
    if descriptors.shape[1] != base.shape[1] or queries.shape[1] != base.shape[1]:
        raise ValueError('Native axis mismatch')
    if not 0 < dimensions <= base.shape[1] - len(base):
        raise ValueError('Residual rank exceeds native complement')
    if not torch.allclose(base @ base.T, torch.eye(len(base), device=base.device), atol=1e-4):
        raise ValueError('Base must be orthonormal')
    x, q = F.normalize(descriptors.float(), dim=-1), F.normalize(queries.float(), dim=-1)
    if not len(x) or not len(q) or (descriptors.norm(dim=1) == 0).any() or (queries.norm(dim=1) == 0).any():
        raise ValueError('Nonempty nonzero evidence required')
    x = x - (x @ base.T) @ base
    q = q - (q @ base.T) @ base
    covariance = x.T @ x / len(x) + query_weight * (q.T @ q / len(q))
    values, vectors = torch.linalg.eigh(covariance)
    if values[-dimensions] <= 1e-9:
        raise ValueError('Source evidence does not support requested residual rank')
    residual = vectors[:, -dimensions:].flip(1).T
    # Remove numerical base leakage and reorthogonalize the small extension.
    residual = residual - (residual @ base.T) @ base
    residual = torch.linalg.qr(residual.T, mode='reduced').Q.T.contiguous()
    return residual


def native_residual_cosine(base_elements, residual_elements, queries, base, residual):
    """Exact cosine in the jointly decoded space; no independent query norm."""
    if base_elements.shape[0] != residual_elements.shape[0]:
        raise ValueError('Element identities must match')
    coords = torch.cat([base_elements, residual_elements], dim=1)
    basis = torch.cat([base, residual], dim=0)
    if coords.shape[1] != len(basis):
        raise ValueError('Coordinate dimension mismatch')
    return F.normalize(coords, dim=-1) @ (F.normalize(queries, dim=-1) @ basis.T).T
