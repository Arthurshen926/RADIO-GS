"""Source-only orthogonal compression with balanced native query directions.

The decoder and query use the same basis. No independently learned query
adapter, labels or evaluation queries enter this fit.
"""
import math

import torch
from torch.nn import functional as F


def _unit(x):
    if x.ndim != 2 or not x.numel() or not torch.isfinite(x).all() or (x.norm(dim=1) <= 0).any():
        raise ValueError('Finite nonzero native vectors required')
    return F.normalize(x.detach().float(), dim=-1)


def fit_query_preserving_basis(descriptors, text_queries, dimensions=64, query_weight=1.):
    x, q = _unit(descriptors), _unit(text_queries)
    if x.shape[1] != q.shape[1] or not 1 <= dimensions <= min(x.shape):
        raise ValueError('Incompatible native axes or rank')
    if not math.isfinite(query_weight) or query_weight < 0:
        raise ValueError('Invalid query covariance weight')
    # Normalize each population by its size, so additional overlapping source
    # proposals cannot silently change the image/text objective ratio.
    covariance = x.T @ x / len(x)
    if query_weight:
        covariance = covariance + query_weight * (q.T @ q / len(q))
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[:, -dimensions:].flip(1).T.contiguous()


def native_response_diagnostic(descriptors, queries, basis):
    x, q = _unit(descriptors), _unit(queries)
    if basis.ndim != 2 or basis.shape[1] != x.shape[1] or q.shape[1] != x.shape[1]:
        raise ValueError('Incompatible native response axes')
    if not torch.allclose(basis @ basis.T, torch.eye(len(basis), device=basis.device), atol=1e-4):
        raise ValueError('An orthonormal shared basis is required')
    truth = x @ q.T
    z = x @ basis.T
    # Normalize the decoded element, never the projected query: the latter
    # changes the native teacher's metric and category ordering.
    predicted = F.normalize(z, dim=-1) @ (q @ basis.T).T
    error = predicted - truth
    return dict(mse=float(error.square().mean()),
                centered_mse=float((error-error.mean(dim=1, keepdim=True)).square().mean()),
                top1_agreement=float((truth.argmax(1) == predicted.argmax(1)).float().mean()),
                max_absolute_error=float(error.abs().max()))
