"""Admit complete image regions using signed, visibility-transported identity.

Unmarked reference pixels have zero authority. A region vote licenses only
positive additions; it does not turn missing evidence into background.
"""
import hashlib
import cv2
import numpy as np
import torch


def scribble_fields(pixel_ids, gaussian_ids, weights, positive, negative, count):
    """Alpha-weighted signed coverage per Gaussian, including unknown mass."""
    p, n = torch.as_tensor(positive), torch.as_tensor(negative)
    if p.dtype != torch.bool or n.dtype != torch.bool or p.shape != n.shape or (p & n).any():
        raise ValueError('disjoint boolean reference scribbles required')
    if weights.ndim != 1 or weights.shape != pixel_ids.shape or weights.shape != gaussian_ids.shape:
        raise ValueError('contribution dimensions differ')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite nonnegative visibility weights required')
    p, n = p.flatten().to(weights.device), n.flatten().to(weights.device)
    mass = weights.new_zeros(count).scatter_add_(0, gaussian_ids, weights)
    values = []
    for sign in (p, n):
        numerator = weights.new_zeros(count).scatter_add_(0, gaussian_ids, weights * sign[pixel_ids])
        values.append(torch.where(mass > 1e-8, numerator / mass.clamp_min(1e-8), 0))
    field = torch.stack(values, 1)
    return field, dict(visible_gaussians=int((mass > 1e-8).sum()),
        positive_gaussians=int((field[:, 0] > 0).sum()), negative_gaussians=int((field[:, 1] > 0).sum()),
        maximum_signed_mass=float(field.sum(1).max()), unknown_is_negative=False)


def admit_component_additions(base, proposal, positive, negative):
    """Accept 8-connected proposal components with more support than exclusion.

    Evidence is summed in float64. No evidence, a tie or net negative evidence
    retains the original response. Acceptance never reduces an image score.
    Downstream 3D fitting is evaluated separately; it is not guaranteed monotone.
    """
    arrays = [np.asarray(x, dtype=np.float32) for x in (base, proposal, positive, negative)]
    b, q, p, n = arrays
    if b.ndim != 2 or any(x.shape != b.shape or not np.isfinite(x).all() or
                           x.min() < 0 or x.max() > 1 + 1e-6 for x in arrays):
        raise ValueError('finite aligned probability/evidence maps required')
    count, labels = cv2.connectedComponents((q >= .5).astype(np.uint8), connectivity=8)
    pm = np.bincount(labels.ravel(), weights=p.ravel().astype(np.float64), minlength=count)
    nm = np.bincount(labels.ravel(), weights=n.ravel().astype(np.float64), minlength=count)
    accepted = (pm > nm) & (pm > 1e-8); accepted[0] = False
    take = accepted[labels]
    output = np.where(take, np.maximum(b, q), b)
    assert np.all(output >= b)
    return output, dict(components=count-1, admitted_components=int(accepted.sum()),
        unknown_components=int(((pm[1:] <= 1e-8) & (nm[1:] <= 1e-8)).sum()),
        admitted_pixels=int(take.sum()), added_foreground_pixels=int(((output >= .5) & (b < .5)).sum()),
        positive_evidence=float(pm[1:].sum()), negative_evidence=float(nm[1:].sum()),
        component_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        decision_sha256=hashlib.sha256(accepted.tobytes()).hexdigest(), image_monotone=True)
