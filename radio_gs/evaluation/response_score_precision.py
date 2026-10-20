"""Numerical diagnostics; never replace a retained benchmark's score silently."""
import cv2
import numpy as np
from scipy.special import expit


def stable_probability_readout(logits, wh):
    """Rank a bilinear mixture of sigmoid probabilities without 1-p cancellation.

    Interpolating logits would define a different response. Here both positive
    and complementary probabilities are interpolated independently in float64.
    Their log ratio has the mixture's ordering even when p rounds to one.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError('finite two-dimensional logits required')
    if len(wh) != 2 or any(type(x) is not int or x <= 0 for x in wh):
        raise ValueError('positive integer width and height required')
    p = cv2.resize(expit(logits), tuple(wh), interpolation=cv2.INTER_LINEAR)
    q = cv2.resize(expit(-logits), tuple(wh), interpolation=cv2.INTER_LINEAR)
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError('logits exceed float64 tail capacity; do not clip silently')
    return p, np.log(p) - np.log(q)


def saturation_counts(probability, truth):
    p, g = np.asarray(probability), np.asarray(truth, dtype=bool)
    if p.shape != g.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('finite aligned probability and truth required')
    return {name: dict(pixels=int(mask.sum()), tp=int((mask & g).sum()),
                       fp=int((mask & ~g).sum()))
            for name, mask in [('zero', p == 0), ('one', p == 1)]}
