"""Transfer a matched reference intervention while retaining the incumbent."""
import numpy as np


def transfer_reference_delta(base, before, after, space='probability'):
    """Apply only the change induced by replacing a reference response.

    Pairing fixes target image, memory model and preprocessing. This identifies
    an intervention response, not a calibrated causal effect on GT accuracy.
    An exactly unchanged teacher response is an exact no-op in either space.
    """
    b, p, q = [np.asarray(v, dtype=np.float32) for v in (base, before, after)]
    if b.ndim != 2 or any(x.shape != b.shape or not np.isfinite(x).all() or
            x.min() < 0 or x.max() > 1 for x in [b, p, q]):
        raise ValueError('finite aligned probabilities required')
    if space == 'probability':
        result = np.clip(b + (q-p), 0, 1)
    elif space == 'log_odds':
        def logit(x):
            x = np.clip(x, 1e-4, 1-1e-4).astype(np.float64)
            return np.log(x) - np.log1p(-x)
        z = logit(b) + logit(q) - logit(p)
        result = (1/(1+np.exp(-z))).astype(np.float32)
    else:
        raise ValueError('unknown residual space')
    result = np.where(q == p, b, result).astype(np.float32)
    return result, dict(space=space, changed_teacher_pixels=int((p != q).sum()),
        updated_pixels=int((result != b).sum()), positive_changes=int((result > b).sum()),
        negative_changes=int((result < b).sum()), maximum_abs_change=float(np.abs(result-b).max()),
        unaltered_teacher_exact_noop=bool(np.array_equal(result[p == q], b[p == q])))
