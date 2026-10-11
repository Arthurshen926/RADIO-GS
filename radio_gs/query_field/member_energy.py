"""Experimental discrete member inference; no fitted query parameters.

Consumes field-owned sparse optical coverage and independently compiled footprint
costs. This functional reference is deliberately not wired into default queries:
its supplied factors determine the extent it can protect.
"""
import numpy as np
from scipy import sparse


def solve_member_energy(matrix, prior, positive, negative, reliability, footprint,
                        *, cutoff, support_weight=.1, prior_weight=.1, max_passes=8):
    """Bounded deterministic coordinate descent on one binary Gaussian set.

    E = weighted positive deficit + weighted negative excess (each class
    normalized separately) + fixed-prior-normalized footprint + prior disagreement.
    Coverage is capped at the decision threshold in the data terms. Unknown and
    zero-reliability rows never request changes; untouched members retain prior.
    The footprint denominator is frozen before inference, not recomputed per step.
    This is a local discrete solution, not a global optimum or extent guarantee.
    """
    matrix = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    rows, count = matrix.shape
    prior = np.asarray(prior, dtype=np.float64)
    positive, negative = np.asarray(positive), np.asarray(negative)
    reliability = np.asarray(reliability, dtype=np.float64)
    footprint = np.asarray(footprint, dtype=np.float64)
    if prior.shape != (count,) or footprint.shape != (count,):
        raise ValueError('invalid member domain')
    if any(x.shape != (rows,) for x in (positive, negative, reliability)):
        raise ValueError('invalid factor domain')
    if positive.dtype != bool or negative.dtype != bool or np.any(positive & negative):
        raise ValueError('invalid factor labels')
    if any(not np.isfinite(x).all() for x in (matrix.data, prior, reliability, footprint)):
        raise ValueError('nonfinite evidence')
    if (matrix.data < 0).any() or (footprint < 0).any() or (reliability < 0).any() or (reliability > 1).any() or (prior < 0).any() or (prior > 1).any():
        raise ValueError('evidence outside valid range')
    if not np.isfinite([cutoff, support_weight, prior_weight]).all() or cutoff <= 0 or min(support_weight, prior_weight) < 0 or not isinstance(max_passes, int) or max_passes < 1:
        raise ValueError('invalid solver policy')
    initial = prior > .5
    selected = initial.copy()
    # Normalize by label counts, not confidence sums: uniformly weak evidence
    # must remain weak relative to the fixed prior and support penalties.
    p = reliability * positive / max(int(positive.sum()), 1)
    n = reliability * negative / max(int(negative.sum()), 1)
    active = (p + n) > 0
    touched = np.asarray(matrix.T @ active.astype(float)).ravel() > 0
    denominator = float(footprint @ prior)
    # Empty/zero-footprint priors cannot define the v10 fixed-prior ratio.
    # Disable that term explicitly instead of dividing by epsilon.
    cost = footprint / denominator if denominator > 0 else np.zeros(count)
    disagreement = np.abs(2 * prior - 1) / max(int(touched.sum()), 1)
    coverage = np.asarray(matrix @ selected.astype(float)).ravel()

    def data_loss(values, indexes):
        covered = np.minimum(values / cutoff, 1.)
        return p[indexes] * (1. - covered) + n[indexes] * covered

    def energy():
        return float(data_loss(coverage, np.arange(rows)).sum() +
                     support_weight * (cost @ selected) +
                     prior_weight * (disagreement @ (selected != initial)))

    trace = [energy()]; flips = 0; passes = 0
    for _ in range(max_passes):
        changed = 0; passes += 1
        # Stable row ID order; no query-specific order fitting or threshold tuning.
        for g in np.flatnonzero(touched):
            start, stop = matrix.indptr[g:g+2]
            indexes = matrix.indices[start:stop]
            sign = -1. if selected[g] else 1.
            updated = np.maximum(coverage[indexes] + sign * matrix.data[start:stop], 0.)
            delta = float((data_loss(updated, indexes) - data_loss(coverage[indexes], indexes)).sum())
            delta += support_weight * sign * cost[g]
            delta += prior_weight * disagreement[g] * (1 if selected[g] == initial[g] else -1)
            if delta < -1e-12:
                selected[g] = ~selected[g]; coverage[indexes] = updated
                changed += 1; flips += 1
        trace.append(energy())
        if not changed: break
    return selected, dict(energy_trace=trace, passes=passes, flips=flips,
                         added=int((selected & ~initial).sum()),
                         removed=int((initial & ~selected).sum()),
                         converged=changed == 0, footprint_denominator=denominator,
                         footprint_enabled=denominator > 0,
                         query_parameter_fitting=False, full_extent_guaranteed=False)
