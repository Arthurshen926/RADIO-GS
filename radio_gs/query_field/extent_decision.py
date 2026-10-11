"""Bounded binary coordinate decisions using complete compiled optical factors.

This is a functional migration candidate, not the default query path. Its scope
input must be supplied by a separately validated query adapter. The research
runner supplies privileged fixed scopes and reports that limitation explicitly.
No learned per-query parameters, gradients, renderer, or source access here.
"""
import math
import numpy as np
from numba import njit
from scipy import sparse

CUTOFF = 10.5 / 255


@njit(cache=True)
def render_row(pointer, index, alpha, selected, row, flip=-1):
    transmittance = np.float32(1.)
    for k in range(pointer[row], pointer[row+1]):
        g = index[k]
        if selected[g] == (g == flip): continue
        a = min(alpha[k], np.float32(.999))
        if a < 1/255: continue
        next_t = np.float32(transmittance * np.float32(np.float32(1)-a))
        if next_t <= 1e-4: break
        transmittance = next_t
    return np.float32(1)-transmittance


@njit(cache=True)
def render_binary(pointer, index, alpha, selected):
    result = np.zeros(len(pointer)-1, np.float32)
    for row in range(len(result)):
        result[row] = render_row(pointer, index, alpha, selected, row)
    return result


@njit(cache=True)
def render_membership(pointer, index, visibility, opacity, member):
    result = np.zeros(len(pointer)-1, np.float32)
    for row in range(len(result)):
        t = np.float32(1.)
        for k in range(pointer[row], pointer[row+1]):
            g = index[k]
            a = min(np.float32(.999), np.float32(np.float32(opacity[g]*member[g])*visibility[k]))
            if a < np.float32(1/255): continue
            next_t = np.float32(t*np.float32(np.float32(1)-a))
            if next_t <= 1e-4: break
            t = next_t
        result[row] = np.float32(1)-t
    return result


@njit(cache=True)
def probability(alpha):
    # Identical decision-alpha transform and BCE clamp to the canonical solver.
    a = min(max(alpha, 1e-6), 1-1e-6)
    return a * (1-CUTOFF) / (a*(1-CUTOFF) + CUTOFF*(1-a))


@njit(cache=True)
def binary_cross_entropy(p, target):
    p = min(max(p, 1e-5), 1-1e-5)
    return -target * math.log(p) - (1-target) * math.log1p(-p)


@njit(cache=True)
def coordinate_decisions(ptr, idx, alpha, colptr, colrows, rowview, initial,
                         target, known, coefficient, weight, cost, max_passes):
    selected = initial.copy(); views = len(weight); rows = len(rowview)
    prob = np.zeros(rows); bce = np.zeros(views); mass = np.zeros(views); overlap = np.zeros(views); truth = np.zeros(views)
    for r in range(rows):
        if not known[r]: continue
        v = rowview[r]; p = probability(render_row(ptr, idx, alpha, selected, r))
        prob[r] = p; bce[v] += coefficient[r] * binary_cross_entropy(p, target[r])
        mass[v] += p; overlap[v] += p * target[r]; truth[v] += target[r]
    trace = np.zeros(max_passes+1); flips = 0; changed = 0; evaluated = 0
    dm = np.zeros(views); dt = np.zeros(views); db = np.zeros(views)
    # At most one proposed flip's affected rows is buffered at a time.
    scratch = np.zeros(int(np.max(colptr[1:] - colptr[:-1])) if len(initial) else 0)
    for step in range(max_passes+1):
        energy = .1 * np.dot(cost, selected.astype(np.float64))
        for v in range(views):
            energy += weight[v] * (bce[v] + 1 - (2*overlap[v]+1)/(mass[v]+truth[v]+1))
        trace[step] = energy
        if step == max_passes or (step > 0 and changed == 0): break
        changed = 0
        for g in range(len(selected)):
            if colptr[g] == colptr[g+1] and cost[g] == 0: continue
            dm[:] = 0.; dt[:] = 0.; db[:] = 0.
            sign = -1. if selected[g] else 1.
            delta = .1 * sign * cost[g]
            for k in range(colptr[g], colptr[g+1]):
                r = colrows[k]
                if not known[r]: continue
                v = rowview[r]; p = probability(render_row(ptr, idx, alpha, selected, r, g))
                scratch[k-colptr[g]] = p; diff = p-prob[r]
                dm[v] += diff; dt[v] += diff*target[r]
                db[v] += coefficient[r] * (binary_cross_entropy(p, target[r])-binary_cross_entropy(prob[r], target[r]))
            for v in range(views):
                delta += weight[v] * (db[v] + (2*overlap[v]+1)/(mass[v]+truth[v]+1)
                    - (2*(overlap[v]+dt[v])+1)/(mass[v]+dm[v]+truth[v]+1))
            evaluated += 1
            if delta < -1e-12:
                selected[g] = not selected[g]; changed += 1; flips += 1
                mass += dm; overlap += dt; bce += db
                for k in range(colptr[g], colptr[g+1]):
                    r = colrows[k]
                    if known[r]: prob[r] = scratch[k-colptr[g]]
    return selected, trace[:step+1], flips, evaluated, changed == 0


class ExtentDecision:
    """One shared Gaussian/ray incidence structure, independent of queries."""
    def __init__(self, factors, count):
        self.count = count
        self.sizes = [len(v.atom) for v in factors.views]
        self.slices = np.r_[0, np.cumsum(self.sizes)]
        lengths = np.concatenate([v.pointer.cpu().numpy()[1:] - v.pointer.cpu().numpy()[:-1] for v in factors.views])
        self.pointer = np.r_[0, np.cumsum(lengths)]
        self.index = np.concatenate([v.index.cpu().numpy() for v in factors.views])
        self.alpha = np.concatenate([v.alpha.cpu().numpy() for v in factors.views])
        self.rowview = np.repeat(np.arange(len(self.sizes)), self.sizes)
        matrix = sparse.csr_matrix((np.ones(len(self.index), np.uint8), self.index, self.pointer), shape=(sum(self.sizes), count)).tocsc()
        self.colptr, self.colrows = matrix.indptr, matrix.indices

    def solve(self, initial, target, positive, negative, confidence, cost, *, max_passes=4):
        initial = np.asarray(initial); target = np.asarray(target, dtype=np.float64)
        positive = np.asarray(positive); negative = np.asarray(negative)
        confidence = np.asarray(confidence, dtype=np.float64); cost = np.asarray(cost, dtype=np.float64)
        if initial.shape != (self.count,) or cost.shape != initial.shape or initial.dtype != bool:
            raise ValueError('invalid member initial state')
        if target.shape != (sum(self.sizes),) or positive.shape != target.shape or negative.shape != target.shape:
            raise ValueError('unaligned extent evidence')
        if positive.dtype != bool or negative.dtype != bool or (positive & negative).any():
            raise ValueError('invalid signed extent evidence')
        if confidence.shape != (len(self.sizes),) or not isinstance(max_passes, int) or not 1 <= max_passes <= 8:
            raise ValueError('invalid view or bounded decision policy')
        if any(not np.isfinite(v).all() for v in [target, confidence, cost]) or (cost < 0).any() or (target < 0).any() or (target > 1).any() or (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('invalid extent values')
        active = confidence > 0
        if not active.any():
            return initial.copy(), dict(fallback=True, passes=0, flips=0, energy_trace=[], query_parameter_fitting=False)
        known = (positive | negative) & active[self.rowview]
        coefficient = np.zeros(len(target)); weight = confidence / active.sum()
        for vi, (a, b) in enumerate(zip(self.slices[:-1], self.slices[1:])):
            coefficient[a:b] = .5 * (positive[a:b] / max(positive[a:b].sum(), 1) + negative[a:b] / max(negative[a:b].sum(), 1))
        selected, trace, flips, evaluated, converged = coordinate_decisions(self.pointer, self.index, self.alpha,
            self.colptr, self.colrows, self.rowview, initial, target, known, coefficient, weight, cost, max_passes)
        if (np.diff(trace) > 1e-8).any(): raise RuntimeError('nonmonotone exact coordinate objective')
        return selected, dict(fallback=False, energy_trace=trace.tolist(), passes=len(trace)-1, flips=flips,
            evaluated=evaluated, converged=bool(converged), added=int((selected & ~initial).sum()),
            removed=int((initial & ~selected).sum()), query_parameter_fitting=False,
            global_optimum_guaranteed=False, scope_input_validated_elsewhere=True)
