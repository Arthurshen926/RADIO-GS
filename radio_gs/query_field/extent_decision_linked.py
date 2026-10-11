"""Identical binary decisions, traversing only currently selected optical hits."""
import numpy as np
from numba import njit
from scipy import sparse
from .extent_decision import ExtentDecision, probability, binary_cross_entropy


@njit(cache=True)
def response(head, following, alpha, row, position=-1, remove=True):
    t = np.float32(1.); k = head[row]; pending = not remove
    while k >= 0 or pending:
        if pending and (k < 0 or position < k):
            hit = position; pending = False
        else:
            hit = k; k = following[k]
            if remove and hit == position: continue
        a = min(alpha[hit], np.float32(.999))
        if a < 1/255: continue
        next_t = np.float32(t*np.float32(np.float32(1)-a))
        if next_t <= 1e-4: break
        t = next_t
    return np.float32(1)-t


@njit(cache=True)
def decisions(ptr, idx, alpha, colptr, colrows, colposition, rowview, initial,
              target, known, coefficient, weight, cost, max_passes):
    selected = initial.copy(); views = len(weight); rows = len(rowview)
    head = np.full(rows, -1, np.int32); following = np.full(len(idx), -1, np.int32)
    prob = np.zeros(rows); bce = np.zeros(views); mass = np.zeros(views); overlap = np.zeros(views); truth = np.zeros(views)
    for r in range(rows):
        if not known[r]: continue
        for k in range(ptr[r+1]-1, ptr[r]-1, -1):
            if selected[idx[k]]: following[k] = head[r]; head[r] = k
        v = rowview[r]; p = probability(response(head, following, alpha, r))
        prob[r] = p; bce[v] += coefficient[r]*binary_cross_entropy(p, target[r])
        mass[v] += p; overlap[v] += p*target[r]; truth[v] += target[r]
    trace = np.zeros(max_passes+1); flips = 0; changed = 0; evaluated = 0
    dm = np.zeros(views); dt = np.zeros(views); db = np.zeros(views)
    scratch = np.zeros(int(np.max(colptr[1:]-colptr[:-1])) if len(initial) else 0)
    for step in range(max_passes+1):
        energy = .1*np.dot(cost, selected.astype(np.float64))
        for v in range(views): energy += weight[v]*(bce[v]+1-(2*overlap[v]+1)/(mass[v]+truth[v]+1))
        trace[step] = energy
        if step == max_passes or (step > 0 and changed == 0): break
        changed = 0
        for g in range(len(selected)):
            if colptr[g] == colptr[g+1] and cost[g] == 0: continue
            dm[:] = 0.; dt[:] = 0.; db[:] = 0.
            sign = -1. if selected[g] else 1.; delta = .1*sign*cost[g]
            for k in range(colptr[g], colptr[g+1]):
                r = colrows[k]
                if not known[r]: continue
                v = rowview[r]; p = probability(response(head, following, alpha, r, colposition[k], selected[g]))
                scratch[k-colptr[g]] = p; diff = p-prob[r]
                dm[v] += diff; dt[v] += diff*target[r]
                db[v] += coefficient[r]*(binary_cross_entropy(p,target[r])-binary_cross_entropy(prob[r],target[r]))
            for v in range(views):
                delta += weight[v]*(db[v]+(2*overlap[v]+1)/(mass[v]+truth[v]+1)
                    -(2*(overlap[v]+dt[v])+1)/(mass[v]+dm[v]+truth[v]+1))
            evaluated += 1
            if delta < -1e-12:
                selected[g] = not selected[g]; changed += 1; flips += 1
                mass += dm; overlap += dt; bce += db
                for k in range(colptr[g], colptr[g+1]):
                    r = colrows[k]
                    if not known[r]: continue
                    prob[r] = scratch[k-colptr[g]]
                    position = colposition[k]; previous = -1; current = head[r]
                    while current >= 0 and current < position:
                        previous = current; current = following[current]
                    if selected[g]:
                        following[position] = current
                        if previous < 0: head[r] = position
                        else: following[previous] = position
                    else:
                        assert current == position
                        if previous < 0: head[r] = following[position]
                        else: following[previous] = following[position]
                        following[position] = -1
    return selected, trace[:step+1], flips, evaluated, changed == 0


class LinkedExtentDecision(ExtentDecision):
    def __init__(self, factors, count):
        super().__init__(factors, count)
        if len(self.index) >= np.iinfo(np.int32).max:
            raise ValueError('linked reference requires fewer than 2^31 incidences')
        positions = sparse.csr_matrix((np.arange(len(self.index), dtype=np.int32), self.index, self.pointer),
                                     shape=(sum(self.sizes), count)).tocsc()
        assert np.array_equal(positions.indptr, self.colptr) and np.array_equal(positions.indices, self.colrows)
        self.colposition = positions.data

    def solve(self, initial, target, positive, negative, confidence, cost, *, max_passes=4):
        # Keep the reference's validation, weighting and reporting literally the
        # same; bind only its numerical kernel through an instance adapter below.
        return solve(self, initial, target, positive, negative, confidence, cost, max_passes=max_passes)


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
    coefficient = np.zeros(len(target)); weight = confidence/active.sum()
    for vi, (a,b) in enumerate(zip(self.slices[:-1],self.slices[1:])):
        coefficient[a:b] = .5*(positive[a:b]/max(positive[a:b].sum(),1)+negative[a:b]/max(negative[a:b].sum(),1))
    selected, trace, flips, evaluated, converged = decisions(self.pointer,self.index,self.alpha,self.colptr,
        self.colrows,self.colposition,self.rowview,initial,target,known,coefficient,weight,cost,max_passes)
    if (np.diff(trace)>1e-8).any(): raise RuntimeError('nonmonotone exact coordinate objective')
    return selected, dict(fallback=False, energy_trace=trace.tolist(),passes=len(trace)-1,flips=flips,
        evaluated=evaluated,converged=bool(converged),added=int((selected & ~initial).sum()),removed=int((initial & ~selected).sum()),
        query_parameter_fitting=False,global_optimum_guaranteed=False,scope_input_validated_elsewhere=True)
