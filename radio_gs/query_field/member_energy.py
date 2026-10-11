"""Experimental discrete member inference; no fitted query parameters.

Consumes field-owned sparse optical coverage and independently compiled footprint
costs. This functional reference is deliberately not wired into default queries:
its supplied factors determine the extent it can protect.
"""
import numpy as np
from scipy import sparse


def solve_member_energy(matrix, prior, positive, negative, reliability, footprint,
                        *, cutoff, support_weight=.1, prior_weight=.1, max_passes=8, exchange_candidates=8):
    """Bounded deterministic coordinate descent on one binary Gaussian set.

    E = weighted positive deficit + weighted negative excess (each class
    normalized separately) + fixed-prior-normalized footprint + prior disagreement.
    Coverage is capped at the decision threshold in the data terms. Unknown and
    zero-reliability rows never request changes; untouched members retain prior.
    The footprint denominator is frozen before inference, not recomputed per step.
    Exchanges consider at most exchange_candidates unselected neighbors per member,
    sharing a reliable positive factor, ordered by footprint cost then row ID.
    This is a bounded-neighborhood solution, not a global optimum or extent guarantee.
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
    if not isinstance(exchange_candidates, int) or exchange_candidates < 0:
        raise ValueError('invalid exchange budget')
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

    row_matrix = matrix.tocsr()
    trace = [energy()]; flips = 0; passes = 0; exchanges = 0; pairs_evaluated = 0
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
        # Joint replacement can cross a barrier that neither individual flip can.
        # Candidates are defined entirely by field geometry and current evidence.
        if exchange_candidates:
            for g in np.flatnonzero(selected & touched):
                g_start, g_stop = matrix.indptr[g:g+2]
                g_rows = matrix.indices[g_start:g_stop]
                positive_rows = g_rows[p[g_rows] > 0]
                if not len(positive_rows):
                    continue
                candidates = np.unique(np.concatenate([
                    row_matrix.indices[row_matrix.indptr[r]:row_matrix.indptr[r+1]]
                    for r in positive_rows]))
                candidates = candidates[~selected[candidates] & touched[candidates]]
                candidates = candidates[np.lexsort((candidates, cost[candidates]))][:exchange_candidates]
                best = None
                for h in candidates:
                    h_start, h_stop = matrix.indptr[h:h+2]
                    h_rows = matrix.indices[h_start:h_stop]
                    indexes = np.union1d(g_rows, h_rows)
                    change = np.zeros(len(indexes))
                    change[np.searchsorted(indexes, g_rows)] -= matrix.data[g_start:g_stop]
                    change[np.searchsorted(indexes, h_rows)] += matrix.data[h_start:h_stop]
                    updated = np.maximum(coverage[indexes] + change, 0.)
                    delta = float((data_loss(updated, indexes)-data_loss(coverage[indexes], indexes)).sum())
                    delta += support_weight * (cost[h]-cost[g])
                    delta += prior_weight * (disagreement[g]*(1 if initial[g] else -1) +
                                             disagreement[h]*(-1 if initial[h] else 1))
                    pairs_evaluated += 1
                    if delta < -1e-12 and (best is None or delta < best[0]):
                        best = (delta, h, indexes, updated)
                if best is not None:
                    _, h, indexes, updated = best
                    selected[g] = False; selected[h] = True
                    coverage[indexes] = updated
                    changed += 1; exchanges += 1; flips += 2
        trace.append(energy())
        if not changed: break
    return selected, dict(energy_trace=trace, passes=passes, flips=flips, exchanges=exchanges,
                         exchange_candidates=exchange_candidates, pairs_evaluated=pairs_evaluated,
                         added=int((selected & ~initial).sum()),
                         removed=int((initial & ~selected).sum()),
                         converged=changed == 0, footprint_denominator=denominator,
                         footprint_enabled=denominator > 0,
                         query_parameter_fitting=False, full_extent_guaranteed=False)


def query_text_member_energy(field, query, negatives, *, decoder=None, **policy):
    """Experimental native TEXT entrypoint; strong prior and field-only inputs.

    Requires compiled support footprint. Its coverage remains sampled; this API
    does not certify full extent or replace the existing default query path.
    """
    import torch
    from dataclasses import replace
    from .joint_exclusion import JointExclusionDecoder, CUTOFF
    from .scope_confidence import query_text_scope_confident, native_scope_confidence
    from .part_conditioning import native_region_probability
    with torch.no_grad():
        if getattr(field, 'support_footprint', None) is None:
            raise ValueError('member energy requires compiled support footprint')
        result = query_text_scope_confident(field, query, negatives)
        initial = result.semantic_probability
        probability = native_region_probability(field.evidence, query, negatives)
        accepted, confidence = native_scope_confidence(field, initial, probability)
        if not torch.equal(accepted, result.accepted_regions):
            raise ValueError('native acceptance changed between readouts')
        decoder = decoder if decoder is not None else JointExclusionDecoder(field)
        if decoder.field is not field:
            raise ValueError('decoder bound to another field')
        if len(confidence) != len(decoder.slices) or len(field.support_footprint.cost) != len(decoder.slices):
            raise ValueError('unaligned member evidence views')
        positive, negative = decoder.labels(accepted)
        reliability = np.repeat(confidence.cpu().numpy(), decoder.slices)
        # Match the source reference's positive-active view footprint selection;
        # data reliability is separate so low confidence retains its magnitude.
        active = torch.stack([accepted[c.regions].any() for c in field.scope_constraints.views])
        cost = field.support_footprint(active).cpu().numpy()
        selected, details = solve_member_energy(decoder.matrix, initial.cpu().numpy(),
            positive, negative, reliability, cost, cutoff=-np.log1p(-CUTOFF), **policy)
        selected = torch.from_numpy(selected).to(initial.device)
        member = initial.clone()
        member[selected & ~(initial > .5)] = 1.
        member[~selected & (initial > .5)] = 0.
        return replace(result, membership_score=member, selected=selected,
            diagnostics=dict(result.diagnostics, member_energy=details,
                native_view_confidence_preserved=True, coverage_view_confidence_preserved=True,
                initializer='semantic', source_access=False, protection_is_sampled_not_full_image=True))
