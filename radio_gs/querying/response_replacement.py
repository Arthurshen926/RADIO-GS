"""Learn the utility of replacing a complete response, not its representation.

The frozen provider owns candidate generation and element membership. This
module only compares intact answers using observable query/evidence agreement.
Its target is a difference of actual set IoUs, not a teacher similarity or an
anchor-correctness bit. Candidate zero is always the incumbent answer.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


STRUCTURE_NAMES = [
    'selected_fraction', 'mean_selected_probability', 'mean_other_probability',
    'mean_probability_uncertainty', 'selected_visibility', 'view_coverage',
    'mean_best_region_iou', 'mean_best_region_precision', 'mean_best_region_recall',
]
QUERY_NAMES = [
    'global_rank_at_response', 'global_rank_outside_response', 'global_rank_contrast',
    'local_rank_at_response', 'local_rank_outside_response', 'local_rank_contrast',
    'query_weighted_iou', 'query_weighted_precision', 'query_weighted_recall',
    'best_query_iou',
]
FEATURE_NAMES = STRUCTURE_NAMES + [f'{n}_{s}' for s in ('mean', 'std', 'q25') for n in QUERY_NAMES]


def query_ranks(scores, available):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    if not np.isfinite(scores).all() or not len(available):
        raise ValueError('finite query evidence on a nonempty domain is required')
    ranks = np.zeros_like(scores)
    ranks[available] = rankdata(scores[available], axis=0) / len(available)
    return ranks


def propose_regions(scores, available, frames, baseline_seed=None, budget=4):
    """All-teacher proposal pool: incumbent, teacher winners, other-view winners.

    The budget limits response decoding, not retained teachers or observations.
    Teacher winners are deduplicated and sorted by the all-teacher fused rank;
    consequently reordering teachers cannot change the proposal pool.
    """
    if budget < 1:
        raise ValueError('positive complete-response budget required')
    ranks = query_ranks(scores, available)
    fused = ranks.mean(1)
    order = sorted(available.tolist(), key=lambda r: (-fused[r], r))
    seed = order[0] if baseline_seed is None else int(baseline_seed)
    if seed not in set(available.tolist()):
        raise ValueError('incumbent seed outside available observation domain')
    result = [seed]
    winners = {int(available[np.argmax(ranks[available, t])]) for t in range(ranks.shape[1])}
    for region in sorted(winners, key=lambda r: (-fused[r], r)):
        if region not in result and len(result) < budget:
            result.append(region)
    for region in order:
        if len(result) >= budget:
            break
        if region not in result and frames[region] not in set(frames[result].tolist()):
            result.append(region)
    return result


class RegionResponseEvidence:
    """Fixed-size, teacher-order-invariant signatures of complete responses.

    Outside-response observations describe a competing query association; they
    are not automatically granted negative membership authority. Visibility is
    explicit, and nested masks are averaged within a view before pooling views.
    """
    def __init__(self, membership, visibility, frames):
        self.membership = membership.tocsc().astype(np.float32)
        self.visibility = visibility.tocsc().astype(np.float32)
        self.frames = np.asarray(frames)
        self.area = np.asarray(self.membership.sum(0)).ravel()
        self.available = np.flatnonzero(self.area > 0)
        if (self.frames.shape != self.area.shape or membership.shape[0] != visibility.shape[0]
                or np.any(self.frames < 0) or np.any(self.frames >= visibility.shape[1])):
            raise ValueError('unaligned response and observation carriers')
        self.groups = [self.available[self.frames[self.available] == f]
                       for f in range(visibility.shape[1])]
        self.visibility_fraction = np.asarray(visibility.mean(1)).ravel()

    def encode(self, probabilities, selections, scores, *, exclude_views=(), permuted=False):
        probabilities = np.asarray(probabilities, dtype=np.float32)
        selections = np.asarray(selections, dtype=bool)
        if probabilities.ndim != 2 or probabilities.shape != selections.shape:
            raise ValueError('expected complete candidate-by-element arrays')
        if probabilities.shape[1] != self.membership.shape[0] or not np.isfinite(probabilities).all():
            raise ValueError('response carrier mismatch or nonfinite probability')
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError('response probabilities outside [0,1]')
        ranks = query_ranks(scores, self.available)
        if ranks.shape[0] != self.membership.shape[1]:
            raise ValueError('query evidence and regional observations differ')
        if permuted:
            rng = np.random.default_rng(20261016)
            ranks = ranks.copy()
            for ids in self.groups:
                # One permutation jointly preserves all teacher scores and their
                # correlation, destroying only query-to-region correspondence.
                ranks[ids] = ranks[ids[rng.permutation(len(ids))]]
        intersect = np.asarray(self.membership.T @ selections.T.astype(np.float32)).T
        seen = np.asarray(self.visibility.T @ selections.T.astype(np.float32)).T
        union = self.area[None] + seen[:, self.frames] - intersect
        iou = np.divide(intersect, union, out=np.zeros_like(intersect), where=union > 0)
        precision = intersect / np.maximum(self.area, 1)
        recall = intersect / np.maximum(seen[:, self.frames], 1)
        excluded = set(exclude_views)
        rows, view_rows = [], []
        for f, ids in enumerate(self.groups):
            if f in excluded or len(ids) < 2:
                continue
            q = ranks[ids]
            local = rankdata(q, axis=0) / len(ids)
            overlap = iou[:, ids]
            plus = overlap / np.maximum(overlap.sum(1, keepdims=True), 1e-8)
            minus = (1-overlap) / np.maximum((1-overlap).sum(1, keepdims=True), 1e-8)
            # Smooth rank weights have a fixed power, never fitted on benchmarks.
            focused = q**8
            focused /= np.maximum(focused.sum(0, keepdims=True), 1e-8)
            global_in, global_out = plus @ q, minus @ q
            local_in, local_out = plus @ local, minus @ local
            per_teacher = np.stack([global_in, global_out, global_in-global_out,
                local_in, local_out, local_in-local_out, overlap @ focused,
                precision[:, ids] @ focused, recall[:, ids] @ focused,
                (overlap[:, :, None]*q[None]).max(1)], -1)
            rows.append(per_teacher.mean(1))
            view_rows.append(np.stack([overlap.max(1), precision[:, ids].max(1), recall[:, ids].max(1)], 1))
        count = selections.sum(1)
        other = (~selections).sum(1)
        structure = np.stack([count/selections.shape[1],
            (probabilities*selections).sum(1)/np.maximum(count, 1),
            (probabilities*(~selections)).sum(1)/np.maximum(other, 1),
            (4*probabilities*(1-probabilities)).mean(1),
            (selections @ self.visibility_fraction)/np.maximum(count, 1),
            (seen > 0).mean(1)], 1)
        geometry = np.mean(view_rows, 0) if view_rows else np.zeros((len(count), 3))
        if rows:
            evidence = np.stack(rows)
            pooled = np.concatenate([evidence.mean(0), evidence.std(0), np.quantile(evidence, .25, axis=0)], 1)
        else:
            pooled = np.zeros((len(count), 3*len(QUERY_NAMES)))
        x = np.concatenate([structure, geometry, pooled], 1).astype(np.float32)
        return x, dict(verification_views=len(rows), excluded_views=sorted(excluded),
                       query_teachers=ranks.shape[1], query_permuted=permuted)


def pair_features(left, right, *, query_blind=False):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape[-1] != len(FEATURE_NAMES):
        raise ValueError('unaligned response signatures')
    if query_blind:
        left, right = left[..., :len(STRUCTURE_NAMES)], right[..., :len(STRUCTURE_NAMES)]
    return np.concatenate([left, right, right-left], axis=-1)


def relative_utilities(estimator, features, *, query_blind=False):
    """Antisymmetric complete-response gain; incumbent self-gain is exactly zero."""
    features = np.asarray(features)
    base = np.broadcast_to(features[0], features.shape)
    forward = estimator.predict(pair_features(base, features, query_blind=query_blind))
    reverse = estimator.predict(pair_features(features, base, query_blind=query_blind))
    gain = (forward-reverse)/2
    gain[0] = 0.
    if not np.isfinite(gain).all():
        raise ValueError('nonfinite predicted replacement utility')
    return gain


def choose_response(gains, margin, *, evidence_available=True):
    """Return a whole candidate; stable ties or abstention retain candidate zero."""
    gains = np.asarray(gains)
    if gains.ndim != 1 or not len(gains) or gains[0] != 0 or not np.isfinite(gains).all():
        raise ValueError('finite gains with an exact zero incumbent required')
    if margin is not None and (not np.isfinite(margin) or margin < 0):
        raise ValueError('margin must be nonnegative or None (always abstain)')
    winner = int(np.argmax(gains))
    return winner if evidence_available and margin is not None and gains[winner] > margin else 0


def calibrate_margin(episodes):
    """One global margin on separate calibration rooms, with a reject-all option.

    Maximize scene-macro actual IoU subject to no calibration room losing IoU.
    This empirical constraint is NOT a distribution-free risk guarantee.
    """
    if not episodes:
        raise ValueError('separate calibration episodes are required')
    scenes = sorted({e['scene'] for e in episodes})
    grid = sorted({0.} | {float(max(e['gains'])) for e in episodes if max(e['gains']) > 0}) + [None]
    candidates = []
    for margin in grid:
        changes = {s: [] for s in scenes}
        for e in episodes:
            k = choose_response(e['gains'], margin, evidence_available=e['evidence_available'])
            changes[e['scene']].append(e['ious'][k]-e['ious'][0])
        per_scene = {s: float(np.mean(v)) for s, v in changes.items()}
        if min(per_scene.values()) >= -1e-12:
            candidates.append(dict(margin=margin, delta=float(np.mean(list(per_scene.values()))), per_scene=per_scene))
    # Prefer the most conservative margin on an exact utility tie.
    best = max(candidates, key=lambda r: (r['delta'], float('inf') if r['margin'] is None else r['margin']))
    return best, dict(rule='scene-macro gain; nonnegative gain in each calibration room; conservative ties',
                      candidates=len(grid), feasible=len(candidates), calibrated_risk_guarantee=False)
