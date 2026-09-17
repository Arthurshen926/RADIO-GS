"""Authorized signed inputs take precedence over conflicting pseudo targets."""
import torch


def conflicting_signed_targets(seed_rows, seed_weights, signs, target, known, observed):
    if len(seed_rows) != len(seed_weights) or len(seed_rows) != len(signs):
        raise ValueError('signed evidence lengths differ')
    if target.ndim != 1 or target.shape != known.shape or target.shape != observed.shape:
        raise ValueError('target axes differ')
    conflicts=[]
    for rows,weights,sign in zip(seed_rows,seed_weights,signs):
        if sign not in (0,1) or rows.ndim != 1 or weights.shape != rows.shape:
            raise ValueError('invalid signed evidence')
        if rows.numel() and (rows.min()<0 or rows.max()>=len(target)):
            raise ValueError('signed row outside target domain')
        if not torch.isfinite(weights).all() or (weights<0).any():
            raise ValueError('invalid signed contribution weights')
        total=weights.sum()
        visible_weights=weights*observed[rows]
        mass=visible_weights.sum()
        contradictory=known[rows] & (target[rows].bool()!=bool(sign))
        # Require both a supported seed and a majority of its observed mixture
        # explicitly labelled opposite. Unknown labels do not vote against it.
        conflicts.append(bool((mass>.5*total) & (mass>0) & (visible_weights[contradictory].sum()>.5*mass)))
    return conflicts
