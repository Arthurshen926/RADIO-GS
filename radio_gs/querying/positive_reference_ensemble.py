"""Positive-only reference scope ensemble. No target or negative-label inputs."""
from __future__ import annotations

import numpy as np


def aggregate_positive_reference_trials(trials, positive, parent):
    """Admit parent-or-better raw-positive coverage, then strict majority.

    Coverage is compared using integer counts, not rounded fractions. The
    parent mask is an input-scope baseline, not ground truth. A rejected SAM
    pixel remains unknown; this helper creates no negative training labels.
    """
    trials, positive, parent = map(np.asarray, (trials, positive, parent))
    if (trials.dtype != bool or positive.dtype != bool or parent.dtype != bool
            or trials.ndim != 3 or positive.ndim != 2
            or trials.shape[1:] != positive.shape or parent.shape != positive.shape
            or len(trials) == 0 or not positive.any()):
        raise ValueError('Nonempty Boolean trials and aligned positive/parent masks required')
    parent_count = int(parent[positive].sum())
    counts = trials[:, positive].sum(axis=1)
    accepted = counts >= parent_count
    fallback = not bool(accepted.any())
    if fallback:
        # Exact parent fallback includes its possible incomplete positive coverage.
        # The downstream frozen prompt repair still enforces raw positives.
        mask = parent.copy()
        probability = parent.astype(np.float32)
    else:
        probability = trials[accepted].mean(axis=0, dtype=np.float32)
        mask = probability > .5
        mask[positive] = True
    return mask, probability, dict(
        trial_positive_counts=counts.tolist(), parent_positive_count=parent_count,
        positive_count=int(positive.sum()), accepted_trials=np.flatnonzero(accepted).tolist(),
        fallback_parent=fallback, threshold=.5, comparison='strict_greater',
        target_information_used=False, sam_complement_is_negative=False,
        changed_pixels=int((mask != parent).sum()), parent_area=int(parent.sum()),
        ensemble_area=int(mask.sum()))
