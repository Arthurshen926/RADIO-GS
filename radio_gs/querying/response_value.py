"""A common utility potential over a complete candidate set.

Pairwise regressors need not assign mutually consistent candidate utilities.
Here candidate order is irrelevant and all relative gains are differences of
one set-conditioned value. Geometry, teachers and complete answers are frozen.
"""
import numpy as np
from .response_replacement import FEATURE_NAMES, STRUCTURE_NAMES


def set_features(features, *, query_blind=False):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite complete candidate signatures required')
    if query_blind:
        x = x[:, :len(STRUCTURE_NAMES)]
    mean = np.broadcast_to(x.mean(0, keepdims=True), x.shape)
    return np.concatenate([x, mean, x-mean], axis=1)


def relative_utilities(estimator, features, *, query_blind=False):
    value = estimator.predict(set_features(features, query_blind=query_blind))
    gain = value-value[0]
    gain[0] = 0.
    if not np.isfinite(gain).all():
        raise ValueError('nonfinite candidate utility')
    return gain


def training_values(reports, *, query_blind=False):
    x, y, weights = [], [], []
    for report in reports:
        for row in report['rows']:
            target = np.asarray(row['ious'], dtype=np.float32)
            x.extend(set_features(row['features'], query_blind=query_blind))
            # Remove query difficulty from supervision. Only relative values
            # affect replacement; a common utility offset cannot change it.
            y.extend(target-target.mean())
            weights.extend([1/(len(report['rows'])*len(target))]*len(target))
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(weights)
