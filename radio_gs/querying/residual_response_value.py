"""Query evidence corrects an out-of-room structural utility prior.

The shuffled-query target is a zero *correction*, not a claim of zero actual
answer utility. Frozen candidates, teachers and the retain/replace guard stay
unchanged. All folds group every task and episode of a room together.
"""
import hashlib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .response_replacement import FEATURE_NAMES, STRUCTURE_NAMES
from .response_value import set_features


def structural_columns(x):
    x = np.asarray(x)
    d, s = len(FEATURE_NAMES), len(STRUCTURE_NAMES)
    if x.ndim != 2 or x.shape[1] != 3*d or not np.isfinite(x).all():
        raise ValueError('finite full set-context signatures required')
    return np.concatenate([x[:, offset:offset+s] for offset in (0, d, 2*d)], axis=1)


class ResidualResponseValue:
    def __init__(self, prior, correction):
        self.prior, self.correction = prior, correction
        self.n_features_in_ = 3*len(FEATURE_NAMES)

    def predict(self, x):
        return self.prior.predict(structural_columns(x)) + self.correction.predict(x)


def pool_arrays(reports):
    xs, nulls, ys, weights, rooms = [], [], [], [], []
    for report in reports:
        for row in report['rows']:
            raw = np.asarray(row['features'], dtype=np.float32)
            shuffled = np.asarray(row['permuted_features'], dtype=np.float32)
            y = np.asarray(row['ious'], dtype=np.float32)
            if (raw.shape != shuffled.shape or len(raw) != len(y)
                    or not np.isfinite(y).all()
                    or not np.allclose(raw[:, :len(STRUCTURE_NAMES)],
                                       shuffled[:, :len(STRUCTURE_NAMES)], rtol=0, atol=1e-6)):
                raise ValueError('matched null must preserve candidates and structural evidence')
            xs.append(set_features(raw)); nulls.append(set_features(shuffled)); ys.append(y-y.mean())
            weights.extend([1/(len(report['rows'])*len(y))]*len(y))
            rooms.extend([report['scene']]*len(y))
    return (np.concatenate(xs), np.concatenate(nulls), np.concatenate(ys),
            np.asarray(weights), np.asarray(rooms))


def room_folds(rooms, count=5):
    unique = sorted(set(rooms), key=lambda s: hashlib.sha256(('residual-value-v1/'+s).encode()).hexdigest())
    if not 2 <= count <= len(unique):
        raise ValueError('at least two nonempty room folds required')
    return {room: i % count for i, room in enumerate(unique)}


def out_of_room_prior(x, y, weights, rooms, estimator, folds=5):
    assignment = room_folds(rooms, folds); structural = structural_columns(x)
    fold_ids = np.array([assignment[s] for s in rooms]); predicted = np.empty(len(y)); receipts = []
    for fold in range(folds):
        held = fold_ids == fold
        model = ExtraTreesRegressor(**estimator, n_jobs=4).fit(structural[~held], y[~held], sample_weight=weights[~held])
        predicted[held] = model.predict(structural[held])
        receipts.append(dict(fold=fold, fit_rooms=sorted(set(rooms[~held])),
                             held_rooms=sorted(set(rooms[held])), candidates=int(held.sum())))
    return predicted, receipts


def fit_residual_models(reports, estimator, folds=5):
    x, null, y, weights, rooms = pool_arrays(reports)
    oof, receipts = out_of_room_prior(x, y, weights, rooms, estimator, folds)
    structural = structural_columns(x)
    prior = ExtraTreesRegressor(**estimator, n_jobs=4).fit(structural, y, sample_weight=weights)
    residual = y-oof
    # Both arms have the same sample count, sample weight and model capacity.
    # The matched control duplicates aligned evidence instead of adding nulls.
    doubled_weights = np.r_[weights, weights]/2
    correction = ExtraTreesRegressor(**estimator, n_jobs=4).fit(
        np.concatenate([x, null]), np.r_[residual, np.zeros_like(residual)], sample_weight=doubled_weights)
    unregularized = ExtraTreesRegressor(**estimator, n_jobs=4).fit(
        np.concatenate([x, x]), np.r_[residual, residual], sample_weight=doubled_weights)
    for model in (prior, correction, unregularized):
        model.n_jobs = 1
    models = dict(full=ResidualResponseValue(prior, correction), query_blind=prior,
                  unregularized=ResidualResponseValue(prior, unregularized))
    detail = dict(folds=receipts, candidates=len(y), room_count=len(set(rooms)),
        original_weight=float(weights.sum()), augmented_weight=float(doubled_weights.sum()),
        oof_structural_mse=float(np.average((oof-y)**2, weights=weights)),
        null_zero_correction_mse=float(np.average(correction.predict(null)**2, weights=weights)),
        unregularized_null_correction_mse=float(np.average(unregularized.predict(null)**2, weights=weights)),
        null_equal_to_aligned_candidates=int(np.all(np.isclose(x,null,rtol=0,atol=1e-6),axis=1).sum()),
        full_trees=2*estimator['n_estimators'], prior_trees=estimator['n_estimators'],
        no_null_control_trees=2*estimator['n_estimators'], benchmark_labels_used=False)
    return models, detail
