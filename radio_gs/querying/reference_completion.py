"""Deterministic original-query constraints for frozen reference completion.

Neither the inherited mask nor absence of a scribble is a negative annotation.
All selected points lie on an actual signed scribble, in native XY coordinates.
"""
import numpy as np


def signed_scribble_points(positive, negative, per_sign=16):
    p, n = np.asarray(positive), np.asarray(negative)
    if p.dtype != bool or n.dtype != bool or p.ndim != 2 or p.shape != n.shape or (p & n).any():
        raise ValueError('disjoint aligned boolean scribbles required')
    if type(per_sign) is not int or per_sign < 1:
        raise ValueError('positive integer point budget required')
    points, labels = [], []
    for mask, label in [(p, 1), (n, 0)]:
        yx = np.argwhere(mask).astype(np.float64)
        if len(yx) == 0:
            raise ValueError('both original query signs required')
        # Deterministic centroid-nearest start followed by farthest-point cover.
        index = int(np.square(yx - yx.mean(0)).sum(1).argmin())
        distance = np.full(len(yx), np.inf)
        selected = np.zeros(len(yx), bool)
        for _ in range(min(per_sign, len(yx))):
            points.append(yx[index, ::-1]); labels.append(label); selected[index] = True
            distance = np.minimum(distance, np.square(yx - yx[index]).sum(1))
            distance[selected] = -1
            index = int(distance.argmax())
    return dict(point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32))


def enforce_reference_scribbles(probability, positive, negative):
    """Impose only original user constraints; leave every unknown pixel intact."""
    x = np.asarray(probability, dtype=np.float32)
    p, n = np.asarray(positive), np.asarray(negative)
    if (x.ndim != 2 or p.dtype != bool or n.dtype != bool or p.shape != x.shape or n.shape != x.shape
            or (p & n).any() or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1):
        raise ValueError('finite aligned response and disjoint signed masks required')
    result = x.copy(); result[p] = 1.; result[n] = 0.
    return result
