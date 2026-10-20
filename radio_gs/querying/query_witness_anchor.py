"""Carry partial query evidence into an uncertain region-to-anchor decision.

The original identity region and complete membership decoder are preserved.
Other-view regional identity provides a weighted spatial center inside the seed;
it is never a new positive point or a negative membership label. Density-model
weights are relative evidence, not calibrated instance correctness.
"""
import numpy as np
from scipy import sparse

from .source_identity import native_identity_posterior


def weighted_coordinate_median(xyz, weight):
    """Coordinate median; equal masses reproduce numpy's even-sample midpoint."""
    xyz = np.asarray(xyz, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if xyz.ndim != 2 or weight.shape != (len(xyz),):
        raise ValueError('aligned coordinates and weights required')
    if not np.isfinite(xyz).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('finite coordinates and nonnegative weights required')
    positive = weight > 0
    if not positive.any():
        raise ValueError('positive mass required')
    xyz, weight = xyz[positive], weight[positive]
    center = []
    for axis in range(xyz.shape[1]):
        order = np.argsort(xyz[:, axis], kind='stable')
        mass = np.cumsum(weight[order]); half = mass[-1] / 2
        lo = min(int(np.searchsorted(mass, half, side='left')), len(order)-1)
        hi = min(int(np.searchsorted(mass, half, side='right')), len(order)-1)
        center.append((xyz[order[lo], axis] + xyz[order[hi], axis]) / 2)
    return np.asarray(center)


class QueryWitnessAnchor:
    def __init__(self, membership, visibility, frames, xyz):
        self.membership = membership.tocsr()
        self.visibility = visibility.tocsr()
        self.frames = np.asarray(frames, dtype=np.int64)
        self.xyz = np.asarray(xyz)
        if (self.membership.shape[0] != len(xyz) or self.visibility.shape[0] != len(xyz)
                or self.frames.shape != (self.membership.shape[1],)
                or (self.frames < 0).any() or (self.frames >= self.visibility.shape[1]).any()):
            raise ValueError('inconsistent map domains')
        self.available = np.flatnonzero(self.membership.getnnz(axis=0))

    def infer(self, seed, native_response, *, permute=False):
        if seed not in self.available:
            raise ValueError('identity seed has no geometric support')
        native_response = np.asarray(native_response)
        if native_response.shape != (self.membership.shape[1],) or not np.isfinite(native_response).all():
            raise ValueError('one finite native score per original region required')
        vertices = self.membership[:, seed].nonzero()[0]
        xyz = self.xyz[vertices]
        # The query-selected seed view cannot corroborate its own identity.
        eligible = self.available[self.frames[self.available] != self.frames[seed]]
        weight = np.zeros(self.membership.shape[1], dtype=np.float64)
        density, model = native_identity_posterior(native_response[eligible])
        weight[eligible] = np.where(density >= .5, density, 0.)
        frame_mass = np.bincount(self.frames, weights=weight, minlength=self.visibility.shape[1])
        if permute:
            # Same weights/count/budget in every view; break only query-to-region
            # assignment. This is one fixed negative control, not a random sweep.
            rng = np.random.default_rng(20261016)
            for frame in np.flatnonzero(frame_mass):
                ids = eligible[self.frames[eligible] == frame]
                weight[ids] = weight[rng.permutation(ids)]
        control_mass = np.bincount(self.frames, weights=weight, minlength=self.visibility.shape[1])
        if not np.allclose(frame_mass, control_mass, rtol=1e-12, atol=1e-12):
            raise AssertionError('within-view identity intervention changed evidence budget')
        normalized = np.divide(weight, frame_mass[self.frames], out=np.zeros_like(weight),
                               where=frame_mass[self.frames] > 0)
        positive = np.asarray(self.membership[vertices] @ normalized).ravel()
        known = np.asarray(self.visibility[vertices] @ (frame_mass > 0).astype(float)).ravel()
        witness = np.divide(positive, known, out=np.zeros_like(positive), where=known > 0)
        fallback = not np.any(witness > 0)
        center = np.median(xyz, axis=0) if fallback else weighted_coordinate_median(xyz, witness)
        anchor = int(vertices[np.argmin(((xyz-center)**2).sum(1))])
        return anchor, dict(seed_region=int(seed), anchor_vertex=anchor, seed_vertices=len(vertices),
            query_identity_model=model, permuted_within_view=bool(permute),
            seed_view_excluded=True, source_views=int((frame_mass>0).sum()),
            admitted_regions=int((weight>0).sum()), source_weight_mass=float(weight.sum()),
            normalized_view_mass=float(normalized.sum()), seed_witness_mass=float(witness.sum()),
            seed_supported_vertices=int((witness>0).sum()), no_witness_fallback=fallback,
            anchor_identity_guaranteed=False, actual_point_contract_changed=False)
