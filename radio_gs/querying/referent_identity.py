"""Separate target appearance from landmark geometry in native coordinates.

All geometrically supported observations remain candidates. Landmark identity
is marginalized, not selected from benchmark labels or a top-k shortlist.
The returned energy ranks identities; it is not foreground probability.
"""
import numpy as np
from scipy.special import softmax, expit
from scipy.spatial.distance import cdist
from .observation_readout import positive_support_rank


def native_logit_scale(checkpoint):
    import json
    from pathlib import Path
    from safetensors import safe_open
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    with safe_open(checkpoint / index['weight_map']['logit_scale'], framework='pt', device='cpu') as weights:
        return float(weights.get_tensor('logit_scale').float().exp().item())


class ReferentIdentity:
    def __init__(self, membership, xyz, logit_scale):
        self.membership = membership.tocsc()
        self.area = np.asarray(membership.sum(0)).reshape(-1)
        self.available = np.flatnonzero(self.area > 0)
        count = np.maximum(self.area, 1)[:, None]
        self.center = np.asarray(membership.T @ xyz) / count
        variance = np.maximum(np.asarray(membership.T @ (xyz * xyz)) / count - self.center ** 2, 0)
        self.radius = np.sqrt(variance.sum(1))
        self.horizontal_radius = np.sqrt(variance[:, :2].sum(1))
        self.vertical_radius = np.sqrt(variance[:, 2])
        self.logit_scale = logit_scale

    def relation_evidence(self, predicate, reference_cosine):
        """Expected proximity/order to a distinct, uncertain landmark.

        Distances are scaled by measured candidate/landmark extent, not a
        scene-specific fitted radius. Overlap downweights obvious self-reference;
        disjoint parts may still belong to the same instance. This is a geometric
        heuristic, not independently established landmark identity.
        """
        ids = self.available
        reference = softmax(self.logit_scale * reference_cosine[ids])
        values = np.zeros(len(self.area), np.float32)
        for start in range(0, len(ids), 256):
            row = ids[start:start + 256]
            shared = (self.membership[:, row].T @ self.membership[:, ids]).toarray()
            distinct = 1 - np.clip(shared / np.minimum(self.area[row, None], self.area[None, ids]), 0, 1)
            scale = np.maximum(self.radius[row, None] + self.radius[None, ids], 1e-6)
            near = np.exp(-cdist(self.center[row], self.center[ids]) / scale)
            if predicate in ('near', 'far'):
                likelihood = near if predicate == 'near' else 1 - near
            else:
                horizontal = np.maximum(self.horizontal_radius[row, None] + self.horizontal_radius[None, ids], 1e-6)
                vertical = np.maximum(self.vertical_radius[row, None] + self.vertical_radius[None, ids], 1e-6)
                direction = {'above': 1, 'below': -1}[predicate]
                order = expit(direction * (self.center[row, 2, None] - self.center[None, ids, 2]) / vertical)
                likelihood = order * np.exp(-cdist(self.center[row, :2], self.center[ids, :2]) / horizontal)
            values[row] = (distinct * likelihood) @ reference
        return values

    def rank(self, target_cosine, relations, positive_only=False):
        energy = self.logit_scale * target_cosine.astype(np.float64)
        support = np.ones(len(self.area), np.float64)
        report = []
        for predicate, phrase, reference_cosine in relations:
            value = self.relation_evidence(predicate, reference_cosine)
            energy += np.log(np.maximum(value, 1e-12))
            support *= value
            report.append(dict(predicate=predicate, reference=phrase,
                maximum_relation_evidence=float(value.max())))
        if positive_only:
            native = np.clip((target_cosine.astype(np.float64)+1)*.5, 0, 1)
            energy = positive_support_rank(native, support) if relations else native
        energy[self.area == 0] = -np.inf
        return energy, dict(relations=report, logit_scale=self.logit_scale,
            candidate_count=len(self.available), candidate_limit=None,
            unknown_relation_policy='native identity floor retained; no negative evidence' if positive_only else 'log relation factor',
            identity_energy_rule='s+s*(1-s)*p, p=joint relation support' if positive_only else 'native logit plus log relation support',
            relation_rule='marginal native landmark identity times distinctness and extent-normalized distance/order',
            calibrated_probability=False)
