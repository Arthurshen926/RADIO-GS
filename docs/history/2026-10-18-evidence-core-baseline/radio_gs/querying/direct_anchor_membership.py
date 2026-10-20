"""Direct whole-instance membership from an actual geometric query anchor.

No intermediate mask is treated as a pure instance or required to intersect an
argmax seed. Exact co-observation ambiguity, native appearance and displacement
are inputs to one independently supervised element decision.
"""
import numpy as np
from scipy import sparse

from .anchor_observation_evidence import joint_anchor_evidence, JOINT_FEATURE_NAMES


FEATURE_NAMES = JOINT_FEATURE_NAMES + [
    'anchor_observed_view_fraction', 'element_observed_view_fraction',
    'signed_dx_m_bounded', 'signed_dy_m_bounded', 'signed_dz_m_bounded',
    'distance_m_bounded', 'horizontal_distance_m_bounded',
    'native_anchor_response', 'has_native_element_support',
]


def view_balanced_transfer(membership, offsets):
    blocks = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        block = membership[:, lo:hi].tocsr()
        mass = np.asarray(block.sum(1)).ravel()
        blocks.append(sparse.diags(1 / np.maximum(mass, 1)) @ block)
    transfer = sparse.hstack(blocks, format='csr')
    mass = np.asarray(transfer.sum(1)).ravel()
    return (sparse.diags(1 / np.maximum(mass, 1)) @ transfer).tocsr()


class DirectAnchorEvidence:
    def __init__(self, membership, visibility, frames, xyz, descriptors, transfer):
        self.membership = membership.tocsr();self.visibility = visibility.tocsr()
        self.frames = np.asarray(frames);self.xyz = np.asarray(xyz, np.float32)
        # This auxiliary reader must not renormalize the native array shared by
        # text/image matching in the caller.
        self.descriptors = np.array(descriptors, dtype=np.float32, copy=True)
        self.descriptors /= np.maximum(np.linalg.norm(self.descriptors, axis=1, keepdims=True), 1e-8)
        self.transfer = transfer.tocsr()
        self.visible_fraction = np.asarray(visibility.sum(1)).ravel() / visibility.shape[1]
        self.supported = transfer.getnnz(axis=1) > 0

    def encode(self, anchor, vertices):
        joint = joint_anchor_evidence(self.membership, self.visibility, self.frames, anchor, vertices)
        delta = self.xyz[vertices] - self.xyz[anchor]
        distance = np.linalg.norm(delta, axis=1)
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        prototype = np.asarray(self.transfer[anchor] @ self.descriptors).ravel()
        prototype /= max(np.linalg.norm(prototype), 1e-8)
        native = np.asarray(self.transfer[vertices] @ (self.descriptors @ prototype)).ravel()
        return np.column_stack([joint, np.full(len(vertices), self.visible_fraction[anchor]),
            self.visible_fraction[vertices], delta / (1 + np.abs(delta)), distance / (1 + distance),
            horizontal / (1 + horizontal), native, self.supported[vertices]]).astype(np.float32)


class DirectAnchorMembership:
    def __init__(self, membership, visibility, frames, xyz, descriptors, transfer, checkpoint):
        import joblib
        self.saved = joblib.load(checkpoint)
        self.evidence = DirectAnchorEvidence(membership, visibility, frames, xyz, descriptors, transfer)

    def infer(self, anchor):
        vertices = np.arange(len(self.evidence.xyz))
        probability = self.saved['estimator'].predict_proba(self.evidence.encode(anchor, vertices))[:, 1].astype(np.float32)
        probability[anchor] = 1.
        return probability, probability > self.saved['threshold'], dict(
            anchor_vertex=int(anchor), threshold=self.saved['threshold'],
            training_condition=self.saved['condition'], element_domain='complete frozen carrier',
            seed_mask_assumed_pure=False, source_rgb_read=False,
            output='explicit independent element selection; probability calibration limited to training domain')

    def infer_region(self, seed, region_identity=None):
        """Use an estimated map anchor; unlike a user point, identity is uncertain."""
        vertices = self.evidence.membership[:, seed].nonzero()[0]
        xyz = self.evidence.xyz[vertices]
        median = np.median(xyz, axis=0)
        distance = ((xyz - median) ** 2).sum(1)
        rule = 'actual region member nearest coordinate-wise3D median'
        if region_identity is None:
            anchor = int(vertices[np.argmin(distance)])
        else:
            response = np.asarray(self.evidence.transfer[vertices] @ region_identity).ravel()
            # Regional identity already combines the available query teachers.
            # Preserve that evidence when converting a mixed region to a point.
            best = np.flatnonzero(response == response.max())
            anchor = int(vertices[best[np.argmin(distance[best])]])
            rule = 'maximum view-balanced query identity within selected region; geometric median breaks ties'
        probability, selected, detail = self.infer(anchor)
        detail.update(anchor_source='estimated from selected region, not provided by the user',
            seed_region=int(seed), anchor_rule=rule,
            anchor_identity_guaranteed=False)
        return probability, selected, detail
