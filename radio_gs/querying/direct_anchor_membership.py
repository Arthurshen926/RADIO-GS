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
    from radio_gs import evidence_core
    if evidence_core.enabled():
        return evidence_core.view_balanced_transfer(membership, offsets)
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

    def native_identity(self, anchor):
        prototype = np.asarray(self.transfer[anchor] @ self.descriptors).ravel()
        prototype /= max(np.linalg.norm(prototype), 1e-8)
        return self.descriptors @ prototype

    def encode(self, anchor, vertices, *, native_identity=None):
        joint = joint_anchor_evidence(self.membership, self.visibility, self.frames, anchor, vertices)
        delta = self.xyz[vertices] - self.xyz[anchor]
        distance = np.linalg.norm(delta, axis=1)
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        identity = self.native_identity(anchor) if native_identity is None else native_identity
        native = np.asarray(self.transfer[vertices] @ identity).ravel()
        return np.column_stack([joint, np.full(len(vertices), self.visible_fraction[anchor]),
            self.visible_fraction[vertices], delta / (1 + np.abs(delta)), distance / (1 + distance),
            horizontal / (1 + horizontal), native, self.supported[vertices]]).astype(np.float32)


class DirectAnchorMembership:
    # Older restored/test objects predate the optional execution setting.
    chunk_size = None

    def __init__(self, membership, visibility, frames, xyz, descriptors, transfer, checkpoint, *, chunk_size=None):
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError('member chunk size must be a positive integer')
        import joblib
        self.saved = joblib.load(checkpoint)
        self.evidence = DirectAnchorEvidence(membership, visibility, frames, xyz, descriptors, transfer)
        self.chunk_size = chunk_size

    def infer(self, anchor):
        vertices = np.arange(len(self.evidence.xyz))
        if self.chunk_size is None:
            probability = self.saved['estimator'].predict_proba(self.evidence.encode(anchor, vertices))[:, 1].astype(np.float32)
        else:
            # All carrier vertices are still returned. Only the temporary feature
            # matrix changes size; no top-k, evidence pruning or dtype reduction.
            identity = self.evidence.native_identity(anchor)
            probability = np.empty(len(vertices), np.float32)
            for start in range(0, len(vertices), self.chunk_size):
                rows = vertices[start:start+self.chunk_size]
                features = self.evidence.encode(anchor, rows, native_identity=identity)
                probability[rows] = self.saved['estimator'].predict_proba(features)[:, 1].astype(np.float32)
        probability[anchor] = 1.
        return probability, probability > self.saved['threshold'], dict(
            anchor_vertex=int(anchor), threshold=self.saved['threshold'],
            training_condition=self.saved['condition'], element_domain='complete frozen carrier',
            seed_mask_assumed_pure=False, source_rgb_read=False,
            output='explicit independent element selection; probability calibration limited to training domain')

    def infer_region(self, seed, region_identity=None):
        """Use an estimated map anchor; unlike a user point, identity is uncertain."""
        vertices = self.evidence.membership[:, seed].nonzero()[0]
        if not len(vertices):
            probability = np.zeros(len(self.evidence.xyz), np.float32)
            return probability, probability.astype(bool), dict(seed_region=int(seed),
                anchor_vertex=None, reason='empty_region_has_no_geometric_anchor',
                element_domain='complete frozen carrier', source_rgb_read=False)
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
