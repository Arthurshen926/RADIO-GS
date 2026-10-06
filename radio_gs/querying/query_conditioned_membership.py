"""Keep uncertain native query evidence through whole-carrier membership.

An estimated anchor is context, not a guaranteed positive. Query scores retain
their native ranking; all supported regions contribute soft observation evidence.
"""
import numpy as np
from scipy import sparse
from scipy.special import softmax
from scipy.stats import rankdata

from .direct_anchor_membership import DirectAnchorEvidence, FEATURE_NAMES as ANCHOR_FEATURES


QUERY_FEATURES = [
    'seed_region_member', 'query_global_rank_mean', 'query_global_rank_max',
    'query_view_rank_mean', 'query_view_rank_max',
    'query_relative_evidence_mean', 'query_relative_evidence_max',
    'query_expected_visible_membership', 'query_view_membership_variance',
    'query_observed_confidence', 'query_identity_entropy', 'query_top_margin',
]
FEATURE_NAMES = ANCHOR_FEATURES + QUERY_FEATURES


class QueryMembershipEvidence(DirectAnchorEvidence):
    def __init__(self, membership, visibility, frames, xyz, descriptors, transfer):
        super().__init__(membership, visibility, frames, xyz, descriptors, transfer)
        self.columns = self.membership.tocsc()
        self.available = np.flatnonzero(np.asarray(membership.sum(0)).ravel() > 0)
        self.frame_regions = [np.flatnonzero(self.frames == f) for f in range(visibility.shape[1])]

    def context(self, identity):
        seed = int(self.available[np.argmax(identity[self.available])])
        vertices = self.columns[:, seed].indices
        xyz = self.xyz[vertices];median = np.median(xyz, axis=0)
        anchor = int(vertices[np.argmin(((xyz - median) ** 2).sum(1))])
        return seed, anchor

    def query_features(self, identity, seed, vertices):
        scores = np.asarray(identity, dtype=np.float64)
        available = self.available;values = scores[available]
        rank = np.zeros(len(scores));rank[available] = rankdata(values) / len(values)
        scale = max(float(values.std()), 1e-8)
        standardized = (values - values.mean()) / scale
        relative = np.zeros(len(scores));relative[available] = np.exp(standardized - standardized.max())
        view_rank = np.zeros(len(scores));weights = np.zeros(len(scores))
        confidence = np.zeros(self.visibility.shape[1])
        supported = np.zeros(len(scores), bool);supported[available] = True
        for f, ids in enumerate(self.frame_regions):
            ids = ids[supported[ids]]
            if not len(ids):continue
            view_rank[ids] = rankdata(scores[ids]) / len(ids)
            confidence[f] = relative[ids].max()
            weights[ids] = relative[ids] / relative[ids].sum()
        # Nested masks share one frame's mass; different views are weighted by
        # query evidence. A soft complement is never supplied as a training label.
        association = sparse.csc_matrix((weights, (np.arange(len(scores)), self.frames)),
            shape=(len(scores), self.visibility.shape[1]))
        b = self.membership[vertices];operator = self.transfer[vertices]
        votes = (b @ association).tocsr()
        known = np.asarray(self.visibility[vertices] @ confidence).ravel()
        positive = np.asarray(votes @ confidence).ravel()
        mean = np.divide(positive, known, out=np.zeros_like(positive), where=known > 0)
        second = np.asarray(votes.multiply(votes) @ confidence).ravel()
        second = np.divide(second, known, out=np.zeros_like(second), where=known > 0)
        columns = [b[:, seed].toarray().ravel()]
        for field in [rank, view_rank, relative]:
            columns.extend([np.asarray(operator @ field).ravel(), b.multiply(field).max(1).toarray().ravel()])
        probability = softmax(standardized)
        entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum() / max(np.log(len(values)), 1.)
        largest = np.sort(values)[-2:]
        margin = float(largest[-1] - largest[0]) / scale
        columns.extend([mean, np.maximum(second - mean ** 2, 0), known / max(confidence.sum(), 1e-8),
                        np.full(len(vertices), entropy), np.full(len(vertices), margin)])
        return np.column_stack(columns).astype(np.float32)

    def encode_query(self, identity, vertices, native_query_response=None):
        seed, anchor = self.context(identity)
        response = identity if native_query_response is None else native_query_response
        return np.column_stack([self.encode(anchor, vertices), self.query_features(response, seed, vertices)]), seed, anchor


class QueryConditionedMembership:
    def __init__(self, membership, visibility, frames, xyz, descriptors, transfer, checkpoint):
        import joblib
        self.saved = joblib.load(checkpoint)
        self.evidence = QueryMembershipEvidence(membership, visibility, frames, xyz, descriptors, transfer)

    def infer(self, identity, native_query_response):
        # Image identity may combine ordinal ranks from different teachers.
        # Its region choice is preserved, but ordinal ranks must not masquerade
        # as native cosine spacing in the query-evidence channels.
        vertices = np.arange(len(self.evidence.xyz))
        if self.saved['condition'] == 'query_only':
            seed, anchor = self.evidence.context(identity)
            features = self.evidence.query_features(native_query_response, seed, vertices)[:, 1:]
        else:
            features, seed, anchor = self.evidence.encode_query(identity, vertices, native_query_response)
            if self.saved['condition'] == 'anchor_only':features = features[:, :len(ANCHOR_FEATURES)]
        rank = self.saved['estimator'].predict_proba(features)[:, 1].astype(np.float32)
        # Neither this estimated point nor its containing region is forced positive.
        return rank, rank > self.saved['threshold'], dict(
            seed_region=seed, estimated_anchor=anchor, anchor_forced_positive=False,
            query_evidence=self.saved['condition'] != 'anchor_only',
            estimated_anchor_used=self.saved['condition'] != 'query_only',
            query_evidence_space='native teacher response; region choice retains original identity fusion',
            threshold=self.saved['threshold'], complete_carrier=True,
            training_condition=self.saved['supervision'], source_rgb_read=False,
            labels_at_inference=False, calibrated_on_benchmark=False,
            scores_are_verified_identity_probabilities=False)
