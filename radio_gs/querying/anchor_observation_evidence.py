"""Query-conditioned geometric evidence before any learned membership decision.

Nested masks in one view share its weight. A visible exclusion is an observation,
not a ground-truth negative; absent common visibility is represented separately.
"""
import numpy as np
from scipy import sparse


def anchor_membership(membership, visibility, region_frames, anchor, positive_membership=None):
    regions=membership[anchor].indices
    frame_counts=np.bincount(region_frames[regions],minlength=visibility.shape[1])
    weights=np.zeros(membership.shape[1],np.float32)
    weights[regions]=1/np.maximum(frame_counts[region_frames[regions]],1)
    if positive_membership is not None:
        anchor_weight=positive_membership[anchor].toarray().reshape(-1)
        # Conditional co-membership must equal one at a supported positive
        # anchor. Normalize the soft kernel by K(anchor, anchor) in each view;
        # a simple sum would count uncertain anchor membership twice.
        frame_mass=np.bincount(region_frames,weights=anchor_weight**2,minlength=visibility.shape[1])
        weights=anchor_weight/np.maximum(frame_mass[region_frames],1e-8)
    positive=np.asarray((positive_membership if positive_membership is not None else membership)@weights).reshape(-1)
    known=np.asarray(visibility@(frame_counts>0).astype(np.float32)).reshape(-1)
    scores=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
    return np.clip(scores,0,1),dict(anchor_regions=len(regions),anchor_views=int((frame_counts>0).sum()),
        query_supported_vertices=int((known>0).sum()),negative_policy='only_visible_elements_in_anchor_observations')


JOINT_FEATURE_NAMES = [
    'anchor_nested_mask_vote', 'anchor_any_mask_view_fraction',
    'anchor_all_masks_view_fraction', 'anchor_granularity_disagreement',
    'anchor_joint_visibility_fraction', 'anchor_has_joint_visibility',
]


def joint_anchor_evidence(membership, visibility, region_frames, anchor, vertices):
    """Retain the exact positive-point observation relation, including ambiguity."""
    regions = membership[anchor].indices
    frames, inverse, counts = np.unique(region_frames[regions], return_inverse=True, return_counts=True)
    association = sparse.csc_matrix((1. / counts[inverse],
                                    (np.arange(len(regions)), inverse)),
                                   shape=(len(regions), len(frames)))
    # One sparse vote per element/view. Nested coarse and fine masks neither
    # become independent views nor lose their disagreement through pooling.
    votes = (membership[vertices][:, regions] @ association).tocsr()
    known = np.asarray(visibility[vertices][:, frames].sum(1)).ravel()
    mass = np.asarray(votes.sum(1)).ravel()
    any_mask = np.asarray((votes > 0).sum(1)).ravel()
    all_masks = np.asarray((votes >= 1. - 1e-6).sum(1)).ravel()
    denominator = np.maximum(known, 1)
    return np.column_stack([
        np.clip(mass / denominator, 0, 1), any_mask / denominator,
        all_masks / denominator, (any_mask - all_masks) / denominator,
        known / max(len(frames), 1), known > 0,
    ]).astype(np.float32)
