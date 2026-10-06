"""Use a separate entity-level prior to set the granularity of an instance.

The external entity decoder is evaluated offline. Query-time input is sparse
membership only. Native identity is never replaced by the generic prompt.
An absent or single-view entity prior leaves the freshly computed base scope.
"""
import numpy as np

from .instance_observation_scope import InstanceObservationScope


class EntityObservationScope:
    def __init__(self, native_membership, native_frames, membership, visibility, frames):
        self.native=native_membership.tocsc()
        self.native_frames=np.asarray(native_frames)
        self.membership=membership.tocsc()
        self.anchor_membership=membership.tocsr()
        self.visibility=visibility.tocsc()
        self.frames=np.asarray(frames)
        self.reader=InstanceObservationScope(membership,visibility,frames)
        self.native_area=np.asarray(native_membership.sum(0)).ravel()

    def infer(self, identity, semantic_rank, anchor=None):
        if anchor is not None:
            regions=self.anchor_membership[anchor].indices
            counts=np.bincount(self.frames[regions],minlength=self.visibility.shape[1])
            detail=dict(anchor_regions=len(regions),anchor_views=int((counts>0).sum()),
                input='positive geometric anchor',prior='offline entity masks')
            if (counts>0).sum()<2:return None,None,dict(detail,applied=False)
            weights=np.zeros(self.membership.shape[1],np.float32)
            weights[regions]=1/counts[self.frames[regions]]
            positive=np.asarray(self.membership@weights).ravel()
            known=np.asarray(self.visibility@(counts>0).astype(np.float32)).ravel()
            fraction=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
            return fraction,fraction>.5,dict(detail,applied=True,selection_rule='visible anchor co-membership strict majority')
        available=np.flatnonzero(self.native_area>0)
        seed=int(available[np.argmax(identity[available])]);seed_frame=self.native_frames[seed]
        scope=self.native[:,seed]
        intersection=(self.membership.T@scope).toarray().ravel()
        coverage=(self.visibility.T@scope).toarray().ravel()
        eligible=(coverage[self.frames]>=.3*self.native_area[seed]) & (intersection>=.8*coverage[self.frames]) & (intersection>0)
        same=np.flatnonzero(eligible&(self.frames==seed_frame))
        candidates=same if len(same) else np.flatnonzero(eligible)
        detail=dict(native_seed_region=seed,prior='offline entity masks',
            eligible_entities=len(candidates),input='unchanged native region identity')
        if not len(candidates):return None,None,dict(detail,applied=False)
        # Prefer the best observed containment, then the smallest enclosing
        # entity. This is a geometric choice, not a semantic confidence weight.
        order=np.lexsort((candidates,self.reader.area[candidates],-coverage[self.frames[candidates]]))
        entity=int(candidates[order[0]])
        _,rank,selected,matched=self.reader.infer(np.zeros(self.membership.shape[1]),semantic_rank,seed=entity)
        detail.update(entity_seed_region=entity,matched_views=matched['matched_views'],
            matched_regions=matched['matched_regions'])
        if matched['matched_views']<2:return None,None,dict(detail,applied=False)
        return rank,selected['identity_scope'],dict(detail,applied=True,
            selection_rule='entity observations with co-visible physical correspondence and visible majority')
