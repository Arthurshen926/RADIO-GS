"""Bind one native region identity to physically corresponding observations.

No query vocabulary, benchmark target, source photograph or learned coordinate
projection enters this solver. Overlap is measured on jointly visible elements;
one region per view avoids mixing a whole object and its parts in that view.
"""
import numpy as np

from .observation_readout import project_observations, positive_support_rank


class InstanceObservationScope:
    def __init__(self, membership, visibility, region_frames, positive_membership=None,surface_consistency=None,
                 exclusion_policy='visible_complement'):
        self.membership=membership.tocsc()
        self.positive_membership=(positive_membership if positive_membership is not None else membership).tocsc()
        self.visibility=visibility.tocsc()
        self.frames=region_frames
        self.area=np.asarray(membership.sum(0)).reshape(-1)
        self.surface=surface_consistency
        self.exclusion_policy=exclusion_policy

    def infer(self, identity, semantic_rank, seed=None):
        available=np.flatnonzero(self.area>0)
        if seed is None:seed=int(available[np.argmax(identity[available])])
        seed_frame=int(self.frames[seed]);scope=self.membership[:,seed]
        intersection=(self.membership.T@scope).toarray().reshape(-1)
        candidates=np.flatnonzero((intersection>0)&(self.frames!=seed_frame))
        weights=np.zeros(self.membership.shape[1],np.float32);weights[seed]=1.
        confidence=np.zeros(self.visibility.shape[1],np.float32);confidence[seed_frame]=1.
        completeness=np.ones(self.visibility.shape[1],np.float32)
        if len(candidates):
            candidate_shared=(self.membership[:,candidates].T@self.visibility[:,seed_frame]).toarray().reshape(-1)
            seed_shared=(self.visibility.T@scope).toarray().reshape(-1)[self.frames[candidates]]
            shared=intersection[candidates]
            iou=shared/np.maximum(candidate_shared+seed_shared-shared,1.)
            compatible=(candidate_shared>=.5*self.area[candidates])&(seed_shared>=.5*self.area[seed])&(iou>=.5)
            purity=shared/np.maximum(candidate_shared,1.)
            recall=shared/np.maximum(seed_shared,1.)
            candidates=candidates[compatible];iou=iou[compatible];purity=purity[compatible];recall=recall[compatible]
            seed_completeness=[]
            for frame in np.unique(self.frames[candidates]):
                local=np.flatnonzero(self.frames[candidates]==frame)
                selected=local[np.argmax(iou[local])];region=candidates[selected]
                weights[region]=iou[selected];confidence[frame]=iou[selected]
                completeness[frame]=recall[selected]
                seed_completeness.append(purity[selected])
            if seed_completeness:completeness[seed_frame]=float(np.mean(seed_completeness))
        return self.decode(identity,semantic_rank,seed,weights,confidence,completeness)

    def decode(self, identity, semantic_rank, seed, weights, confidence,completeness=None):
        """Read a selected set of view-exclusive observations on any carrier."""
        seed_frame=int(self.frames[seed])
        positive=np.asarray(project_observations(self.positive_membership,weights)).reshape(-1)
        known=np.asarray(project_observations(self.visibility,confidence)).reshape(-1)
        if self.exclusion_policy=='covisible_completeness' and completeness is not None:
            # A partial observation cannot exclude portions it failed to cover.
            # Only exclusion mass is discounted; no new positive, association,
            # semantic response or unseen geometry is invented. The degree of
            # incompleteness is measured in the other view's shared domain.
            exclusion=np.asarray(project_observations(self.visibility,confidence*completeness)).reshape(-1)
            exclusion-=np.asarray(project_observations(self.positive_membership,weights*completeness[self.frames])).reshape(-1)
            known=positive+np.maximum(exclusion,0)
        probability=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
        surface_detail=None
        scope_selection=probability>.5
        if self.surface is not None:
            probability,scope_selection,surface_detail=self.surface.readout(np.clip(probability,0,1),np.flatnonzero(known>0))
        # Partial scope supplies positive support, not a license to exclude its
        # complement. Keep the native ranking outside the scope and retain native
        # ordering inside equal-support regions. This bounded score is not a
        # calibrated probability: s + s(1-s)p, with no fitted scale or threshold.
        seed_support=self.positive_membership[:,seed].toarray().reshape(-1)
        seed_rank=positive_support_rank(semantic_rank,seed_support)
        scope_rank=positive_support_rank(semantic_rank,np.clip(probability,0,1))
        selected={'identity_region':seed_support>.5,
                  'identity_scope':scope_selection}
        return seed_rank,scope_rank,selected,dict(seed_region=seed,seed_frame=seed_frame,
            seed_native_similarity=float(identity[seed]),seed_elements=int(self.area[seed]),
            matched_regions=np.flatnonzero(weights).tolist(),matched_views=int((confidence>0).sum()),
            query_supported_vertices=int((known>0).sum()),
            scope_rule='one region per view, co-visible IoU >= .5 and bilateral visible fraction >= .5',
            semantic_coordinates_changed=False,calibrated_membership=False,
            absent_scope_policy='native ranking prior retained; visible-mask exclusions are estimates, not verified negatives',
            ranking_rule='s + s(1-s)p; native floor and order preserved at equal support',
            selection_rule=('shared surface '+self.surface.solver+' with original .5 decision prior' if self.surface is not None else
                'seed membership or strict majority of confidence-weighted visible observations'),
            surface_consistency=surface_detail,
            exclusion_policy=self.exclusion_policy,
            selection_is_independent_of_rank_threshold=True)
