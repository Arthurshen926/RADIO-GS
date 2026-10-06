"""Whole-instance correspondence supervised separately from native identity.

The model estimates candidate purity, visible completeness and, separately,
purity of the elements that would be ADDED to a seed. A large shared core cannot
hide an incorrect extension. Its inputs contain no query/category/scene IDs.
It operates on regional associations of a frozen carrier, not teacher vectors
as foreground probabilities. Returned scope votes remain uncalibrated rankings.
"""
import numpy as np

from .instance_observation_scope import InstanceObservationScope
from .observation_readout import positive_support_rank


FEATURE_NAMES = [
    'intersection_over_seed','intersection_over_candidate',
    'seed_visible_in_candidate','candidate_visible_in_seed','covisible_iou',
    'log_area_ratio','same_view','native_appearance_cosine',
    'relative_centroid_distance','relative_seed_radius',
    'seed_fraction_of_view','candidate_fraction_of_view',
    'seed_fraction_of_scene','candidate_fraction_of_scene',
    'seed_relative_radius','candidate_relative_radius',
]
ANCHOR_REGION_FEATURE_NAMES = ['anchor_in_region','anchor_visible_in_region_view',
    'anchor_to_region_distance','anchor_to_region_horizontal_distance',
    'anchor_to_region_height','anchor_to_region_native_cosine']


class RegionPairFeatures:
    def __init__(self, membership, visibility, frames, xyz, descriptors):
        self.membership=membership.tocsc();self.visibility=visibility.tocsc()
        self.frames=np.asarray(frames);self.area=np.asarray(membership.sum(0)).ravel()
        self.visible_area=np.asarray(visibility.sum(0)).ravel()
        self.descriptors=np.asarray(descriptors,dtype=np.float32)
        self.descriptors/=np.maximum(np.linalg.norm(self.descriptors,axis=1,keepdims=True),1e-8)
        xyz=np.asarray(xyz,dtype=np.float64);xyz=xyz-xyz.mean(0)
        self.centre=np.asarray(membership.T@xyz)/np.maximum(self.area[:,None],1)
        second=np.asarray(membership.T@(xyz*xyz).sum(1)).ravel()/np.maximum(self.area,1)
        self.radius=np.sqrt(np.maximum(second-(self.centre*self.centre).sum(1),0))
        self.scene_radius=np.sqrt(np.mean((xyz*xyz).sum(1)))
        self.xyz=xyz

    def pairs(self, seed):
        scope=self.membership[:,seed]
        intersection=(self.membership.T@scope).toarray().ravel()
        ids=np.flatnonzero((intersection>0)&(np.arange(len(self.area))!=seed))
        shared=intersection[ids];a=max(self.area[seed],1);b=np.maximum(self.area[ids],1)
        seed_visible=(self.visibility.T@scope).toarray().ravel()[self.frames[ids]]
        candidate_visible=(self.membership[:,ids].T@self.visibility[:,self.frames[seed]]).toarray().ravel()
        total_radius=self.radius[seed]+self.radius[ids]+1e-6
        x=np.column_stack([
            shared/a,shared/b,seed_visible/a,candidate_visible/b,
            shared/np.maximum(seed_visible+candidate_visible-shared,1),
            np.log(b/a),self.frames[ids]==self.frames[seed],
            self.descriptors[ids]@self.descriptors[seed],
            np.linalg.norm(self.centre[ids]-self.centre[seed],axis=1)/total_radius,
            self.radius[seed]/total_radius,
            np.full(len(ids),a/max(self.visible_area[self.frames[seed]],1)),
            b/np.maximum(self.visible_area[self.frames[ids]],1),
            np.full(len(ids),a/self.membership.shape[0]),b/self.membership.shape[0],
            np.full(len(ids),self.radius[seed]/max(self.scene_radius,1e-6)),
            self.radius[ids]/max(self.scene_radius,1e-6),
        ]).astype(np.float32)
        return ids,x,intersection[ids]

    def anchor_pairs(self,seed,anchor,local_appearance):
        """Condition every region, including the seed, on the actual point.

        A seed mask's majority instance need not contain the positive point.
        Its own purity is predicted rather than being manufactured as one.
        """
        ids,x,shared=self.pairs(seed);area=self.area[seed]
        fraction=area/max(self.visible_area[self.frames[seed]],1)
        relative=self.radius[seed]/max(self.scene_radius,1e-6)
        self_pair=np.array([1,1,1,1,1,0,1,1,0,.5,fraction,fraction,
                            area/self.membership.shape[0],area/self.membership.shape[0],relative,relative])
        ids=np.r_[ids,seed];x=np.vstack([x,self_pair]);shared=np.r_[shared,area]
        delta=(self.centre[ids]-self.xyz[anchor])/np.maximum(self.radius[ids,None],.01)
        distance=np.linalg.norm(delta,axis=1);horizontal=np.linalg.norm(delta[:,:2],axis=1)
        point=np.asarray(local_appearance[anchor],np.float32)
        exact=np.column_stack([self.membership[anchor,ids].toarray().ravel(),
            self.visibility[anchor,self.frames[ids]].toarray().ravel(),
            distance/(1+distance),horizontal/(1+horizontal),delta[:,2]/(1+np.abs(delta[:,2])),
            self.descriptors[ids]@point])
        return ids,np.column_stack([x,exact]).astype(np.float32),shared


class ConditionalRegionMembership:
    def __init__(self, membership, visibility, frames, xyz, descriptors, checkpoint):
        import joblib
        self.features=RegionPairFeatures(membership,visibility,frames,xyz,descriptors)
        self.model=joblib.load(checkpoint)
        self.reader=InstanceObservationScope(membership,visibility,frames)
        self.anchors=membership.tocsr()

    def infer(self, identity, semantic_rank, anchor=None):
        f=self.features
        if anchor is None:
            available=np.flatnonzero(f.area>0);seed=int(available[np.argmax(identity[available])])
        else:
            available=self.anchors[anchor].indices
            if not len(available):return None,None,dict(applied=False,reason='no anchor observation')
            # A positive point defines identity. Start with its finest supported
            # region; the supervised completion model, not area maximization,
            # decides whether containing masks describe the same instance.
            seed=int(available[np.argmin(f.area[available])])
        ids,x,intersection=f.pairs(seed)
        prediction=np.clip(self.model['estimator'].predict(x),0,1) if len(ids) else np.zeros((0,3))
        precision,recall,added=prediction.T
        additions=f.area[ids]-intersection
        eligible=(added>=self.model['addition_threshold'])&(additions>0)
        # No-extension correspondences can complete unseen viewpoints without
        # a same-view region-size preference. Full candidate purity is separate.
        eligible|=(additions==0)&(precision>=self.model['addition_threshold'])
        eligible&=(f.frames[ids]!=f.frames[seed])|(additions>0)
        if not eligible.any():
            return None,None,dict(applied=False,seed_region=seed,reason='no admitted conditional observation')
        merit=precision*recall/np.maximum(precision+recall-precision*recall,1e-8)
        weights=np.zeros(len(f.area),np.float32);confidence=np.zeros(f.visibility.shape[1],np.float32)
        weights[seed]=1.;confidence[f.frames[seed]]=1.
        admitted=[]
        for frame in np.unique(f.frames[ids[eligible]]):
            local=np.flatnonzero(eligible&(f.frames[ids]==frame))
            best=local[np.argmax(merit[local])];region=int(ids[best])
            if frame==f.frames[seed]:weights[seed]=0
            weights[region]=float(precision[best]);confidence[frame]=float(precision[best])
            admitted.append(dict(region=region,precision=float(precision[best]),
                visible_recall=float(recall[best]),added_purity=float(added[best])))
        positive=np.asarray(f.membership@weights).ravel()
        known=np.asarray(f.visibility@confidence).ravel()
        support=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
        support=np.clip(support,0,1)
        if anchor is not None:support[anchor]=1.
        rank=support if anchor is not None else positive_support_rank(semantic_rank,support)
        return rank,support>.5,dict(applied=True,seed_region=seed,admitted=admitted,
            matched_views=int((confidence>0).sum()),candidate_pairs=len(ids),
            addition_threshold=self.model['addition_threshold'],
            training_condition=self.model['condition'],source_rgb_read=False,
            native_identity_changed=False,probability_calibrated_on_benchmark=False,
            output='strict majority of learned-admitted region observations; query identity remains separate')
