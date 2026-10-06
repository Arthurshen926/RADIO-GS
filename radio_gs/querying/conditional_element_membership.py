"""Decode instance-conditioned elements, without turning missing masks into negatives.

The region model supplies evidence channels. Only independent instance labels
train the final element decision; no region purity is itself a membership label.
"""
import numpy as np
from scipy.spatial import cKDTree

from .conditional_region_membership import RegionPairFeatures
from .observation_readout import positive_support_rank
from .anchor_observation_evidence import joint_anchor_evidence, JOINT_FEATURE_NAMES


FEATURE_NAMES = [
    'seed_member','candidate_count_fraction','candidate_view_visibility',
    'mean_purity','max_purity','mean_visible_recall','max_visible_recall',
    'mean_added_purity','max_added_purity','mean_appearance','max_appearance',
    'purity_weighted_recall','high_purity_fraction','nearest_seed_distance',
    'seed_centroid_distance','absolute_height_difference',
]
LOCAL_FEATURE_NAMES=['element_to_seed_region_cosine','element_to_seed_surface_cosine',
                     'appearance_view_coherence','appearance_observed_fraction']
ANCHOR_FEATURE_NAMES=['element_to_anchor_cosine','anchor_distance','anchor_horizontal_distance','anchor_height_offset']


class ElementMembershipEvidence:
    def __init__(self, membership, visibility, frames, xyz, descriptors, region_model,
                 local_appearance=None,local_observation=None,condition_on_anchor=False,joint_observations=False,
                 region_condition_on_anchor=False):
        self.regions=RegionPairFeatures(membership,visibility,frames,xyz,descriptors)
        self.xyz=np.asarray(xyz,np.float32)
        self.region_model=region_model
        self.anchors=membership.tocsr()
        self.local_appearance=local_appearance;self.local_observation=local_observation
        self.condition_on_anchor=condition_on_anchor
        self.joint_observations=joint_observations
        self.region_condition_on_anchor=region_condition_on_anchor

    def seed(self, identity, anchor):
        f=self.regions
        if anchor is not None:
            available=self.anchors[anchor].indices
            return int(available[np.argmin(f.area[available])]) if len(available) else None
        available=np.flatnonzero(f.area>0)
        return int(available[np.argmax(identity[available])])

    def encode(self, seed, anchor=None):
        f=self.regions
        ids,pair_features,_=(f.anchor_pairs(seed,anchor,self.local_appearance)
                            if self.region_condition_on_anchor else f.pairs(seed))
        # The corrected region head predicts even the seed's quality: a mask
        # containing the anchor can mix instances. Legacy arms retain their old
        # seed prior only as explicit experimental controls.
        prediction=np.clip(self.region_model.predict(pair_features),0,1) if len(ids) else np.zeros((0,3))
        appearance=(pair_features[:,7]+1)*.5
        if not self.region_condition_on_anchor:
            ids=np.r_[ids,seed];prediction=np.vstack([prediction,[1.,0.,1.]])
            appearance=np.r_[appearance,1.]
        membership=f.membership[:,ids].tocsr()
        support=np.flatnonzero(np.asarray(membership.sum(1)).ravel()>0)
        b=membership[support];count=np.asarray(b.sum(1)).ravel();den=np.maximum(count,1)
        seed_members=f.membership[:,seed].indices
        in_seed=np.isin(support,seed_members).astype(np.float32)
        frames=np.unique(f.frames[ids])
        visible=np.asarray(f.visibility[support][:,frames].sum(1)).ravel()/len(frames)
        columns=[in_seed,count/len(ids),visible]
        for values in [prediction[:,0],prediction[:,1],prediction[:,2],appearance]:
            columns.extend([np.asarray(b@values).ravel()/den,b.multiply(values).max(1).toarray().ravel()])
        columns.extend([np.asarray(b@(prediction[:,0]*prediction[:,1])).ravel()/den,
                        np.asarray(b@(prediction[:,0]>=.8)).ravel()/den])
        radius=max(f.radius[seed],.01)
        points=self.xyz[support];seed_xyz=self.xyz[seed_members];centre=seed_xyz.mean(0)
        nearest=cKDTree(seed_xyz).query(points)[0]/radius
        distance=np.linalg.norm(points-centre,axis=1)/radius
        height=np.abs(points[:,2]-centre[2])/radius
        # Bounded distances preserve order without using room-specific scales.
        columns.extend([nearest/(1+nearest),distance/(1+distance),height/(1+height)])
        if self.local_appearance is not None:
            local=np.asarray(self.local_appearance[support],np.float32)
            prototype=np.asarray(self.local_appearance[seed_members],np.float32).sum(0)
            prototype/=max(np.linalg.norm(prototype),1e-8)
            columns.extend([local@f.descriptors[seed],local@prototype,
                            self.local_observation[support,0],self.local_observation[support,1]])
        if self.condition_on_anchor:
            delta=(points-self.xyz[anchor])/radius
            length=np.linalg.norm(delta,axis=1);horizontal=np.linalg.norm(delta[:,:2],axis=1)
            columns.extend([local@np.asarray(self.local_appearance[anchor],np.float32),
                            length/(1+length),horizontal/(1+horizontal),delta[:,2]/(1+np.abs(delta[:,2]))])
        if self.joint_observations:
            columns.extend(joint_anchor_evidence(self.anchors,f.visibility.tocsr(),f.frames,anchor,support).T)
        return support,np.column_stack(columns).astype(np.float32)


class ConditionalElementMembership:
    def __init__(self, membership, visibility, frames, xyz, descriptors, checkpoint,
                 local_appearance=None,local_observation=None,surface_consistency=None):
        import joblib
        self.model=joblib.load(checkpoint)
        self.surface=surface_consistency
        self.evidence=ElementMembershipEvidence(membership,visibility,frames,xyz,descriptors,
                                               self.model['region_estimator'],local_appearance,local_observation,
                                               self.model.get('condition_on_anchor',False),self.model.get('joint_observations',False),
                                               self.model.get('region_condition_on_anchor',False))

    def infer(self, identity, semantic_rank, anchor=None):
        if self.evidence.condition_on_anchor and anchor is None:
            return None,None,dict(applied=False,reason='model trained for exact positive-point identity; retain native semantic scope')
        seed=self.evidence.seed(identity,anchor)
        if seed is None:return None,None,dict(applied=False,reason='no anchor observation')
        vertices,features=self.evidence.encode(seed,anchor)
        probability=np.zeros(self.evidence.anchors.shape[0],np.float32)
        probability[vertices]=self.model['estimator'].predict_proba(features)[:,1]
        if anchor is not None:probability[anchor]=1.
        surface_detail=None
        selected=probability>self.model['threshold']
        if self.surface is not None:
            probability,selected,surface_detail=self.surface.readout(probability,vertices,anchor,threshold=self.model['threshold'])
        rank=probability if anchor is not None else positive_support_rank(semantic_rank,probability)
        return rank,selected,dict(applied=True,seed_region=seed,
            candidate_elements=len(vertices),threshold=self.model['threshold'],
            training_condition=self.model['condition'],source_rgb_read=False,native_identity_changed=False,
            probability_calibrated_on_benchmark=False,
            surface_consistency=surface_detail,
            output='independent element supervision; no hard region admission or per-view majority')
