"""Binary Gaussian regional scope for an explicit cross-carrier experiment.

Boolean region membership and visibility are distinct from projected evidence
mass. Binary semantics do not imply mesh/Gaussian feature distribution parity.
Runtime reads only buffers in the same CapabilityField; scipy state is ephemeral.
"""
import numpy as np
from scipy import sparse
import torch
from torch import nn
from .region_quality import RegionQuality
from .region_admission import admit_regions
from radio_gs.querying.conditional_region_membership import RegionPairFeatures


class RegionCarrier(nn.Module):
    def __init__(self,state,count):
        super().__init__();self.count=count
        q={k.removeprefix('quality.'):v for k,v in state.items() if k.startswith('quality.')}
        self.quality=RegionQuality(q)
        base={k:v for k,v in state.items() if not k.startswith('quality.')}
        required={'frames','descriptors','view_count'}|{p+'_'+s for p in ['membership','visibility'] for s in ['pointer','index']}
        if set(base)!=required:raise ValueError('incomplete Gaussian region carrier')
        for k,v in base.items():self.register_buffer(k,v.detach())
        if self.frames.ndim!=1 or self.frames.dtype!=torch.long or self.descriptors.shape!=(len(self.frames),768) or self.view_count.ndim!=0 or int(self.view_count)<=0:
            raise ValueError('invalid regional metadata')
        if not bool(torch.isfinite(self.descriptors).all()) or bool(((self.frames<0)|(self.frames>=self.view_count)).any()):raise ValueError('invalid regional descriptors/frames')
        for p,width in [('membership',len(self.frames)),('visibility',int(self.view_count))]:
            ptr=getattr(self,p+'_pointer');idx=getattr(self,p+'_index')
            if ptr.dtype!=torch.long or idx.dtype!=torch.long or ptr.shape!=(count+1,) or idx.ndim!=1 or int(ptr[0])!=0 or int(ptr[-1])!=len(idx) or bool((ptr.diff()<0).any()) or bool(((idx<0)|(idx>=width)).any()):raise ValueError('invalid binary regional incidence')

    def matrix(self,prefix):
        ptr=getattr(self,prefix+'_pointer').cpu().numpy();idx=getattr(self,prefix+'_index').cpu().numpy()
        width=len(self.frames) if prefix=='membership' else int(self.view_count)
        return sparse.csr_matrix((np.ones(len(idx),np.float32),idx,ptr),shape=(self.count,width))

    def decoder(self,xyz):
        return RegionScopeDecoder(self,xyz)


class RegionScopeDecoder:
    def __init__(self,carrier,xyz):
        self.carrier=carrier
        self.features=RegionPairFeatures(carrier.matrix('membership'),carrier.matrix('visibility'),
            carrier.frames.cpu().numpy(),xyz.detach().cpu().numpy(),carrier.descriptors.float().cpu().numpy().copy())

    @torch.no_grad()
    def scope(self,probability):
        f=self.features;c=self.carrier
        if probability.shape!=(len(f.area),) or not bool(torch.isfinite(probability).all()):raise ValueError('unaligned region identity')
        eligible=np.flatnonzero((f.area>0)&(probability.detach().cpu().numpy()>.5))
        if not len(eligible):return None,dict(applied=False,reason='no native positive region')
        seed=int(eligible[np.argmax(probability.detach().cpu().numpy()[eligible])])
        ids,x,shared=f.pairs(seed);device=c.frames.device
        prediction=c.quality(torch.from_numpy(x).to(device))
        result=admit_regions(c.quality,prediction,seed=seed,candidates=torch.from_numpy(ids).to(device),
            intersection=torch.from_numpy(shared).to(device),area=torch.from_numpy(f.area).to(device),
            frames=c.frames,view_count=int(c.view_count))
        if result is None:return None,dict(applied=False,seed=seed,reason='no quality-admitted extension')
        weights,confidence,chosen=result
        positive=np.asarray(f.membership@weights.cpu().numpy()).ravel()
        known=np.asarray(f.visibility@confidence.cpu().numpy()).ravel()
        scope=np.divide(positive,known,out=np.zeros_like(positive),where=known>0).clip(0,1)
        return (torch.from_numpy(scope),torch.from_numpy(known>0)),dict(applied=True,seed=seed,
            admitted=chosen.cpu().tolist(),candidate_pairs=len(ids),views=int((confidence>0).sum()),
            mesh_to_gaussian_distribution_shift=True,source_access=False,query_fitting=False)


@torch.no_grad()
def query_text_region_scope(field,query,negatives,*,region_decoder=None,exclusion_decoder=None):
    from .scope_confidence import native_text_evidence,native_scope_confidence
    from .joint_exclusion import JointExclusionDecoder
    if field.region_carrier is None:raise ValueError('field has no Gaussian region carrier')
    decoder=region_decoder or field.region_carrier.decoder(field.xyz)
    if decoder.carrier is not field.region_carrier:raise ValueError('region decoder belongs to another field')
    exclusion=exclusion_decoder or JointExclusionDecoder(field)
    if exclusion.field is not field:raise ValueError('exclusion decoder belongs to another field')
    initial,probability=native_text_evidence(field,query,negatives)
    accepted,_=native_scope_confidence(field,initial,probability)
    fixed,_=exclusion.decode(initial,accepted,joint=False)
    scope,detail=decoder.scope(probability)
    values=dict(semantic_initial=initial,frozen_visibility=fixed,seed_addition=fixed.clone(),
        conditional_addition=fixed.clone(),conditional_replace=initial.clone())
    if 'seed' in detail:
        mask=torch.from_numpy(decoder.features.membership[:,detail['seed']].toarray().ravel()>0).to(initial.device)
        values['seed_addition'][mask]=1.
    if scope is not None:
        member,known=[x.to(initial.device) for x in scope]
        values['conditional_addition'][member>.5]=1.
        values['conditional_replace'][known]=member[known]
    available=field.observed|field.semantic_measure.supported
    for k in values:values[k]=values[k].masked_fill(~available,0)
    return values,dict(detail,source_access=False,query_parameter_fitting=False,
        single_seed_scope=True,unknown_preserves_native_prior=True,default_promoted=False)
