"""Correspondence identities from native patch descriptors, without RGB access.

The same operator works on regional observations of any geometric carrier.
Its scores select an identity; they are not membership/foreground probabilities.
"""
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def match_regions(features, region_weights, patch_offsets, region_offsets,
                  query_features, query_weights, device='cuda'):
    count=len(query_features);sizes=[len(q) for q in query_features];width=max(sizes)
    queries=torch.zeros(count,width,features.shape[1],device=device)
    mass=torch.zeros(count,width,device=device)
    for i,(values,weights) in enumerate(zip(query_features,query_weights)):
        queries[i,:len(values)]=F.normalize(torch.as_tensor(values,device=device).float(),dim=-1)
        mass[i,:len(values)]=torch.as_tensor(weights,device=device)
    mass/=mass.sum(1,keepdim=True)
    valid=mass>0
    output={key:np.zeros((region_weights.shape[0],count),np.float32) for key in ['chamfer','reciprocal']}
    qindex=torch.arange(width,device=device)[None,None,:]
    for frame in range(len(patch_offsets)-1):
        p0,p1=patch_offsets[frame:frame+2];r0,r1=region_offsets[frame:frame+2]
        if r0==r1:continue
        patch=F.normalize(torch.as_tensor(np.array(features[p0:p1]),device=device).float(),dim=-1)
        cosine=queries@patch.T
        cosine.masked_fill_(~valid[:,:,None],-2.)
        reverse,reverse_index=cosine.max(dim=1)
        weights=torch.as_tensor(region_weights[r0:r1,p0:p1].toarray(),device=device)
        weights/=weights.sum(1,keepdim=True)
        for start in range(0,len(weights),16):
            local=weights[start:start+16]
            # All nonempty patch cells participate. The chunk is a memory batch,
            # not a shortlist or an observation cap.
            scores=cosine[:,None].expand(-1,len(local),-1,-1)
            scores=scores.masked_fill(local[None,:,None,:]==0,-2.)
            forward,nearest=scores.max(dim=-1)
            forward_mean=(forward*mass[:,None,:]).sum(-1)
            reverse_mean=reverse@local.T
            output['chamfer'][r0+start:r0+start+len(local)]=(.5*(forward_mean+reverse_mean)).T.cpu().numpy()
            returned=torch.gather(reverse_index[:,None].expand(-1,len(local),-1),2,nearest)
            reciprocal=(returned==qindex)&valid[:,None,:]
            matched_mass=torch.gather(local[None].expand(count,-1,-1),2,nearest)
            pair_mass=torch.sqrt(mass[:,None,:]*matched_mass)
            # Reciprocal pairs are one-to-one. With unit total mass on each
            # side, their weighted similarity lies in [0,1]; no fitted threshold.
            coverage=(pair_mass*reciprocal*(forward.clamp(-1,1)+1)*.5).sum(-1)
            output['reciprocal'][r0+start:r0+start+len(local)]=coverage.T.cpu().numpy()
    return output
