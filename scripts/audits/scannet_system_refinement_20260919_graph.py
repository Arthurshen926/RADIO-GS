"""Registered fixed-capacity graph correction of bounded native/scope logits."""
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch.nn import functional as F
from radio_gs.querying.query_conditioned_diffusion import QueryConditionedDiffusionConfig,solve_continuous_query_support,knn_feature_distances

def graph_refine(model,geo,native,bounded):
 ids=torch.where(model.observed)[0];xyz=geo.xyz[ids].cpu().numpy();dist,index=cKDTree(xyz).query(xyz,k=17,workers=4)
 # Explicit self removal handles tied positions without assuming self is first.
 clean_index=np.empty((len(ids),16),dtype=np.int64);clean_dist=np.empty((len(ids),16),dtype=np.float32)
 for i in range(len(ids)):
  keep=index[i]!=i;clean_index[i]=index[i][keep][:16];clean_dist[i]=dist[i][keep][:16]
 nb=torch.from_numpy(clean_index).cuda();distance=torch.from_numpy(clean_dist).cuda();sigma=distance[:,-1].median().clamp_min(1e-8);spatial=torch.exp(-distance.square()/(2*sigma.square()));features=F.normalize(model.latent[ids,64:],dim=1);fd=knn_feature_distances(features,nb,distance_chunk_size=1024).square();positive=fd[fd>0];fs=positive.median().clamp_min(1e-8) if positive.numel() else torch.tensor(1.,device='cuda');structure=spatial*torch.exp(-fd/fs)
 config=QueryConditionedDiffusionConfig(kernel='continuous_convex_v2',edge_binarize_threshold=None,laplacian_weight=1.,cg_iterations=64,cg_tolerance=1e-5,hard_observation_confidence=.99);confidence=torch.full((len(ids),),.5,device='cuda');prob=bounded[ids].sigmoid();eps=1e-6;original_logodds=torch.logit(prob.clamp(eps,1-eps));results={};stats={}
 for name,aff in [('spatial_graph',spatial),('structure32_graph',structure)]:
  updated=bounded.clone();shifts=[]
  for qi in range(prob.shape[1]):
   proposal=solve_continuous_query_support(prob[:,qi],confidence,nb,aff,prob[:,qi],config=config);delta=torch.logit(proposal.clamp(eps,1-eps))-original_logodds[:,qi];updated[ids,qi]=native[ids,qi]+(bounded[ids,qi]+delta-native[ids,qi]).clamp(-1,1);shifts.append(float(delta.abs().mean()))
  results[name]=updated;stats[name]=dict(mean_abs_logodds_correction=float(np.mean(shifts)),mean_affinity=float(aff.mean()))
 identity_delta=torch.logit(prob.clamp(eps,1-eps))-original_logodds;assert torch.count_nonzero(identity_delta)==0
 return results,dict(neighbors=16,lambda_value=1.,confidence=.5,hard_nodes=0,observed_nodes=len(ids),spatial_sigma=float(sigma),feature_squared_distance_median=float(fs),clamp_epsilon=eps,saturated_fraction=float(((prob<=eps)|(prob>=1-eps)).float().mean()),identity_logodds_delta_exact=True,arms=stats)
