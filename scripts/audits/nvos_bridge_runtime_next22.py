"""Pure explicit-score reference completion runtime; no cross-query cache.

scores must be the caller's query-bound prompt_sam element probabilities BEFORE
component pruning. This routine never constructs/reads target RGB or GT, and
never changes persistent field weights. Callers bind source scores and masks to
geometry, field, positive prompt, SAM mask and registered parameter identities.
"""
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from gsplat import rasterization
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.prompt_transfer_20260919_evaluate import refine as refine_positive


def load_bound_query_state(path, expected_binding, *, device='cpu'):
 path=Path(path);binding=json.loads(Path(str(path)+'.binding.json').read_text())
 required={'field_sha256','geometry_sha256','positive_sha256','sam_mask_sha256','query_receipt_sha256','parameters_sha256','stage'}
 if set(expected_binding)!=required:raise ValueError('Explicit complete query identity required')
 if any(binding[k]!=expected_binding[k] for k in required):raise ValueError('Query-state binding mismatch')
 if sha256_file(path)!=binding['state_sha256']:raise ValueError('Query-state content changed')
 payload=torch.load(path,map_location=device,weights_only=False)
 return payload['scores'],binding


def complete_reference_query(scores, observed, geo, camera, prior, width, height, positive, sam, mode, field, *, base_stage, enable_bridge=False):
 from scipy.spatial import cKDTree
 from scipy.sparse import coo_matrix
 from scipy.sparse.csgraph import connected_components
 from torch.nn import functional as F
 from radio_gs.querying.query_conditioned_diffusion import QueryConditionedDiffusionConfig,solve_continuous_query_support
 if base_stage!='prompt_sam':raise ValueError('Completion requires pre-component prompt_sam state, not a pruned output')
 if mode not in ('reference_component','reference_core_component'):raise ValueError('Unregistered completion mode')
 if scores.shape!=(len(geo.xyz),1) or not torch.isfinite(scores).all():raise ValueError('One finite bound base score per element required')
 if positive.dtype!=torch.bool or sam.dtype!=torch.bool or positive.shape!=(height,width) or sam.shape!=positive.shape:raise ValueError('Reference raster mismatch')
 base=scores.detach()
 base_record=dict(provided_explicit_base_scores=True,global_cache=False)
 ids=torch.where(observed)[0];xyz=geo.xyz[ids].cpu().numpy()
 dist,neigh=cKDTree(xyz).query(xyz,k=17,workers=4);dist=dist[:,1:];neigh=neigh[:,1:]
 radius=3*float(np.median(dist[:,0][dist[:,0]>0]));neighbors=torch.from_numpy(neigh.copy()).long().to(scores.device)
 structure=F.normalize(field.latent.detach()[ids,field.native_semantic_dim:],dim=1)
 cosine=torch.empty_like(neighbors,dtype=torch.float32)
 for start in range(0,len(ids),8192):
  end=min(len(ids),start+8192);cosine[start:end]=(structure[start:end,None,:]*structure[neighbors[start:end]]).sum(-1)
 spatial=torch.from_numpy(dist<=radius).to(scores.device)
 edge=spatial
 mask_np=sam.cpu().numpy().astype(np.uint8);component_count,component_labels=cv2.connectedComponents(mask_np,connectivity=8)
 positive_labels=np.unique(component_labels[positive.cpu().numpy()&mask_np.astype(bool)]);positive_labels=positive_labels[positive_labels!=0]
 connected_sam=np.isin(component_labels,positive_labels)
 if mode=='reference_core_component':connected_sam=cv2.erode(connected_sam.astype(np.uint8),np.ones((5,5),np.uint8),iterations=1).astype(bool)
 admitted_reference=torch.from_numpy(connected_sam).to(scores.device)
 k=camera.intrinsic.to(scores.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
 with torch.enable_grad():
  value=base.detach().clone().requires_grad_(True)
  colors=torch.cat([value,observed.float()[:,None]],-1)
  rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.to(scores.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  mass=rendered[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
  response=torch.where(valid,rendered[0,...,0]/mass.clamp_min(1e-8),0)
  positive_mass=torch.autograd.grad(response[positive.to(scores.device)&valid].sum(),value,retain_graph=True)[0].flatten()
  total_mass=torch.autograd.grad(response[valid].sum(),value,retain_graph=True)[0].flatten()
  sam_mass=torch.autograd.grad(response[sam.to(scores.device)&valid].sum(),value,retain_graph=True)[0].flatten()
  reference_mass=torch.autograd.grad(response[admitted_reference&valid].sum(),value)[0].flatten()
 sam_fraction=sam_mass/total_mass.clamp_min(1e-8)
 seed_fraction=positive_mass/total_mass.clamp_min(1e-8);seeds=(seed_fraction[ids]>=.1)&(total_mass[ids]>1e-8)
 original_seeds=seeds.clone()
 reference_fraction=reference_mass/total_mass.clamp_min(1e-8)
 if mode!='baseline':seeds=seeds|((reference_fraction[ids]>=(.9 if mode=='reference_core_component' else .5))&(total_mass[ids]>1e-8))
 unary=base[ids,0]
 if mode=='structure_smooth_component':
  cfg=QueryConditionedDiffusionConfig(kernel='continuous_convex_v2',edge_binarize_threshold=None,laplacian_weight=1.,cg_iterations=64)
  affinity=edge.float()*torch.exp(-(1-cosine).clamp_min(0)/.5)
  unary=solve_continuous_query_support(unary,torch.full_like(unary,.5),neighbors,affinity,unary,config=cfg)
 candidate=(unary>=.5)|seeds
 from radio_gs.querying.uncertain_scope_bridge import candidate_bridge_edges
 accepted,bridge=candidate_bridge_edges(candidate,unary,neighbors,edge,enabled=enable_bridge)
 rows=torch.arange(len(ids),device=scores.device)[:,None].expand_as(neighbors)[accepted].cpu().numpy();cols=neighbors[accepted].cpu().numpy()
 graph=coo_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),shape=(len(ids),len(ids))).tocsr()
 count,labels=connected_components(graph,directed=False);seed_labels=np.unique(labels[seeds.cpu().numpy()]);keep=torch.from_numpy(np.isin(labels,seed_labels)).to(scores.device)&candidate
 filtered=base.clone();filtered[ids,0]=torch.where(keep,unary,torch.zeros_like(unary))
 # No geometry-supported anchors -> abstain, never fabricate all-negative output.
 if not seeds.any():filtered=base.clone()
 final,repair=refine_positive(filtered,observed,geo,camera,prior,width,height,positive,sam,'prompt_only')
 def diagnostic_reference(current):
  r,al,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=torch.cat([current,observed.float()[:,None]],-1),viewmats=camera.world_to_camera.to(scores.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
  probability=torch.where(valid,r[0,...,0]/r[0,...,1].clamp_min(1e-8),0)
  return dict(positive_agreement=float((probability[positive.to(scores.device)]>=.5).float().mean()),sam_positive_recall=float((probability[sam.to(scores.device)]>=.5).float().mean()))
 record=dict(mode=mode,bridge_enabled=enable_bridge,uncertain_connectors=int(bridge.sum()),connectors_are_not_output_candidates=True,additional_seed_elements=int((seeds&~original_seeds).sum()),sam_2d_components=int(component_count-1),prompt_connected_sam_components=len(positive_labels),admitted_reference_pixels=int(admitted_reference.sum()),reference_bridge_is_verified_identity=False,removed_sam_majority_elements=int(((sam_fraction[ids]>=.5)&candidate&~keep).sum()),sam_majority_candidate_elements=int(((sam_fraction[ids]>=.5)&candidate).sum()),sam_evidence_mass_removed_fraction=float(sam_mass[ids][candidate&~keep].sum()/sam_mass[ids][candidate].sum().clamp_min(1e-8)),filter_selected_elements=int((filtered>=.5).sum()),repaired_selected_elements=int((final>=.5).sum()),reintroduced_elements=int(((final>=.5)&(filtered<.5)).sum()),reference_after_filter=diagnostic_reference(filtered),reference_after_repair=diagnostic_reference(final),base_refinement=base_record,positive_repair=repair,radius=radius,structure_dimension=structure.shape[1],seed_elements=int(seeds.sum()),components=int(count),seeded_components=len(seed_labels),candidate_elements=int(candidate.sum()),retained_elements=int(keep.sum()),edges=int(accepted.sum()),mean_absolute_score_change_from_baseline=float((final-base).abs().mean()),abstained_no_seeds=not bool(seeds.any()),sam_outside_negative=False)
 return final,record

