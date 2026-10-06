"""Compile source-region pseudo supervision on the NEW Gaussian domain."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.region_visibility import normalize_visible_region_mass
from radio_gs.training.source_region_tracks import consistent_region_matches

def matches_by_view_pair(matches, view_ids):
 groups={}
 for match in matches:
  q,t,_=match;groups.setdefault((view_ids[q],view_ids[t]),[]).append(match)
 return groups

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--correspondence-policy',choices=['pairwise','scope_consistent'],default='pairwise')
 p.add_argument('--allow-missing-text-validation',action='store_true',help='Record absent legacy text validation evidence explicitly; never fabricate labels')
 p.add_argument('--source-split-diagnostic',type=Path,required=True);a=p.parse_args();root=a.root;device=a.device;torch.set_num_threads(4)
 if (root/'episodes.pt').exists():raise FileExistsError(root/'episodes.pt')
 info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
 assert sha256_file(root/'responsibility.json')==info['authority_sha256']
 def load(record):
  path=root/record['path'];assert sha256_file(path)==record['sha256'];return torch.load(path,map_location='cpu',weights_only=False)
 sam=load(info['teacher_files']['sam']);sig=load(info['teacher_files']['siglip']);queries=load(info['query_file']);dino=load(info['teacher_files']['dino'])
 frames=info['frames'];h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width'];n=auth['num_gaussians']
 masks=sam['masks'].to(device).reshape(len(sam['masks']),-1);views=sam['view_indices'].to(device)
 desc=F.normalize(sig['descriptors'].float().to(device),dim=1);text=F.normalize(sig['text_embeddings'].float().to(device),dim=1)
 ddesc=F.normalize(queries['region_dino'].float().to(device),dim=1)
 prob=torch.zeros(len(masks),n,device=device);seen=[];mats=[];pixel_mass=[];hits_by_view=[]
 for vi,r in enumerate(auth['views']):
  path=root/r['relative_path'];assert sha256_file(path)==r['sha256'];hit=torch.load(path,map_location=device,weights_only=False)
  pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
  mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce()
  rowsum=torch.sparse.sum(mat,dim=0).to_dense();selected=torch.where(views==vi)[0]
  lifted,visible=normalize_visible_region_mass(torch.sparse.mm(mat.T,masks[selected].float().T),rowsum)
  prob[selected]=lifted.T
  seen.append(visible);mats.append(mat);pixel_mass.append(mass);hits_by_view.append(hit)
 seen=torch.stack(seen);hard=prob>.7
 scores=desc@text.T;top=scores.topk(2,dim=1);labels=top.indices[:,0];margin=top.values[:,0]-top.values[:,1]
 matches=[]
 for vi in range(len(frames)):
  for vj in range(vi+1,len(frames)):
   left=torch.where(views==vi)[0];right=torch.where(views==vj)[0]
   shared=seen[vi]&seen[vj]
   l=hard[left][:,shared].float();r=hard[right][:,shared].float()
   inter=l@r.T;union=l.sum(1)[:,None]+r.sum(1)[None]-inter
   iou=inter/union.clamp_min(1)
   bestj=iou.argmax(1);besti=iou.argmax(0)
   for li,rj in enumerate(bestj.tolist()):
    if besti[rj]!=li or inter[li,rj]<20 or iou[li,rj]<.3:continue
    q,t=int(left[li]),int(right[rj])
    if desc[q]@desc[t]<.65 or ddesc[q]@ddesc[t]<.5:continue
    matches.extend([(q,t,float(iou[li,rj])),(t,q,float(iou[li,rj]))])
 # Dataset adaptation diagnostic: find source-only connected validation pairs.
 from itertools import combinations
 valid_prompt=[]
 for region,vi in enumerate(views.tolist()):
  visible=pixel_mass[vi]>.1
  valid_prompt.append(bool((masks[region]&visible).any() and ((~masks[region])&visible).any()))
 def split_counts(pair,inside):
  admitted=[(q,t) for q,t,_ in matches if ((int(views[q]) in pair) and (int(views[t]) in pair) if inside else (int(views[q]) not in pair) and (int(views[t]) not in pair))]
  return dict(image=len(admitted),prompt=sum(valid_prompt[q] for q,t in admitted),text=sum(bool(labels[q]==labels[t] and min(margin[q],margin[t])>=.001) for q,t in admitted))
 candidates=[]
 for pair in combinations(range(len(frames)),2):
  va=split_counts(pair,True);tr=split_counts(pair,False)
  if min(va.values())>0 and min(tr.values())>0:candidates.append(dict(validation_frames=sorted(frames[i] for i in pair),train_counts=tr,validation_counts=va))
 candidates.sort(key=lambda r:r['validation_frames'])
 if not candidates:raise ValueError('No source-only connected two-view validation split supports all three queries')
 original_pair=tuple(frames.index(f) for f in info['validation_frames'])
 original_counts=dict(validation=split_counts(original_pair,True),train=split_counts(original_pair,False))
 original_valid=all(min(r.values())>0 for r in original_counts.values())
 selected=next(r for r in candidates if r['validation_frames']==sorted(info['validation_frames'])) if original_valid else candidates[0]
 result=dict(original_counts=original_counts,original_split_retained=original_valid,source_only=True,benchmark_labels_used=False,teacher_manifest_sha256=sha256_file(root/'teachers.json'),selection='Retain original split if nonempty; otherwise lexicographically first two-frame pair with nonempty source geometry/appearance admitted text/image/prompt evidence in both splits; no trained-field metrics or benchmark labels.',candidates=candidates,selected=selected,implementation_sha256=sha256_file(__file__))
 if a.source_split_diagnostic.exists():raise FileExistsError(a.source_split_diagnostic)
 a.source_split_diagnostic.write_text(json.dumps(result,indent=2)+'\n');print(selected,flush=True);return
 # Pairwise mutual best links need not define consistent target identity.
 # Separate splits BEFORE graph reconciliation: validation must not choose
 # training correspondences. Mixed-split links never supply supervision.
 consistency={};filtered=[]
 for split,allowed in [('train',info['train_frames']),('validation',info['validation_frames'])]:
  selected=[(q,t,s) for q,t,s in matches if frames[int(views[q])] in allowed and frames[int(views[t])] in allowed]
  kept,consistency[split]=consistent_region_matches(selected,sam['masks'].to(device),views)
  filtered.extend(kept)
 if a.correspondence_policy=='scope_consistent':matches=filtered
 # A negative requires a second cross-view matched region, disjoint in BOTH
 # query and target views, with distinct appearance evidence. Not mere absence.
 episodes=[];view_ids=views.tolist();pair_matches=matches_by_view_pair(matches,view_ids)
 for q,t,overlap in matches:
  vi,vj=int(views[q]),int(views[t]);split='train' if frames[vi] in info['train_frames'] and frames[vj] in info['train_frames'] else 'validation' if frames[vi] in info['validation_frames'] and frames[vj] in info['validation_frames'] else None
  if split is None:continue
  neg=[]
  for nq,nt,_ in pair_matches[(vi,vj)]:
   if nq==q:continue
   if (masks[q]&masks[nq]).any() or (masks[t]&masks[nt]).any():continue
   if desc[q]@desc[nq]>.8 or ddesc[q]@ddesc[nq]>.8:continue
   neg.append(nt)
  for modality in ('image','prompt'):
   episodes.append({'query_region':q,'target_region':t,'target_view':vj,'modality':modality,'negative_regions':sorted(set(neg)),'split':split,'pair_iou':overlap})
  if labels[q]==labels[t] and min(margin[q],margin[t])>=.001:
   word=int(labels[q])
   positives=torch.where((views==vj)&(labels==word)&(margin>=.001))[0].tolist()
   # Text describes all matching regions, not one arbitrarily chosen instance.
   negatives=[x for x in neg if int(labels[x])!=word and margin[x]>=.001]
   episodes.append({'query_region':q,'target_region':t,'target_view':vj,'modality':'text','text_index':word,
    'positive_regions':positives,'negative_regions':negatives,'split':split,'pair_iou':overlap})
 # Registered prompt evidence: native DINO seed features plus WORLD coordinates.
 geometry=FrozenGaussianGeometry.load(a.geometry,
  expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
 geometry.assert_binding(auth['metadata']['frozen_geometry_binding'])
 lo=geometry.xyz.amin(0);extent=(geometry.xyz.amax(0)-lo).clamp_min(1e-6)
 prompts=[];seedrows=[];seedpixels=[];prompt_valid=[];prompt_rejections=[]
 for region,vi in enumerate(views.tolist()):
  validpixel=(pixel_mass[vi]>.1).reshape(h,w)
  m=masks[region].reshape(h,w);positions=[]
  for positive in (True,False):
   support=(m if positive else ~m)&validpixel
   if not support.any():break
   distance=torch.from_numpy(distance_transform_edt(support.cpu().numpy())).to(device)
   positions.append(int(distance.argmax()))
  admitted=len(positions)==2
  prompt_valid.append(admitted)
  if not admitted:
   # Preserve region indices and non-prompt teacher evidence. Sentinel rows
   # cannot enter a prompt task; admission is carried explicitly below.
   prompts.append(torch.zeros(2*dino.shape[1]+6));seedrows.append([-1,-1]);seedpixels.append([-1,-1])
   prompt_rejections.append({'region':region,'view':vi,'reason':'missing_positive_support' if not positions else 'missing_negative_support'})
   continue
  hit=hits_by_view[vi];indices=[]
  for pixel in positions:
   selected=torch.where(hit['pixel_ids']==pixel)[0]
   best=selected[hit['base_weights'][selected].argmax()]
   indices.append(int(hit['gaussian_ids'][best]))
  dense=F.interpolate(dino[vi:vi+1].float().to(device),size=(h,w),mode='bilinear',align_corners=False)[0].reshape(dino.shape[1],-1)
  coordinates=((geometry.xyz[indices]-lo)/extent).reshape(-1)
  prompts.append(torch.cat([dense[:,positions[0]],dense[:,positions[1]],coordinates]).cpu())
  seedrows.append(indices);seedpixels.append(positions)
 queries['prompt']=torch.stack(prompts).half();queries['seed_rows']=torch.tensor(seedrows);queries['seed_pixels']=torch.tensor(seedpixels)
 queries['prompt_valid']=torch.tensor(prompt_valid,dtype=torch.bool)
 episodes=[e for e in episodes if e['modality']!='prompt' or prompt_valid[e['query_region']]]
 queries['prompt_world_normalization']={'minimum':lo.cpu(),'extent':extent.cpu()}
 counts={split:{m:sum(e['split']==split and e['modality']==m for e in episodes) for m in ('text','image','prompt')} for split in ('train','validation')}
 missing=[(split,m) for split,rows in counts.items() for m,c in rows.items() if c==0]
 if any(pair!=('validation','text') or not a.allow_missing_text_validation for pair in missing):raise ValueError('incomplete three-query evidence: '+str(counts))
 payload={'schema':'radio_gs.three_query_region_episodes.v1','episodes':episodes,'queries':queries,'masks':sam['masks'],'proposal_views':sam['view_indices'],
  'region_probability':prob.cpu().half(),'seen':seen.cpu(),'teacher_manifest_sha256':sha256_file(root/'teachers.json'),
  'geometry_binding':auth['metadata']['frozen_geometry_binding'],'counts':counts,
  'label_semantics':('pairwise_geometry_appearance_and_global_scope_consistent' if a.correspondence_policy=='scope_consistent' else 'pairwise_geometry_and_native_appearance')+'_region_pseudo_labels; physical_identity_unverified; unlisted unknown',
  'correspondence_policy':a.correspondence_policy,
  'prompt_admission':{'valid_regions':sum(prompt_valid),'rejected_regions':prompt_rejections},
  'missing_evidence':missing,'allow_missing_text_validation':a.allow_missing_text_validation,
  'correspondence_consistency':consistency,
  'text_scope':'union_of_same_text_label_regions_in_target_view','negative_semantics':'cross_view_matched_disjoint_distinct_appearance_regions',
  'heldout_query_views':info['validation_frames'],'benchmark_ground_truth':False}
 torch.save(payload,root/'episodes.pt');(root/'episodes.json').write_text(json.dumps({'counts':counts,'matches':len(matches),'sha256':sha256_file(root/'episodes.pt'),
  'prompt_admission':payload['prompt_admission'],'episodes_with_explicit_negatives':sum(bool(e['negative_regions']) for e in episodes)},indent=2)+'\n')
 print(counts,flush=True)
if __name__=='__main__':main()
