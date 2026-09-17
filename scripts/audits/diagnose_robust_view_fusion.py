"""Compare matched 64D linear/robust fusion on identical source-view queries."""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.robust_source_fusion import robust_view_fusion
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root.parent/'robust_fusion_diagnosis';out.mkdir(exist_ok=False);snap=out/'sources';snap.mkdir();impl={}
for f in [Path(__file__),Path('radio_gs/training/robust_source_fusion.py'),Path('radio_gs/training/source_pixel_evidence.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];inputs={}
def load(name,expected):
 path=root/name;assert sha256_file(path)==expected;inputs[str(path)]=expected;return torch.load(path,map_location='cpu',weights_only=False)
ep=load('episodes.pt',json.loads((root/'episodes.json').read_text())['sha256']);assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json');tr=info['teacher_files']['siglip'];teacher=load(tr['path'],tr['sha256']);cap=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert cap['teacher_manifest_sha256']==ep['teacher_manifest_sha256'] and cap['geometry_binding']==ep['geometry_binding']
parent=root/'pixel_evidence96';pr=json.loads((parent/'report.json').read_text());field=load('pixel_evidence96/field.pt',pr['field_sha256']);sd=field['state_dict'];basis=sd['native_decoder.weight'].to(a.device);observed=sd['observed'].to(a.device);n=len(observed);q=F.normalize(cap['text_embeddings'].float().to(a.device),dim=-1)@basis;desc=F.normalize(teacher['descriptors'].float().to(a.device),dim=-1)@basis;h,w=ep['masks'].shape[-2:];pixels=h*w
vectors=[];masses=[];maps=[];trainviews=[info['frames'].index(f) for f in info['train_frames']]
with torch.no_grad():
 for vi in trainviews:
  rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);pid=hit['pixel_ids'].long().to(a.device);gid=hit['gaussian_ids'].long().to(a.device);weight=hit['base_weights'].float().to(a.device);ids=torch.where(ep['proposal_views']==vi)[0]
  weights,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1).to(a.device),pid,gid,weight,n,partition_overlap=True);mass=weights.sum(0);vectors.append(weights.T@desc[ids]/mass[:,None].clamp_min(1e-8));masses.append(mass)
  pm=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,weight);maps.append(torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/pm[pid].clamp_min(1e-8),(pixels,n)).coalesce())
 vectors=torch.stack(vectors);masses=torch.stack(masses)
 def fuse(excluded=None,iterations=5):
  parts=[];valid=[]
  for first in range(0,n,8192):
   z,v=robust_view_fusion(vectors[:,first:first+8192],masses[:,first:first+8192],excluded_view=excluded,iterations=iterations);parts.append(z);valid.append(v)
  return torch.cat(parts),torch.cat(valid)
 linear,valid=fuse(iterations=0);robust,_=fuse();torch.testing.assert_close(linear[observed],sd['latent'].to(a.device)[observed,:64],atol=2e-5,rtol=2e-4)
 torch.save(dict(latent=robust.cpu(),valid=(valid&observed).cpu(),basis_sha256=sha256_file(parent/'field.pt'),inputs=inputs,implementations=impl,source_only=True,benchmark_data_used=False),out/'anchor.pt')
 allscores={'linear':F.normalize(linear,dim=-1)@q.T,'robust':F.normalize(robust,dim=-1)@q.T};rows=[]
 for k,vi in enumerate(trainviews):
  mat=maps[k];lvo,support=fuse(k,0);rvo,_=fuse(k,5);scores={};common=None
  for name,cos,seen in [('linear',allscores['linear'],valid),('robust',allscores['robust'],valid),('linear_lvo',F.normalize(lvo,dim=-1)@q.T,support),('robust_lvo',F.normalize(rvo,dim=-1)@q.T,support)]:
   seen=seen&observed;coverage=torch.sparse.mm(mat,seen.float()[:,None])[:,0];good=coverage>.5;response=torch.sparse.mm(mat,((cos+1)/2)*seen[:,None])/coverage[:,None].clamp_min(1e-8);scores[name]=torch.where(good[:,None],response,0);common=good if common is None else common&good
  targets={}
  for ri in torch.where(cap['proposal_views']==vi)[0].tolist():
   r=cap['records'][ri];mask=cap['masks'][ri].flatten().bool().to(a.device)
   if r['confidence']<.7 or int(mask.sum())<20:continue
   for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(mask))|mask
  for qi,target in targets.items():rows.append(dict(frame=info['frames'][vi],phrase=cap['phrases'][qi],query_index=qi,common_positive_support=float(common[target].float().mean()),**{name:binary_ranking_metrics(value[:,qi],target)['average_precision'] for name,value in scores.items()}))
  print(a.scene,info['frames'][vi],len(rows),flush=True)
macro={name:sum(r[name] for r in rows)/len(rows) for name in ['linear','robust','linear_lvo','robust_lvo']};result=dict(schema='robust_source_view_fusion.v1',macro=macro,rows=rows,inputs=inputs,implementations=impl,anchor_sha256=sha256_file(out/'anchor.pt'),source_only=True,teacher_masks_are_not_gt=True,benchmark_data_used=False,fixed_iterations=5,epsilon=1e-4,matched_basis_64=True,original_anchor_reproduced=True)
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)
