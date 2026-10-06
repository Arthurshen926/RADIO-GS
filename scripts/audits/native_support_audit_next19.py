"""Source-only audit: check whether inherited observed flags drop measured evidence."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
out=Path('output/native_support_audit_next19');out.mkdir(exist_ok=False)
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 root=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field';info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];ep=torch.load(root/'episodes.pt',map_location='cpu',weights_only=False);assert sha256_file(root/'episodes.pt')==json.loads((root/'episodes.json').read_text())['sha256'];fp=root/'bounded_fallback_admission/field.pt';p=torch.load(fp,map_location='cpu',weights_only=False);obs=p['state_dict']['observed'];n=len(obs);mass=torch.zeros(n);visibility=torch.zeros(n);counts=torch.zeros(n,dtype=torch.int32)
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256'];hit=torch.load(hp,map_location='cpu',weights_only=False);pid,gid,val=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();ix=torch.where(ep['proposal_views']==vi)[0];mask=ep['masks'][ix].flatten(1).any(0);den=torch.zeros(mask.numel()).scatter_add_(0,pid,val);weights=val/den[pid].clamp_min(1e-8);view=torch.zeros(n).scatter_add_(0,gid,weights*mask[pid]);mass+=view;visibility.scatter_add_(0,gid,weights);counts+=(view>1e-6).int()
 valid=mass>1e-6;excluded=valid&~obs
 row={'scene':scene,'elements':n,'observed':int(obs.sum()),'source_measured':int(valid.sum()),'measured_but_excluded':int(excluded.sum()),'excluded_multiview':int((excluded&(counts>=2)).sum()),'unobserved_source_visible':int(((visibility>1e-6)&~obs).sum()),'observed_without_source_mass':int((obs&~valid).sum()),'field_sha256':sha256_file(fp),'source_only':True}
 torch.save({'geometry_binding':p['geometry_binding'],'observed':obs,'source_mass':mass,'source_visibility':visibility,'source_views':counts},out/(scene+'.pt'));rows.append(row);print(row,flush=True)
(out/'report.json').write_text(json.dumps({'source_only':True,'gt_used':False,'rows':rows,'implementation_sha256':sha256_file(__file__)},indent=2)+'\n')
