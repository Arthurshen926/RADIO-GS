"""Source-only audit of grounding retention and native score compatibility."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
if a.output.exists():raise FileExistsError(a.output)
root=a.root;g=torch.load(root/'grounded_text_supervision.pt',weights_only=False);d=torch.load(root/'episodes.pt',weights_only=False);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'episodes.pt')==g['episode_sha256'];assert sha256_file(root/'teachers.json')==g['teacher_manifest_sha256'];assert sha256_file(root/'responsibility.json')==info['authority_sha256'];assert sha256_file(g['grounding_path'])==g['grounding_sha256'];raw=torch.load(g['grounding_path'],weights_only=False);assert raw['frames']==info['train_frames']
r=json.loads((root/a.run/'report.json').read_text());assert sha256_file(root/a.run/'field.pt')==r['field_sha256'];payload=torch.load(root/a.run/'field.pt',weights_only=False);assert payload['geometry_binding']==g['geometry_binding'];f=ElementRetrievalField(**payload['architecture']);f.load_state_dict(payload['state_dict']);f.to(a.device).eval().requires_grad_(False)
with torch.no_grad():
 score=F.normalize(f.native_decoder(f.latent),dim=-1)@F.normalize(d['queries']['text'].float().to(a.device),dim=-1).T
 records=[]
 for i,frame in enumerate(info['train_frames']):
  vi=info['frames'].index(frame);rec=auth['views'][vi];path=root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();mass=torch.zeros(auth['num_pixels'],device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(auth['num_pixels'],len(f.latent))).coalesce();support=torch.sparse.mm(mat,f.observed.float()[:,None])[:,0];render=torch.sparse.mm(mat,score*f.observed[:,None])/support[:,None].clamp_min(1e-8)
  for qi,q in enumerate(g['queries']):
   rawpos=(raw['probability'][i,qi].flatten().to(a.device)>.8)&(support>.5)
   if not rawpos.any():continue
   matches=[(j,e) for j,e in enumerate(g['tasks']) if e['target_view']==vi and e['text_index']==qi];row={'frame':frame,'query':q,'raw_positive_pixels':int(rawpos.sum()),'compiled_task':bool(matches),'corroborated_positive_elements':int(g['positive_elements'][qi].sum())}
   if matches:
    j,e=matches[0];pos=g['positive_pixels'][j].to(a.device)&(support>.5);neg=g['negative_pixels'][j].to(a.device)&(support>.5);known=pos|neg;x=(render[:,qi]+1)/2;row.update(positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),positive_retention=float(pos.sum()/rawpos.sum()),positive_mean_cosine=float(render[pos,qi].mean()),negative_mean_cosine=float(render[neg,qi].mean()),known_ranking=binary_ranking_metrics(x[known],pos[known]))
   records.append(row)
result={'schema':'source_grounding_native_compatibility.v1','source_only':True,'benchmark_labels_used':False,'field_sha256':r['field_sha256'],'grounded_sha256':sha256_file(root/'grounded_text_supervision.pt'),'implementation_sha256':sha256_file(__file__),'scope':'raw SAM3 positives are teacher evidence, not ground truth; retained-mask and known-domain ranking only','records':records};a.output.write_text(json.dumps(result,indent=2)+'\n');print('wrote',a.output)
