"""Diagnose added-support mixing; ablated render is not a deployed method."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.retrieval_supervision import episode_membership_targets
from radio_gs.training.retrieval_metrics import partial_retrieval_metrics,summarize_partial_episodes
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);device='cuda:2';root=a.root
 receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256']
 d=torch.load(root/'episodes.pt',weights_only=False);auth=json.loads((root/'responsibility.json').read_text());identity=json.loads((root/'native_query_encoder.json').read_text())
 assert sha256_file(root/'teachers.json')==d['teacher_manifest_sha256']
 assert sha256_file(root/'responsibility.json')==json.loads((root/'teachers.json').read_text())['authority_sha256']
 geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
 def load(run):
  report=json.loads((run/'report.json').read_text());field,_=load_element_retrieval_field(run/'field.pt',expected_sha256=report['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=device);return field,report['field_sha256']
 field,digest=load(a.run);ref,refdigest=load(a.reference_run);old=field.observed&ref.observed;added=field.observed&~ref.observed
 h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];mats=[]
 for record in auth['views']:
  path=root/record['relative_path'];assert sha256_file(path)==record['sha256'];hit=torch.load(path,map_location=device,weights_only=False)
  pid,gid,weight=hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float();mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight)
  mats.append(torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(old))).coalesce())
 rows=[]
 with torch.no_grad():
  for e in d['episodes']:
   if e['split']!='validation':continue
   m=e['modality'];qi=e.get('text_index',e['query_region']);scores=field.retrieve(d['queries'][m][qi:qi+1].float().to(device),m)['scores'][:,0];mat=mats[e['target_view']]
   oldmass=torch.sparse.mm(mat,old.float()[:,None])[:,0];newmass=torch.sparse.mm(mat,added.float()[:,None])[:,0];total=oldmass+newmass
   oldsum=torch.sparse.mm(mat,(scores*old)[:,None])[:,0];newsum=torch.sparse.mm(mat,(scores*added)[:,None])[:,0]
   full=(oldsum+newsum)/total.clamp_min(1e-8);restricted=oldsum/oldmass.clamp_min(1e-8)
   target,known,_,_=episode_membership_targets(e,d['masks'].flatten(1).to(device),d['region_probability'].to(device),old,oldmass)
   rows.append({'modality':m,'query_region':e['query_region'],'full':partial_retrieval_metrics(full,target,known,oldmass>.5),'old_support_only_diagnostic':partial_retrieval_metrics(restricted,target,known,oldmass>.5),'mean_added_weight_on_known':float((newmass/total.clamp_min(1e-8))[known].mean()),'mean_response_change_on_known':float((full-restricted)[known].mean())})
 macro={m:{key:summarize_partial_episodes([r[key] for r in rows if r['modality']==m]) for key in ['full','old_support_only_diagnostic']} for m in ['text','image','prompt']}
 result={'schema':'added_support_response_diagnostic.v1','field_sha256':digest,'reference_field_sha256':refdigest,'episode_sha256':receipt['sha256'],'not_deployment_method':True,'macro':macro,'episodes':rows}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print({m:{k:v['known_iou_all_episodes_zero_for_undefined'] for k,v in x.items()} for m,x in macro.items()})
if __name__=='__main__':main()
