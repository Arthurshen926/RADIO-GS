"""Source-only global concept-bias calibration in the deployed clamp/render order."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';out=root/'source_scope_precision_calibration';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py');inputs={}
def load(name,digest):
 path=root/name;assert sha256_file(path)==digest;inputs[str(path)]=digest;return torch.load(path,weights_only=False)
info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json');pr=json.loads((root/'joint_supervised_image_residual/report.json').read_text());payload=load('joint_supervised_image_residual/field.pt',pr['field_sha256']);assert e['geometry_binding']==payload['geometry_binding']==auth['metadata']['frozen_geometry_binding'];field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups'];n=len(field.latent);P=e['masks'][0].numel();head=field.grounded_transfer;offset=head.identity_selection['trusted_concept_start'];heldout=set(info['train_frames'][::5]);biases=[0.,.05,.1,.2,.3];rows=[]
with torch.no_grad():
 z=head.features(field.latent);raw=(head.decoder(z)+head.residual(z))[:,offset:];assert raw.shape[1]==len(groups)
 for frame in info['train_frames']:
  vi=info['frames'].index(frame);r=auth['views'][vi];hit=load(r['relative_path'],r['sha256']);gid=hit['gaussian_ids'].long().to(a.device);pid=hit['pixel_ids'].long().to(a.device);w=hit['base_weights'].float().to(a.device);pm=torch.zeros(P,device=a.device).scatter_add_(0,pid,w);mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/pm[pid].clamp_min(1e-8),(P,n)).coalesce();coverage=torch.sparse.mm(mat,field.observed.float()[:,None]);valid=coverage>.5;ids=[j for j,g in enumerate(groups) if vi in g['views']];target=torch.stack([e['masks'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0).flatten() for j in ids],1).to(a.device)
  for bias in biases:
   probability=(raw[:,ids]-bias).clamp(1e-6,1-1e-6);score=torch.sparse.mm(mat,probability*field.observed[:,None])/coverage.clamp_min(1e-8);pred=(score>.5)&valid;inter=(pred&target).sum(0);iou=inter/(pred|target).sum(0).clamp_min(1);precision=inter/pred.sum(0).clamp_min(1);recall=inter/target.sum(0).clamp_min(1)
   for ci,j in enumerate(ids):rows.append(dict(frame=frame,group=j,bias=bias,validation=frame in heldout,iou=float(iou[ci]),precision=float(precision[ci]),recall=float(recall[ci])))
  print(a.scene,frame,flush=True)
 candidates=[]
 for bias in biases:
  rec=dict(bias=bias)
  for name,valid in [('train',False),('validation',True)]:
   rr=[r for r in rows if r['bias']==bias and r['validation']==valid];rec[name]={k:sum(r[k] for r in rr)/len(rr) for k in ['iou','precision','recall']};rec[name]['observations']=len(rr)
  candidates.append(rec)
 selected=max(candidates,key=lambda r:(r['train']['iou'],-r['bias']));passed=selected['validation']['iou']>=candidates[0]['validation']['iou']-1e-6
 head.decoder.bias[offset:]-=selected['bias'];state={k:v.cpu() for k,v in field.state_dict().items()}
 for k,v in payload['state_dict'].items():
  if k=='grounded_transfer.decoder.bias':assert torch.equal(v[:offset],state[k][:offset])
  else:assert torch.equal(v,state[k]),k
 torch.save(dict(payload,state_dict=state),out/'field.pt');report=dict(schema='source_scope_precision_calibration.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],source_only=True,benchmark_data_used=False,source_acceptance_passed=passed,selected=selected,candidates=candidates,heldout_frames=sorted(heldout),validation_scope='Frames held out from global bias fitting only; all extent/base parameters already saw their source masks/images.',selection='Maximum training observation IoU, smallest bias on ties; validation is acceptance only',scope='Only concept decoder output bias; native/geometry/independent region tensors exact; old extent benefit gate absent',inputs=inputs,implementation_sha256=sha256_file(__file__),rows=rows);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,selected,passed,flush=True)
