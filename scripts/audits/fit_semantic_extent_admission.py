"""Source-only bounded semantic compatibility extension of extent admission."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.extent_benefit_gate import scope_semantic_features
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';source=root/'extent_utility_training_data';receipt=json.loads((source/'report.json').read_text());assert sha256_file(source/'training.pt')==receipt['sha256'];d=torch.load(source/'training.pt',weights_only=False)
parent=root/'joint_supervised_image_residual';assert sha256_file(parent/'field.pt')==d['parent_field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False);state=payload['state_dict']
epath=root/'caption_region_evidence.pt';assert sha256_file(epath)==d['inputs'][str(epath)];e=torch.load(epath,weights_only=False)
records={}
for r in d['observations']:
 v=(r['base_query_index'],r['selected_group'],r['concept_accepted'])
 if r['query_index'] in records:assert records[r['query_index']]==v
 records[r['query_index']]=v
ids,groups,accepted=zip(*(records[q] for q in d['query_ids']));accepted=torch.tensor(accepted)
semantic=scope_semantic_features(e['text_embeddings'][list(ids)].float(),state['grounded_transfer.prototypes'][list(groups)],state['grounded_transfer.routing_scope_images'][list(groups)],accepted)
x=torch.cat([d['features'],semantic],1);valid=d['validation'];train=~valid;w=d['weight'];gain=d['extent_iou']-d['native_iou']
oldpath=root/'counterfactual_extent_benefit_admission';oldreport=json.loads((oldpath/'report.json').read_text());assert sha256_file(oldpath/'field.pt')==oldreport['field_sha256'];old=torch.load(oldpath/'field.pt',weights_only=False)['state_dict'];old_allow=((d['features']-old['extent_benefit_admission.center'])/old['extent_benefit_admission.scale'])@old['extent_benefit_admission.coefficient'][:-1]+old['extent_benefit_admission.coefficient'][-1]>0
center=x[train].mean(0);scale=x[train].std(0).clamp_min(1e-4);z=torch.cat([(x-center)/scale,torch.ones(len(x),1)],1)
def summary(allow):
 out={}
 for name,mask in [('train',train),('validation',valid),('routed_validation',valid&d['routed'])]:
  def avg(v):return float((v[mask]*w[mask]).sum()/w[mask].sum())
  out[name]=dict(pairs=int(mask.sum()),allowed=int(allow[mask].sum()),utility=avg(torch.where(allow,d['extent_iou'],d['native_iou'])),native=avg(d['native_iou']),extent=avg(d['extent_iou']))
 return out
previous=summary(old_allow);assert abs(previous['validation']['utility']-oldreport['selected']['validation_iou'])<1e-5
candidates=[];coefs=[]
for penalty in [.1,1.,10.]:
 ridge=torch.eye(z.shape[1])*penalty;ridge[-1,-1]=0
 coefficient=torch.linalg.solve(z[train].T@(z[train]*w[train,None])+ridge,z[train].T@(gain[train]*w[train]));allow=z@coefficient>0;s=summary(allow)
 passed=all(s[k]['utility']>=previous[k]['utility']-1e-6 for k in ['validation','routed_validation'])
 candidates.append(dict(penalty=penalty,source_acceptance_passed=passed,metrics=s));coefs.append(coefficient)
choices=[i for i,c in enumerate(candidates) if c['source_acceptance_passed']];index=max(choices or range(len(candidates)),key=lambda i:(candidates[i]['metrics']['validation']['utility'],candidates[i]['penalty']))
arch=dict(payload['architecture'],extent_benefit_admission={'kind':'semantic_linear'});new=ElementRetrievalField(**arch);missing=new.load_state_dict(state,strict=False);assert not missing.unexpected_keys and set(missing.missing_keys)=={'extent_benefit_admission.center','extent_benefit_admission.scale','extent_benefit_admission.coefficient'};new.extent_benefit_admission.load_state_dict(dict(center=center,scale=scale,coefficient=coefs[index]))
for k,v in state.items():assert torch.equal(v,new.state_dict()[k])
out=root/'semantic_extent_benefit_admission';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=new.state_dict()),out/'field.pt');snap=out/'sources';snap.mkdir();impl={}
for f in [Path(__file__),Path('radio_gs/models/extent_benefit_gate.py'),Path('radio_gs/models/element_retrieval_field.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
r=dict(schema='semantic_extent_admission.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=d['parent_field_sha256'],training_data_sha256=receipt['sha256'],previous_gate_sha256=oldreport['field_sha256'],source_only=True,benchmark_data_used=False,all_parent_tensors_exact=True,source_acceptance_passed=bool(choices),previous=previous,candidates=candidates,selected=candidates[index],implementations=impl,validation_scope='Phrase family holdout for admission only; base field and extents share source images/masks')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,json.dumps(r['selected']),flush=True)
