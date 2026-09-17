"""One fixed depth-two utility policy, with an unchanged source phrase split."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.extent_benefit_gate import ExtentUtilityTree
from radio_gs.training.extent_utility_tree import fit_extent_utility_tree
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
root=Path('output/pixel_evidence_rebuild_20260917')/a.scene/'field';source=root/'extent_utility_training_data';receipt=json.loads((source/'report.json').read_text());assert sha256_file(source/'training.pt')==receipt['sha256'];d=torch.load(source/'training.pt',weights_only=False)
parent=root/'joint_supervised_image_residual';assert sha256_file(parent/'field.pt')==d['parent_field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
x=d['features'];gain=d['extent_iou']-d['native_iou'];valid=d['validation'];train=~valid;w=d['weight'];groups=torch.tensor(d['base_query_ids'])
policy=fit_extent_utility_tree(x[train],gain[train],w[train],groups[train]);tree=ExtentUtilityTree();tree.load_state_dict(policy);allow=tree.decide_features(x)
oldreport=json.loads((root/'counterfactual_extent_benefit_admission/report.json').read_text());assert sha256_file(root/'counterfactual_extent_benefit_admission/field.pt')==oldreport['field_sha256'];old=torch.load(root/'counterfactual_extent_benefit_admission/field.pt',weights_only=False)['state_dict'];z=(x-old['extent_benefit_admission.center'])/old['extent_benefit_admission.scale'];coef=old['extent_benefit_admission.coefficient'];old_allow=z@coef[:-1]+coef[-1]>0
summaries={}
for name,mask in [('train',train),('validation',valid),('routed_validation',valid&d['routed'])]:
 def avg(v):return float((v[mask]*w[mask]).sum()/w[mask].sum())
 summaries[name]=dict(pairs=int(mask.sum()),native=avg(d['native_iou']),extent=avg(d['extent_iou']),previous_gate=avg(torch.where(old_allow,d['extent_iou'],d['native_iou'])),tree=avg(torch.where(allow,d['extent_iou'],d['native_iou'])),allowed=int(allow[mask].sum()))
assert abs(summaries['validation']['previous_gate']-oldreport['selected']['validation_iou'])<1e-5
passed=all(summaries[k]['tree']>=summaries[k]['previous_gate']-1e-6 for k in ['validation','routed_validation'])
arch=dict(payload['architecture'],extent_benefit_admission={'kind':'utility_tree'});new=ElementRetrievalField(**arch);missing=new.load_state_dict(payload['state_dict'],strict=False);assert not missing.unexpected_keys and set(missing.missing_keys)=={'extent_benefit_admission.feature','extent_benefit_admission.cutoff','extent_benefit_admission.decision'};new.extent_benefit_admission.load_state_dict(policy)
for k,v in payload['state_dict'].items():assert torch.equal(v,new.state_dict()[k])
out=root/'source_extent_utility_tree';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=new.state_dict()),out/'field.pt');snap=out/'sources';snap.mkdir();impl={}
for f in [Path(__file__),Path('radio_gs/training/extent_utility_tree.py'),Path('radio_gs/models/extent_benefit_gate.py'),Path('radio_gs/models/element_retrieval_field.py')]:shutil.copy2(f,snap/f.name);impl[str(f)]=sha256_file(f)
r=dict(schema='source_extent_utility_tree.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=d['parent_field_sha256'],training_data_sha256=receipt['sha256'],previous_gate_sha256=oldreport['field_sha256'],source_only=True,benchmark_data_used=False,all_parent_tensors_exact=True,max_depth=2,minimum_query_groups_per_leaf=5,source_acceptance_passed=passed,source_metrics=summaries,policy={k:v.tolist() for k,v in policy.items()},implementations=impl,validation_scope='Identical phrase-family holdout as counterfactual gate, not independent from base field/extent source supervision')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,passed,summaries,flush=True)
