"""Seal source conflict attribution, source-chosen correction and complete controls."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
result=dict(schema='scope_conflict_repair.v1',promoted=False,retained='counterfactual_extent_benefit_admission',scenes={})
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene;field=root/'field';artifacts=[];diagnostics={}
 for name in ['scope_conflict_audit','scope_core_conflict_audit','scope_core_conflict_radius1_audit']:
  path=root/name/'report.json';d=json.loads(path.read_text());assert d['source_only'] and not d['benchmark_data_used'];assert abs(sum(d['fp_decoded_mass_fraction'].values())-1)<1e-5;assert sha256_file(root/name/'source.py')==d['implementation_sha256']
  for p,h in d['inputs'].items():assert sha256_file(p)==h
  diagnostics[name]=d['fp_decoded_mass_fraction'];artifacts.append(path)
 for name in ['scope_depth_correspondence.json','scope_depth_scope_pairs.json']:
  dp=root/name;dd=json.loads(dp.read_text());assert dd['source_only'] and not dd['benchmark_data_used'] and not dd['unavailable_calibration_frames']
  for p,h in dd['inputs'].items():assert sha256_file(p)==h
  artifacts.append(dp)
 parent=torch.load(field/'joint_supervised_image_residual/field.pt',weights_only=False);calroot=field/'source_scope_precision_calibration';report=json.loads((calroot/'report.json').read_text());assert sha256_file(calroot/'field.pt')==report['field_sha256'];assert report['parent_field_sha256']==sha256_file(field/'joint_supervised_image_residual/field.pt');assert report['source_acceptance_passed'];assert report['selected']==max(report['candidates'],key=lambda r:(r['train']['iou'],-r['bias']));new=torch.load(calroot/'field.pt',weights_only=False);offset=parent['architecture']['grounded_transfer']['identity_selection']['trusted_concept_start']
 for k,v in parent['state_dict'].items():
  expected=v.clone()
  if k=='grounded_transfer.decoder.bias':expected[offset:]-=report['selected']['bias']
  torch.testing.assert_close(expected,new['state_dict'][k],rtol=0,atol=0)
 gatepath=field/'scope_precision_benefit_admission';gr=json.loads((gatepath/'report.json').read_text());assert gr['parent_field_sha256']==report['field_sha256'] and gr['source_routed_validation_passed'];assert sha256_file(gatepath/'field.pt')==gr['field_sha256'];gate=torch.load(gatepath/'field.pt',weights_only=False)
 for k,v in new['state_dict'].items():assert torch.equal(v,gate['state_dict'][k])
 metrics={};protocol=None
 for name in ['source_scope_precision_calibration','scope_precision_benefit_admission','counterfactual_extent_benefit_admission']:
  val=root/(name+'_validation');execution=json.loads((val/'execution.json').read_text());assert execution['finished'] and execution['image_prompt_observations_exact'];doc=json.loads((val/'text.json').read_text());actual={k:v for k,v in doc['protocol'].items() if k!='fields'}
  if protocol is None:protocol=actual
  else:assert protocol==actual
  for st in execution['stages']:assert st['status']=='complete' and st['returncode']==0 and sha256_file(val/(st['name']+'.json'))==st['output_sha256']
  metrics[name]={k:doc['runs'][name][k] for k in ['observation_miou','category_macro_miou','average_precision_macro']};artifacts.extend([val/'execution.json',val/'text.json',val/'reference.json'])
 coldpath=root/'scope_precision_benefit_admission_validation/cold.json';cold=json.loads(coldpath.read_text());assert cold['field_sha256']==gr['field_sha256'] and cold['training_cache_reads_forbidden'] and cold['native_queries_encoded_online'];assert cold['modalities']['text']['concept_route_consistent'] and cold['modalities']['text']['posterior_max_difference']<.01 and cold['modalities']['prompt']['posterior_max_difference']==0
 oldcold=json.loads((root/'counterfactual_extent_benefit_admission_validation/cold.json').read_text());assert cold['modalities']['image']==oldcold['modalities']['image'];artifacts.extend([calroot/'report.json',gatepath/'report.json',coldpath])
 result['scenes'][scene]=dict(source_bias=report['selected']['bias'],source_selection_passed=True,native_geometry_and_independent_region_tensors_exact=True,image_prompt_observations_exact=True,gate_refit_on_changed_scopes=True,diagnostics=diagnostics,metrics=metrics,cold=cold,artifacts={str(p):sha256_file(p) for p in artifacts})
result['additional_artifacts']={str(p):sha256_file(p) for p in [Path('paper/artifacts/scope_boundary_sensitivity_20260917.json'),Path('paper/artifacts/scope_resolution_audit_20260917.json')]}
files=['scripts/audits/diagnose_scope_conflicts.py','scripts/audits/diagnose_scope_core_conflicts.py','scripts/audits/calibrate_scope_false_positives.py','scripts/audits/fit_counterfactual_extent_admission.py','scripts/audits/audit_scope_resolution.py','tests/test_scope_precision_contract.py','scripts/audits/diagnose_scope_depth_correspondence.py','scripts/audits/diagnose_depth_consistent_scope_pairs.py',__file__];result['current_implementations']={p:sha256_file(p) for p in files};result['acceptance']='Ramen improves but figurines IoU regresses; reject shared replacement and do not route by benchmark scene identity.'
Path('paper/artifacts/scope_conflict_repair_20260917.json').write_text(json.dumps(result,indent=2)+'\n');print('verified conflict diagnostics and complete calibration controls; previous candidate retained')
