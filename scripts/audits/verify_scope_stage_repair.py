"""Seal source-stage localization and rejected matched final-layer controls."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
result=dict(schema='scope_stage_repair.v1',benchmark_data_used=False,final_retrieval_gain_claimed=False,retained='counterfactual_extent_benefit_admission',scenes={})
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene;field=root/'field';stages=root/'scope_transfer_weighted_stages/report.json';d=json.loads(stages.read_text());assert d['source_only'] and not d['benchmark_data_used']
 for p,h in d['inputs'].items():assert sha256_file(p)==h
 assert sha256_file(root/'scope_transfer_weighted_stages/source.py')==d['implementation_sha256']
 e=torch.load(field/'caption_region_evidence.pt',weights_only=False);auth=json.loads((field/'responsibility.json').read_text());assert e['geometry_binding']==auth['metadata']['frozen_geometry_binding']
 parent=torch.load(field/'joint_supervised_image_residual/field.pt',weights_only=False);controls={};artifacts=[stages]
 for name in ['weighted_scope_decoder','equal_scope_decoder']:
  path=field/name;report=json.loads((path/'report.json').read_text());assert sha256_file(path/'field.pt')==report['field_sha256'];assert sha256_file(field/'joint_supervised_image_residual/field.pt')==report['parent_field_sha256'];assert not report['source_acceptance_passed'];new=torch.load(path/'field.pt',weights_only=False)
  offset=parent['architecture']['grounded_transfer']['identity_selection']['trusted_concept_start'];changed={'grounded_transfer.decoder.weight','grounded_transfer.decoder.bias','grounded_transfer.residual.2.weight','grounded_transfer.residual.2.bias'}
  for k,v in parent['state_dict'].items():assert torch.equal(v[:offset],new['state_dict'][k][:offset]) if k in changed else torch.equal(v,new['state_dict'][k]),k
  for p,h in report['inputs'].items():assert sha256_file(p)==h
  controls[name]=report['validation'];artifacts.append(path/'report.json')
 assert controls['weighted_scope_decoder']['refit_heldout']<controls['equal_scope_decoder']['refit_heldout']
 union=field/'union_grounded_semantic_transfer96';ur=json.loads((union/'report.json').read_text());assert sha256_file(union/'field.pt')==ur['field_sha256'];assert ur['group_lifting']=='pixel_union_before_projection';assert ur['union_implementation_sha256']==sha256_file('radio_gs/training/scope_union_lifting.py');artifacts.append(union/'report.json')
 retained=field/'counterfactual_extent_benefit_admission';rr=json.loads((retained/'report.json').read_text());assert sha256_file(retained/'field.pt')==rr['field_sha256']
 result['scenes'][scene]=dict(observations=len(d['rows']),accepted_observations=sum(r['concept_accepted'] for r in d['rows']),stages=d['macro'],decoder_controls=controls,only_concept_final_layer_changed=True,retained_sha256=rr['field_sha256'],union_initialization_executed=True,artifacts={str(p):sha256_file(p) for p in artifacts})
files=['radio_gs/training/scope_union_lifting.py','scripts/audits/fit_grounded_semantic_transfer.py','scripts/audits/train_rendered_extent.py','scripts/audits/diagnose_scope_transfer_stages.py','scripts/audits/diagnose_scope_weighted_stages.py','scripts/audits/fit_weighted_scope_decoder.py','tests/test_scope_union_lifting.py',__file__];result['current_implementations']={p:sha256_file(p) for p in files}
Path('paper/artifacts/scope_stage_repair_20260917.json').write_text(json.dumps(result,indent=2)+'\n');print('verified source stages, matched rejected controls, repaired union initialization, unchanged retained fields')
