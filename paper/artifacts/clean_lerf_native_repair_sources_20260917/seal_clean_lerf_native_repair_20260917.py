"""Seal verified clean LERF repairs and a scoped image capability improvement."""
import json,shutil
from pathlib import Path
import torch
from scripts.audits.summarize_ropefixed_rebuilds import summarize
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');out=Path('paper/artifacts/clean_lerf_native_repair_seal_20260917.json');assert not out.exists()
summary=summarize(base)
for key in ['lerf','lerf_multiscale','lerf_view_balanced']:
 assert all(v['status']=='complete' for v in summary[key].values())
assert summary['all_precision_validations_finished'] and summary['all_clean_joint_controls_finished']
records={};files={}
def bind(p):files[str(p.resolve())]=sha256_file(p)
for scene in ['figurines','ramen']:
 root=base/'lerf_ropefix_multiscale_pipeline_20260917'/scene;field=root/'field';names=['source_precision_gate96','joint_native_retrieval_residual','joint_supervised_image_residual']
 states={}
 for name in names:
  p=field/name/'field.pt';r=json.loads(p.with_name('report.json').read_text());assert sha256_file(p)==r['field_sha256'];states[name]=torch.load(p,weights_only=False,map_location='cpu');bind(p);bind(p.with_name('report.json'))
 old,shared,scoped=[states[n] for n in names]
 assert scoped['architecture']['native_semantic_residual']['modalities']==['image']
 assert scoped['architecture']['native_semantic_residual']['relative_bound']==.25
 assert old['geometry_binding']==scoped['geometry_binding']==shared['geometry_binding']
 assert old['query_encoder']==scoped['query_encoder']==shared['query_encoder']
 for key,value in old['state_dict'].items():assert torch.equal(value,scoped['state_dict'][key]),key
 for key,value in shared['state_dict'].items():assert torch.equal(value,scoped['state_dict'][key]),key
 old_text=json.loads((root/'precision_gate_validation/text.json').read_text())['runs'][names[0]]
 new_text=json.loads((root/'scoped_joint_image_validation/text.json').read_text())['runs'][names[2]]
 assert {k:v for k,v in old_text.items() if k!='field_sha256'}=={k:v for k,v in new_text.items() if k!='field_sha256'}
 old_ref=json.loads((root/'precision_gate_validation/reference.json').read_text())['runs'][names[0]]
 new_ref=json.loads((root/'scoped_joint_image_validation/reference.json').read_text())['runs'][names[2]]
 shared_ref=json.loads((root/'clean_joint_validation/reference.json').read_text())['runs'][names[1]]
 assert len(old_ref['observations'])==len(new_ref['observations'])==len(shared_ref['observations'])
 for before,after,joint in zip(old_ref['observations'],new_ref['observations'],shared_ref['observations']):
  assert {k:v for k,v in before.items() if k!='image'}=={k:v for k,v in after.items() if k!='image'}
  assert after['image']==joint['image']
 for task in ['text','reference','cold']:
  bind(root/'scoped_joint_image_validation'/(task+'.json'))
 records[scene]=dict(full_text_iou=new_text['observation_miou'],full_text_ap=new_text['average_precision_macro'],before_reference=old_ref['macro'],after_reference=new_ref['macro'],original_text_prompt_observations_exact=True,shared_residual_image_observations_exact=True,all_parent_state_exact=True,added_global_parameters=sum(v.numel() for k,v in scoped['state_dict'].items() if k.startswith('native_residual_decoder.')))
 for tag in ['clean_source_semantic_stages','clean_membership_gradients','multiscale_ropefixed_native_compression','viewbalanced_ropefixed_native_compression']:
  bind(Path('paper/artifacts')/(scene+'_'+tag+'_20260917.json'))
 del states,old,shared,scoped
sources=Path('paper/artifacts/clean_lerf_native_repair_sources_20260917');sources.mkdir(exist_ok=False)
for name in ['scripts/audits/seal_clean_lerf_native_repair_20260917.py','scripts/audits/diagnose_source_semantic_stages.py','scripts/audits/diagnose_current_native_compression.py','scripts/audits/audit_protected_membership_gradients.py','scripts/audits/calibrate_native_gate_precision.py','scripts/audits/reuse_verified_source_gate_calibration.py','scripts/audits/train_joint_native_retrieval_residual.py','scripts/audits/export_image_supervised_native_residual.py','scripts/audits/summarize_ropefixed_rebuilds.py']:
 src=Path(name);dest=sources/src.name;shutil.copy2(src,dest);bind(dest)
for name in ['lerf_view_balanced_complete_comparison_20260917.json','sam3_complex_rope_root_cause_20260917.json','lerf_text_grounding_retention_ropefix_20260917.json']:bind(Path('paper/artifacts')/name)
result=dict(schema='clean_lerf_native_repair_seal.v1',independent_final_test=False,complete_joint_benchmark=False,decision='retain scoped image capability candidate on corrected multiscale source inputs; reject shared text/image residual and shared view-balanced replacement',records=records,verified_rebuilds={k:summary[k] for k in ['lerf','lerf_multiscale','lerf_view_balanced']},files=files,limitations=['source masks are pseudo labels, physical correspondence/granularity not certified','image-only application selected after development comparison','immutable native anchor does not guarantee every corrected image-query rank','matched compression control old source snapshot must be retained separately; no claim to capture every external runtime dependency'])
out.write_text(json.dumps(result,indent=2)+'\n');print(records,flush=True)
