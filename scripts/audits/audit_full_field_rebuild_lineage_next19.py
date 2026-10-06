"""CPU lineage proof for each genuinely rebuilt complete semantic field."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--base-root',type=Path,default=Path('output/full_field_rebuild_next19'));a=p.parse_args();torch.set_num_threads(4);root=a.base_root/a.scene;out=root/'lineage_audit.json';assert not out.exists();execution=json.loads((root/'execution.json').read_text());assert execution['complete'];r=root/'reconstruction/field';b=root/'base/field';old=Path('output/depth_opacity_pipeline_20260918')/a.scene;expected=json.loads(Path('output/full_field_rebuild_audit_next19/report.json').read_text())['scenes'][a.scene];info=json.loads((r/'teachers.json').read_text());assert info['train_frames']==expected['eligible_expanded_train'];assert info['validation_frames']==expected['old_validation'];assert sha256_file(r/'teachers.json')==sha256_file(b/'teachers.json');modeldirs=list(b.glob('*/field.pt'))+list(r.glob('*/field.pt'));new_model_hashes={};reports={}
for path in modeldirs:
 report=json.loads(path.with_name('report.json').read_text());digest=sha256_file(path);assert digest==report['field_sha256'],str(path);new_model_hashes[str(path)]=digest;reports[str(path.parent)]=report
known=set(new_model_hashes.values());known.update(sha256_file(Path(path)/'report.json') for path in reports)
sourcefiles=['teachers.json','episodes.pt','episodes.json','caption_region_evidence.pt','caption_region_evidence.json','source_grounded_text.pt','source_grounded_text.json','grounded_text_supervision.pt','grounded_text_supervision.json','source_fusion_initialization.pt','source_fusion_initialization.json','source_phrase_groups.json','responsibility.json','source_frames.json']
for folder in [b,r,root/'base/raw_field']:
 for f in sourcefiles:
  if (folder/f).exists():known.add(sha256_file(folder/f))
checks={};unresolved=[]
for path,report in reports.items():
 checks[path]={}
 for key in ['parent_field_sha256','retained_control_sha256','episode_sha256','grounded_text_supervision_sha256','raw_source_grounding_sha256','evidence_sha256','authority_sha256','calibration_parent_report_sha256','source_training_report_sha256']:
  if key in report and report[key] is not None:
   ok=report[key] in known;checks[path][key]=dict(digest=report[key],bound_to_new_root=ok)
   if not ok:unresolved.append(dict(path=path,key=key,digest=report[key]))
assert not unresolved,unresolved
load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
new=load(r/'bounded_fallback_admission/field.pt');previous=load(old/'reconstruction/field/bounded_fallback_admission/field.pt');base=load(b/'ropefix96_protected/field.pt');assert new['geometry_binding']==previous['geometry_binding']==base['geometry_binding'];sd,osd,bsd=new['state_dict'],previous['state_dict'],base['state_dict'];assert sd['latent'].shape==osd['latent'].shape;native_changed=not torch.equal(sd['latent'][:,:64],osd['latent'][:,:64]);structure_changed=not torch.equal(sd['latent'][:,64:],osd['latent'][:,64:]);basis_changed=not torch.equal(sd['native_decoder.weight'],osd['native_decoder.weight']);assert native_changed and structure_changed and basis_changed,'Expansion did not rebuild all field parts';assert torch.equal(sd['latent'][:,64:],bsd['latent'][:,64:]),'Unexpected suffix change after base train';br=json.loads((b/'ropefix96_protected/report.json').read_text());assert br['protect_native_semantic'] and br['freeze_native_decoder'];assert br['source_regions_supervised']==br['source_regions_available'];assert br['training_coverage']['excluded_untrained_visible_elements']==0
observed=sd['observed'];before=osd['observed'];evaluation_path=root/'evaluation/report.json';evaluation=json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
if evaluation is not None:assert evaluation['field_sha256']==sha256_file(r/'bounded_fallback_admission/field.pt');assert evaluation['official_four_query'] is False
episode_counts={}
for name,folder in [('initial_raw',root/'base/raw_field'),('augmented',b),('reconstruction',r)]:
 receipt=json.loads((folder/'episodes.json').read_text());assert sha256_file(folder/'episodes.pt')==receipt['sha256'];episode_counts[name]=dict(counts=receipt['counts'],episode_sha256=receipt['sha256'],receipt_sha256=sha256_file(folder/'episodes.json'))
base_execution=json.loads((root/'base/execution.json').read_text());training_stages=[stage for stage in base_execution['stages'] if stage.get('name')=='field' or 'train_three_query_field' in ' '.join(stage.get('command',[]))];assert training_stages,'No field training execution receipt';assert any(stage.get('status')=='complete' for stage in training_stages),'No completed training stage'
runtime_execution_overrides=[]
for execution_path in [root/'base/execution.json',root/'reconstruction/execution.json']:
 receipt=json.loads(execution_path.read_text())
 for stage in receipt['stages']:
  if 'runtime_override' not in stage:continue
  override=stage['runtime_override'];assert stage['status']=='complete'
  if 'implementation_sha256' in override:
   assert sha256_file(override['command'][1])==override['implementation_sha256']
  for source,digest in stage.get('implementations',{}).items():assert sha256_file(source)==digest
  runtime_execution_overrides.append(dict(execution=str(execution_path),execution_sha256=sha256_file(execution_path),stage=stage['name'],original_command=stage['command'],original_implementations=stage.get('implementations',{}),override=override))
recovery_receipts={}
for name in ['pixel_memory_recovery.json','grounded_cpu_verification.json','recovery_v5_adopted.json','receipt_metadata_correction/correction.json']:
 if (root/name).exists():recovery_receipts[name]=dict(path=str(root/name),sha256=sha256_file(root/name))
result=dict(runtime_execution_overrides=runtime_execution_overrides,recovery_receipts=recovery_receipts,base_training_execution=training_stages,base_execution_sha256=sha256_file(root/'base/execution.json'),episode_populations=episode_counts,episode_validation_contract='Original2 teacher-validation frames, cross-view matched pseudo regions; nonzero counts are not comprehensive generalization validation. Distinct from last4 training-view scope fit validation.',scene=a.scene,complete=True,semantic_reconstruction_not_geometry_retraining=True,new_old_geometry_binding_equal=True,train_frames=info['train_frames'],validation_frames=info['validation_frames'],source_budget_old=expected['train_count_before'],source_budget_new=len(info['train_frames']),base_steps_old=expected['base_training_steps'],base_steps_new=br['steps'],native_prefix_changed=native_changed,native_basis_changed=basis_changed,structure_suffix_changed=structure_changed,final_suffix_equals_newly_trained_base=True,support=dict(old=int(before.sum()),new=int(observed.sum()),added=int((observed&~before).sum()),lost=int((before&~observed).sum())),parent_source_bindings=checks,new_model_hashes=new_model_hashes,base_training_coverage=br['training_coverage'],source_acceptance=execution.get('source_acceptance'),evaluation_complete=evaluation is not None,protocols=dict(text='LERF2D native-grid plus selected-render JPEG official contracts',image='GT-masked reference crop category-union development',prompt='registered positive+negative reference points category-union development; no physical3D truth'),no_official_four_query_claim=True,old_scope_validation=expected['old_scope_validation'],new_scope_validation=info['train_frames'][-4:],implementation_sha256=sha256_file(__file__),execution_sha256=sha256_file(root/'execution.json'))
out.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['support'],br['steps'])
