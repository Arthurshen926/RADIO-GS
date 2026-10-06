"""Seal the complete v2 rebuild and compare all tasks with immutable v1 controls."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

torch.set_num_threads(4)
base=Path('output/v2_scope_rebuild_20260918')
run='counterfactual_extent_benefit_admission'
summary=dict(schema='v2_scope_rebuild_verification.v1',independent_final_test=False,scenes={},all_expert_plan_items_complete=False)
recipe=dict(schema='v2_scope_recipe.v1',name='native64_structure32_v2_scopes',status='correctness_repaired_development_candidate',training_recipe_complete_from_verified_base=True,from_raw_rgb_training_recipe_complete=False,independent_final_test=False,scenes={})
for scene in ['figurines','ramen']:
 root=base/scene;fieldroot=root/'field';oldroot=Path('output/pixel_evidence_rebuild_20260917')/scene/'field'
 execution=json.loads((root/'execution.json').read_text());assert execution['finished']
 for stage in execution['steps']:
  assert stage['status']=='complete'
  for path,digest in stage['outputs'].items():assert sha256_file(path)==digest
  for path,digest in stage['implementations'].items():assert sha256_file(root/'sources'/stage['name']/Path(path).name)==digest
 for path,digest in execution['reused_input_hashes'].items():assert sha256_file(path)==digest
 gp=fieldroot/'caption_grounded_regions.pt';ep=fieldroot/'caption_region_evidence.pt'
 grounding=torch.load(gp,weights_only=False);evidence=torch.load(ep,weights_only=False)
 assert grounding['schema']=='caption_grounded_regions.v2'
 assert evidence['grounding_sha256']==sha256_file(gp)
 assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256']
 for g in ['source_phrase_groups.json','grounded_semantic_transfer96/report.json','source_region_extent96/report.json','source_extent_rendered_residual96/report.json']:
  assert json.loads((fieldroot/g).read_text())['evidence_sha256']==sha256_file(ep)
 old=torch.load(oldroot/run/'field.pt',weights_only=False);new=torch.load(fieldroot/run/'field.pt',weights_only=False)
 assert old['geometry_binding']==new['geometry_binding'] and old['query_encoder']==new['query_encoder']
 for key in ['latent','observed','native_decoder.weight','native_decoder.bias']:
  assert torch.equal(old['state_dict'][key],new['state_dict'][key]),key
 tasks={}
 for task in ['text','reference']:
  online=json.loads((root/(task+'_online.json')).read_text());replay=json.loads((root/(task+'_replay.json')).read_text());assert online['runs']==replay['runs']
  a=json.loads((root/(task+'_online.routes.json')).read_text());b=json.loads((root/(task+'_replay.routes.json')).read_text());assert a['traces']==b['traces']
  for mode,trace in [('online',a),('replay',b)]:
   path=root/(task+'_'+mode+'.json');assert trace['evaluation_sha256']==sha256_file(path);assert trace['encoding_bank_sha256']==sha256_file(path.with_suffix('.encodings.pt'))
  prior=json.loads((Path('output/current_recipe_audit_20260918')/scene/'sealed_recipe'/(task+'_online.json')).read_text())
  result=online['runs'][run];previous=prior['runs'][run]
  if task=='text':tasks[task]=dict(old_iou=previous['observation_miou'],new_iou=result['observation_miou'],old_ap=previous['average_precision_macro'],new_ap=result['average_precision_macro'],observations=result['tasks'])
  else:
   assert len(result['observations'])==len(previous['observations'])
   assert all(x['prompt']==y['prompt'] for x,y in zip(result['observations'],previous['observations']))
   for modality in ['image','prompt']:
    tasks[modality]=dict(old_iou=previous['macro'][modality]['iou'],new_iou=result['macro'][modality]['iou'],old_ap=previous['macro'][modality]['average_precision'],new_ap=result['macro'][modality]['average_precision'],observations=result['tasks_per_modality'])
 hashes={str(path):sha256_file(path) for path in [root/'execution.json',gp,ep,fieldroot/run/'field.pt',fieldroot/run/'report.json',fieldroot/'teachers.json',fieldroot/'native_query_encoder.json',fieldroot/'responsibility.json']}
 summary['scenes'][scene]=dict(metrics=tasks,native_anchor_and_geometry_exact=True,full_online_replay_metrics_and_scores_exact=True,prompt_per_observation_exact=True,caption_regions=len(evidence['records']),hashes=hashes)
 recipe['scenes'][scene]=dict(root=str(fieldroot),run=run,geometry_report=f'output/geometry_only_repair_final_20260914/{scene}/report.json',input_sha256=hashes)
summary['joint_iou_improvement']=all(m['new_iou']>=m['old_iou'] for s in summary['scenes'].values() for m in s['metrics'].values())
summary['joint_ap_nondegradation']=all(m['new_ap']>=m['old_ap'] for s in summary['scenes'].values() for m in s['metrics'].values())
summary['implementation_sha256']=sha256_file(__file__)
out=Path('paper/artifacts/v2_scope_rebuild_20260918.json');assert not out.exists();out.write_text(json.dumps(summary,indent=2)+'\n')
recipe['verification_sha256']=sha256_file(out);path=Path('configs/recipes/v2_scope_96d.json');assert not path.exists();path.write_text(json.dumps(recipe,indent=2)+'\n')
print(json.dumps({s:r['metrics'] for s,r in summary['scenes'].items()},indent=2))
