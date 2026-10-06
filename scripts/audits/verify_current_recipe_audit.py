"""Verify full-task replay equality and report rejected query-scope intervention."""
import json
from pathlib import Path
from itertools import combinations
from radio_gs.utils.immutable_artifacts import sha256_file
recipe=Path('configs/recipes/current_retained_96d.json');cfg=json.loads(recipe.read_text());root=Path('output/current_recipe_audit_20260918');report=dict(schema='current_recipe_and_query_collisions.v1',recipe_sha256=sha256_file(recipe),promoted=False,independent_final_test=False,v2_full_rebuild_complete=False,scenes={})
run='counterfactual_extent_benefit_admission'
for scene,config in cfg['scenes'].items():
 for path,h in config['input_sha256'].items():assert sha256_file(path)==h
 base=root/scene;out=base/'sealed_recipe';execution=json.loads((out/'execution.json').read_text());assert execution['complete'] and all(r['status']=='complete' for r in execution['stages']);comparisons={};artifacts={}
 for task in ['text','reference']:
  a=json.loads((out/(task+'_online.json')).read_text());b=json.loads((out/(task+'_replay.json')).read_text());old=json.loads((Path(config['root']).parent/'counterfactual_extent_benefit_admission_validation'/(task+'.json')).read_text())
  assert a['runs']==b['runs']==old['runs'],task+' full per-observation metrics changed'
  for mode in ['online','replay']:
   path=out/(task+'_'+mode+'.json');trace=json.loads(path.with_suffix('.routes.json').read_text());assert trace['evaluation_sha256']==sha256_file(path);assert trace['encoding_bank_sha256']==sha256_file(path.with_suffix('.encodings.pt'));assert not trace['query_relative_scope_control'];assert not trace['training'];artifacts[str(path)]=sha256_file(path)
  comparisons[task]={'all_observations_exact_vs_recorded_replay':True,'all_observations_exact_vs_retained_evaluation':True}
 trace=json.loads((out/'text_online.routes.json').read_text());r=trace['traces'][0];names=trace['texts'];collision=[{'left':names[i],'right':names[j]} for i,j in combinations(range(len(names)),2) if r['score_sha256'][i]==r['score_sha256'][j]]
 source=json.loads((base/'source_collisions_detailed.json').read_text());control=json.loads((base/'source_collisions_relative.json').read_text());assert source['field_sha256']==control['field_sha256'];assert source['evidence_sha256']==control['evidence_sha256'];assert [q['query'] for q in source['queries']]==[q['query'] for q in control['queries']]
 def counts(d):
  pp=[p for p in d['identical_membership_coobserved_pairs'] if p['minimum_source_mask_iou']<.5]
  return {'low_agreement_pairs':len(pp),'nonempty_low_agreement_pairs':sum(p['left']['foreground_elements']>0 for p in pp),'same_accepted_concept_nonempty_pairs':sum(p['left']['foreground_elements']>0 and p['left']['concept_accepted'] and p['right']['concept_accepted'] and p['left']['concept']==p['right']['concept'] for p in pp)}
 candidatepath=out/'text_relative_control.json';candidate=json.loads(candidatepath.read_text());assert candidate['effective_readout']=='query_relative_scope_frozen_policy_control';cr=json.loads(candidatepath.with_suffix('.routes.json').read_text());assert cr['query_relative_scope_control'] and cr['evaluation_sha256']==sha256_file(candidatepath);assert cr['encoding_replay_sha256']==sha256_file(out/'text_online.encodings.pt');retained=json.loads((out/'text_online.json').read_text())['runs'][run];new=candidate['runs'][run];sourcecontrol=json.loads((base/'collision_scope_control.json').read_text());assert sourcecontrol['source_only'] and not sourcecontrol['benchmark_data_used']
 for name in ['source_collisions_detailed.json','source_collisions_relative.json','collision_scope_control.json']:artifacts[str(base/name)]=sha256_file(base/name)
 artifacts[str(candidatepath)]=sha256_file(candidatepath)
 report['scenes'][scene]=dict(full_evaluation=comparisons,text_queries=len(names),fixed_concept_used=sum(c and u for c,u in zip(r['concept_accepted'],r['final_scope_used'])),final_scope_used=sum(r['final_scope_used']),identical_real_text_responses=collision,source_collisions=counts(source),source_relative_collisions=counts(control),source_selected_case_iou=sourcecontrol['macro'],text_control={'retained_iou':retained['observation_miou'],'relative_iou':new['observation_miou'],'retained_ap':retained['average_precision_macro'],'relative_ap':new['average_precision_macro']},artifacts=artifacts)
casepath=root/'scope_casebook.json';casebook=json.loads(casepath.read_text());assert not casebook['human_verified']
for case in casebook['cases']:assert not case['training_use'] and sha256_file(case['figure'])==case['figure_sha256']
report['casebook_sha256']=sha256_file(casepath);report['casebook_cases']=len(casebook['cases']);report['conclusion']='Retained full online/replayed metrics reproduce exactly. Query-relative correction removes fixed-head collisions but has no shared retrieval gain; reject promotion. V2 full reconstruction, human-confirmed scope labels and cross-benchmark joint evaluation remain pending.'
path=Path('paper/artifacts/current_recipe_query_scope_audit_20260918.json');path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['scenes'],indent=2))
