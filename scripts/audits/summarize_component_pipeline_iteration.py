"""Seal the complete component-pipeline development iteration, including failures."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 base=Path('/mnt/pool/sqy/results/RADIO-GS/output');art=Path('paper/artifacts');output=art/'component_pipeline_iteration_20260915.json'
 if output.exists():raise FileExistsError(output)
 inventory=[];evaluations={};cold={}
 for scene,old,oldsteps,steps in [('figurines','four_teacher_three_query_v2_20260915',366,768),('ramen','three_query_coverage_expansion_20260915',660,1932)]:
  oldroot=base/old/scene;root=base/'component_retrieval_pipeline_20260915'/scene
  names=[(oldroot,p+str(oldsteps)) for p in ['element_random_anchor','element_initialized_free','element_initialized_anchor','element_random_free','element_dense_free','element_dense_anchor','element_dense_anchor_no_radio','element_direction_fixed','element_direction_constrained','grounded_element_free','grounded_element_constrained','grounded_element_joint']]
  names += [(oldroot,'component_budget_control'+str(steps))]
  names += [(root,p+str(steps)) for p in ['component_pipeline','component_native_free','component_native_constrained','component_native_no_radio','component_area_constrained','component_area_no_radio','component_pipeline_seed43_','component_pipeline_no_radio_']]
  geometry_report=base/'geometry_only_repair_final_20260914'/scene/'report.json';g=json.loads(geometry_report.read_text())['results']['independent_scale'];assert sha256_file(Path(g['geometry']))==g['geometry_sha256']
  for r,name in names:
   folder=r/name;report=json.loads((folder/'report.json').read_text());digest=sha256_file(folder/'field.pt');assert digest==report['field_sha256'];assert report['geometry_binding_unchanged'] and report['benchmark_ground_truth'] is False
   assert report['source_regions_available']==report['source_regions_supervised'];assert report['episode_coverage']['complete'];assert report['conflicting_training_labels_excluded'] and report['guard_signed_targets']
   if 'no_radio' in name:assert 'radio' not in report['opened_teacher_files'] and 'radio' not in report['enabled_auxiliary_teachers']
   if report.get('native_element_min_cosine',0):assert report['native_retention']['minimum']>=report['native_element_min_cosine']-1e-5
   inventory.append({'scene':scene,'name':name,'field':str(folder/'field.pt'),'field_sha256':digest,'report_sha256':sha256_file(folder/'report.json'),'geometry_sha256':g['geometry_sha256'],'steps':report['steps'],'seed':report['seed'],'teacher_roles':report['enabled_auxiliary_teachers'],'opened_teachers':report['opened_teacher_files'],'source_regions':report['source_regions_supervised'],'coverage':report['training_coverage'],'native_retention':report.get('native_retention'),'optimization_tasks_after':report['optimization_tasks_after']})
  for suffix in ['consistent_element_teacher_eval','dense_element_teacher_eval','native_direction_retention_eval','grounded_element_teacher_eval','component_pipeline_eval','component_budget_eval','component_area_eval','component_replication_eval','component_three_query','component_budget_three_query','component_replication_three_query','component_source_order']:
   p=art/f'{scene}_{suffix}_20260915.json';d=json.loads(p.read_text());evaluations[str(p)]={'sha256':sha256_file(p),'metrics':{n: {k:v for k,v in r.items() if k not in ['observations','prompt_refinement']} for n,r in d['runs'].items()}}
  p=art/f'{scene}_component_no_radio_cold_20260915.json';d=json.loads(p.read_text());assert d['training_cache_reads_forbidden'] and d['geometry_unchanged'];cold[scene]={'path':str(p),'sha256':sha256_file(p),**d}
 result={'schema':'component_pipeline_iteration.v1','completed_training_runs':len(inventory),'policy':'retain correctness repairs; unified source supports are preferred development candidate, not a universal metric promotion; hard native-direction constraints rejected as default','official_benchmark_protocol':False,'independent_3d_truth_evaluated':False,'training_inventory':inventory,'evaluations':evaluations,'cold':cold}
 output.write_text(json.dumps(result,indent=2)+'\n');print(output,len(inventory))
if __name__=='__main__':main()
