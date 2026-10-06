"""Completed native-map cohort and matching legacy subset, no winner selection."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('output/scannet_native_completion')

def main():
 rows=json.loads((ROOT/'source_plan.json').read_text())['scenes']
 reports={r['scene']:json.loads((ROOT/r['scene']/'evaluation/report.json').read_text()) for r in rows}
 builds={s:json.loads((ROOT/s/'build.json').read_text()) for s in reports}
 legacy=json.loads(Path('output/scannet_runtime_integration/summary.json').read_text())
 common=sorted(set(reports)&set(legacy['scenes']))
 means=lambda scenes:{s:float(np.mean([reports[scene]['point_metrics']['map'][s]['miou'] for scene in scenes])) for s in ['19','15','10']}
 execution=json.loads((ROOT/'execution.json').read_text())
 extra=[json.loads(p.read_text()) for p in ROOT.glob('scene*/shared_kernel_execution.json')]
 reused=sum(p.is_symlink() for p in ROOT.glob('scene*/source_sam/frame_*.pt'))
 source_count=sum(b['source_count'] for b in builds.values());regions=sum(sum(r['regions'] for r in b['records']) for b in builds.values())
 result=dict(scene_macro=means(reports),matched_eight_scene_macro=means(common),legacy_eight_scene_macro=legacy['scene_macro']['point_metrics'],new_minus_legacy_map={s:means(common)[s]-legacy['scene_macro']['point_metrics']['map'][s] for s in ['19','15','10']},new_minus_legacy_source={s:means(common)[s]-legacy['scene_macro']['point_metrics']['source'][s] for s in ['19','15','10']},scenes=reports,source_frames=source_count,source_frames_per_scene={s:b['source_count'] for s,b in builds.items()},native_encoded_regions=regions,retained_same_recipe_source_caches=reused,newly_generated_sam_frames_including_initial_partial_run=source_count-reused,source_images_capped=False,shared_kernel_for_all_scenes=all(b['feature_writer']=='radio_gs.pipeline.native_reconstruction.NativeFieldAccumulator' for b in builds.values()),native_field_file_bytes=sum((ROOT/s/'native_field.pt').stat().st_size for s in reports),elements=sum(b['elements'] for b in builds.values()),supported_elements=sum(b['supported_elements'] for b in builds.values()),query_specific_map_parameters=False,source_assisted_head_migrated=False,official_protocol_equivalence_verified=False,independent_test=False,new_model_promoted=False,cost=dict(cohort_worker_seconds=execution['summed_worker_seconds'],shared_kernel_rebuild_worker_seconds=sum(r['seconds'] for r in extra),cohort_wall_seconds=execution['elapsed_seconds'],includes='SAM generation, native encoding, map writing, query encoding and point scoring within scheduled workers; second shared-kernel rewrites separately',excludes='historical cache/model creation and interrupted initial 17-source SAM job; worker seconds are allocated-GPU wall time, not hardware busy time'),source_head_gap=dict(authorized_class_teacher_frames=176,missing_class_teacher_scenes=['scene0200_00','scene0645_00'],new_text_forwards_for_complete_source_contract=(1158-176)*19,classification_scale_calibration_required=True),claim='complete shared-reconstruction map-only baseline; no SOTA or restored full source-assisted system claim')
 result['scene_macro_macc']={s:float(np.mean([r['point_metrics']['map'][s]['macc'] for r in reports.values()])) for s in ['19','15','10']}
 result['matched_geometry_sha256_identical']=all(json.loads((Path('configs/consolidated/scannet')/(scene+'.scene.json')).read_text())['geometry']['sha256']==json.loads((Path('configs/consolidated/scannet_native')/(scene+'.scene.json')).read_text())['geometry']['sha256'] for scene in common)
 class_rows={}
 for scene in common:
  old=json.loads((Path('output/scannet_runtime_integration')/scene/'evaluation/report.json').read_text())
  for cid,row in reports[scene]['point_metrics']['map']['19']['per_class'].items():
   if row['gt_count']==0:continue
   entry=class_rows.setdefault(cid,dict(name=row['name'],new=[],legacy_map=[],legacy_source=[]))
   entry['new'].append(row['iou'])
   for arm in ['map','source']:entry['legacy_'+arm].append(old['point_metrics'][arm]['19']['per_class'][cid]['iou'])
 result['matched_class_iou_diagnostic']={cid:dict(name=r['name'],scenes=len(r['new']),**{k:float(np.mean(r[k])) for k in ['new','legacy_map','legacy_source']}) for cid,r in class_rows.items()}
 (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 config=Path('configs/recipes/scannet_unified_native.json');recipe=json.loads(config.read_text());recipe['status']='complete_not_promoted_over_retained_source_system';recipe['results']=str(ROOT/'summary.json');config.write_text(json.dumps(recipe,indent=2)+'\n')
 cohort=dict(scenes=[dict(scene=s,scene_config=f'configs/consolidated/scannet_native/{s}.scene.json',query_config=f'configs/consolidated/scannet_native/{s}.queries.json',evaluation_config=f'configs/consolidated/scannet_native/{s}.evaluation.json',report=str(ROOT/s/'evaluation/report.json')) for s in reports],method='shared_native_map_only',source_frames=source_count,official_protocol_equivalence_verified=False)
 Path('configs/consolidated/scannet_native/cohort.json').write_text(json.dumps(cohort,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['scene_macro','matched_eight_scene_macro','legacy_eight_scene_macro','source_frames','native_encoded_regions','cost']},indent=2))
if __name__=='__main__':main()
