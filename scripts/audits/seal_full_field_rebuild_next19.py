"""Seal only complete four-scene reconstruction, scoring and lineage evidence."""
import json,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19');dst=Path('paper/artifacts/full_field_rebuild_next19_complete.json');assert not dst.exists()
scenes=['figurines','ramen','teatime','waldo_kitchen'];counts=dict(zip(scenes,[56,71,59,22]));arms={'native_compact_calibrated','final','structure32_graph'};bindings={}
def bind(p):
 p=Path(p);assert p.is_file(),p;bindings[str(p)]=sha256_file(p);return p
def read(p):return json.loads(bind(p).read_text())
queue=read(root/'official_evaluation_queue.json');assert queue['complete'] and all(queue['states'][s]=='complete' for s in scenes)
summary=read(root/'evaluation_summary.json');paired=read(root/'paired_comparison.json');assert summary['complete'] and paired['complete'];assert summary['observations']==208
read(root/'registration.json');plot=read(root/'paired_comparison_plot.json');assert plot['plot_sha256']==sha256_file(bind(root/'paired_comparison.png'));assert plot['source_sha256']==sha256_file(root/'paired_comparison.json')
acceptance={}
for scene in scenes:
 base=root/scene;execution=read(base/'execution.json');report=read(base/'evaluation/report.json');lineage=read(base/'lineage_audit.json');assert execution['complete'] and report['complete'] and lineage['complete'];assert report['observations']==counts[scene] and report['cold_functional_replay_exact'];assert set(report['lerf2d'])==arms and set(report['selected'])==arms
 for arm in arms:
  assert len(report['lerf2d'][arm]['observations'])==counts[scene]
  assert report['selected'][arm]['count']==counts[scene] and report['selected'][arm]['missing']==0
 acceptance[scene]=report['source_acceptance'];assert acceptance[scene]==execution['source_acceptance']==lineage['source_acceptance']
 for receipt in lineage.get('recovery_receipts',{}).values():assert sha256_file(bind(receipt['path']))==receipt['sha256']
 for override in lineage.get('runtime_execution_overrides',[]):
  detail=override['override']
  if 'implementation_sha256' in detail:assert sha256_file(bind(detail['command'][1]))==detail['implementation_sha256']
 for path,digest in lineage['new_model_hashes'].items():assert sha256_file(bind(path))==digest
 for path,key in [('scores.pt','scores_sha256'),('selections.pt','selection_sha256')]:assert sha256_file(bind(base/'evaluation'/path))==report[key]
 for task in ['text','reference']:
  online=read(base/'evaluation'/f'{task}_online.json');replay=read(base/'evaluation'/f'{task}_replay.json');assert online['runs']==replay['runs']
 for path,digest in lineage['new_model_hashes'].items():bind(Path(path).with_name('report.json'))
 for name in ['source_plan.json','source_sam_execution.json']:bind(base/'base'/name)
old_native_replay={}
for scene in ['teatime','waldo_kitchen']:
 supplement=Path('output/old_native_control_next19')/scene
 identity=read(supplement/'sealed_input_identity.json');preparation=read(supplement/'preparation.json');assert identity['verified'];old_native_replay[scene]=identity
 assert identity['field_sha256']==preparation['old_field_sha256'] and identity['query_sha256']==preparation['old_query_sha256']
 for name in ['report.json','selected/report.json','scores.pt']:bind(supplement/name)
manifest=Path('output/frozen_cross_benchmark_method_20260918/source_manifest.json');old=read(manifest);assert len(old)==1773
changed=[path for path,digest in old.items() if not Path(path).is_file() or sha256_file(path)!=digest];assert not changed,changed
for name in ['native_calibration_replay.json','sam_regeneration_equivalence_next19.json','evaluation_overlap_authorization.json']:bind(root/name)
for name in ['evaluate_full_field_rebuild_next19.py','audit_full_field_rebuild_lineage_next19.py','compare_full_field_rebuild_next19.py','plot_full_field_rebuild_next19.py','seal_full_field_rebuild_next19.py']:bind(Path('scripts/audits')/name)
result=dict(complete=True,sealed_unix=time.time(),scenes=scenes,observations_per_arm=208,arms=sorted(arms),source_acceptance=acceptance,all_source_acceptance=all(acceptance.values()),new_cold_replay_exact_all=True,old_native_supplement_identity=old_native_replay,frozen_source_count=1773,frozen_source_changes=changed,geometry_retrained=False,expanded_source_and_training_budget=True,independent_test=False,official_four_query=False,promoted=False,no_per_scene_winners=True,artifacts=bindings,macro=summary['macro'])
dst.write_text(json.dumps(result,indent=2)+'\n');print(dst);print(json.dumps(result['macro'],indent=2))
