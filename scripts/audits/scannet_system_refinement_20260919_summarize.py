"""Seal full8 fixed controls; no per-scene winner selection."""
import json
from pathlib import Path
import numpy as np
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/scannet_system_refinement_20260919_v2');continuation=Path('output/scannet_system_refinement_20260919_continued');out=root/'summary.json';assert not out.exists();scenes=['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0347_00','scene0400_00','scene0590_00'];names=['native','calibrated_native','scope','equal_logit','bounded_logit'];reports={};paths={}
for scene in scenes:
 path=root/scene/'report.json'
 if not path.exists():path=continuation/scene/'report.json'
 x=json.loads(path.read_text());assert x['baseline_exact'] and x['predictions_saved_before_gt'];reports[scene]=x;paths[scene]=dict(path=str(path),sha256=sha256_file(path))
macro={name:{split:{metric:float(np.mean([reports[s]['metrics'][name][split][metric] for s in scenes])) for metric in ['miou','macc']} for split in ['19','15','10']} for name in names};source={name:{split:{metric:float(np.mean([reports[s]['source_macro'][split][name][metric] for s in scenes])) for metric in ['agreement','miou']} for split in ['19','15','10']} for name in names};changes={name:{scene:{split:reports[scene]['metrics'][name][split]['miou']-reports[scene]['metrics']['native'][split]['miou'] for split in ['19','15','10']} for scene in scenes} for name in names[1:]}
result=dict(schema='scannet_fixed_native_scope_controls.v1',scenes=scenes,cohort_complete=len(reports)==8,official_protocol='scannet-ovs-paper8-v1',input_track='provided-unlabelled-mesh',geometry_unchanged=True,independent_test=False,baseline_replayed_exact=True,macro=macro,source_teacher_self_consistency=source,per_scene_change_vs_raw_native=changes,source_rejected_scenes=[s for s in scenes if reports[s]['scope_source_acceptance'] is False],source_acceptance={s:reports[s]['scope_source_acceptance'] for s in scenes},prediction_registration='Parameters fixed before evaluations; source metrics do not fit weights. Native and scope logits have common probability form but are not proven cross-class calibrated.',promotion=False,scope_contract_limitation='Scope heads model source object ranges; ScanNet labels category unions across instances. No claim blending removes this mismatch.',reports=paths,implementation_sha256=sha256_file(__file__))
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(macro,indent=2));print('source',json.dumps(source,indent=2));print('source_rejected',result['source_rejected_scenes'])
