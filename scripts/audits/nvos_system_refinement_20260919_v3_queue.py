"""Registered positive-only spatial prompt retention; immutable baseline and GT firewall."""
import json,os,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
out=Path('output/nvos_system_refinement_20260919_v3');out.mkdir(exist_ok=False)
tasks=['flower','fortress','fern','horns_center','horns_left','leaves','orchids','trex']
r=dict(schema='nvos_system_refinement_registration.v1',tasks=tasks,geometry='same frozen baseline',variants=['baseline','spatial_component','structure_component','structure_smooth_component'],iterations=80,lr=.05,positive_target=.55,sam_positive_weight=.25,fidelity_weight=.01,negative_scribbles_allowed=False,target_rgb_allowed=False,target_gt_before_prediction_allowed=False,source_sam_outside='unknown, never negative',no_target_tuning=True,k=16,radius_multiplier=3,structure_cosine_min=.5,positive_seed_fraction=.1,graph_lambda=1.,graph_confidence=.5,cg_iterations=64,reference_exclusion='none; connected-instance inference only',baseline_iou=.68612240254879,baseline_numeric_tolerance=1e-6,baseline_binary_masks_must_replay_exact=True,score_contract='unchanged baseline continuous bilinear then >=0.5',full_method=False,script_sha256=sha256_file('scripts/audits/nvos_system_refinement_20260919_v3_evaluate.py'),stages=[],complete=False)
(out/'registration.json').write_text(json.dumps(r,indent=2)+'\n')
for task in tasks:
 parent=Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
 cmd=[sys.executable,'scripts/audits/nvos_system_refinement_20260919_v3_evaluate.py','--scene',task,'--root',str(parent/'field'),'--geometry-report',str(parent/'frozen_geometry/report.json'),'--runs','baseline','spatial_component','structure_component','structure_smooth_component','--query-root','output/nvos_positive_queries_20260918','--device','cuda:0','--output',str(out/task)]
 row=dict(task=task,command=cmd,status='running',start=time.time());r['stages'].append(row);(out/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
 with (out/(task+'.log')).open('x') as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
 row.update(status='failed' if ret.returncode else 'complete',returncode=ret.returncode,seconds=time.time()-row['start']);(out/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
 if ret.returncode:raise RuntimeError(task+' failed')
results={task:json.loads((out/task/'report.json').read_text()) for task in tasks}
r['macro']={variant:{metric:sum(x['results'][variant][metric] for x in results.values())/8 for metric in ['iou','pixel_accuracy','average_precision']} for variant in r['variants']};r['complete']=True
(out/'execution.json').write_text(json.dumps(r,indent=2)+'\n');print(r['macro'],flush=True)
