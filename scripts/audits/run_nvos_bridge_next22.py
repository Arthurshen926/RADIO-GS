"""Fixed eight-task positive-only connector control, no target-driven settings."""
import json,os,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
out=Path('output/nvos_bridge_next22');out.mkdir(exist_ok=False)
tasks=['flower','fortress','fern','horns_center','horns_left','leaves','orchids','trex']
reg=dict(tasks=tasks,variants=['baseline','bridge'],baseline='reference_core_component_next19',bridge_min_score=.25,minimum_candidate_neighbors=2,bridge_to_bridge_edges=False,bridge_nodes_selected=False,k=16,radius_multiplier=3,positive_only=True,target_rgb=False,target_gt_used_after_prediction_only=True,source_teacher_reads=False,parameter_search=False,default_promoted=False,implementations={p:sha256_file(p) for p in ['scripts/audits/evaluate_nvos_bridge_next22.py','scripts/audits/nvos_bridge_runtime_next22.py','radio_gs/querying/uncertain_scope_bridge.py']},stages=[],complete=False)
(out/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
for task in tasks:
 root=Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
 cmd=[sys.executable,'scripts/audits/evaluate_nvos_bridge_next22.py','--scene',task,'--root',str(root/'field'),'--geometry-report',str(root/'frozen_geometry/report.json'),'--runs','baseline','bridge','--query-root','output/nvos_positive_queries_20260918','--device','cuda:0','--output',str(out/task)]
 row=dict(task=task,command=cmd,start=time.time());reg['stages'].append(row)
 with (out/(task+'.log')).open('x') as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 row.update(returncode=ret.returncode,seconds=time.time()-row['start']);(out/'execution.json').write_text(json.dumps(reg,indent=2)+'\n')
 if ret.returncode:raise RuntimeError(task+' failed')
 print(task,'complete',flush=True)
reports={s:json.loads((out/s/'report.json').read_text()) for s in tasks}
reg['macro']={a:{m:sum(r['results'][a][m] for r in reports.values())/8 for m in ['iou','pixel_accuracy','average_precision']} for a in ['baseline','bridge']};reg['complete']=True
(out/'execution.json').write_text(json.dumps(reg,indent=2)+'\n');print(reg['macro'],flush=True)
