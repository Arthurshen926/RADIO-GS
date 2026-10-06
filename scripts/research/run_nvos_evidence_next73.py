"""Cross old/new carriers with the same source-rendered prior and video."""
import os, subprocess, json, argparse
p=argparse.ArgumentParser();p.add_argument('--fusion',choices=['joint','completion','unsupported','video'],default='joint');p.add_argument('--gpu',default='1');p.add_argument('--tracks-base',default='output/source_video_next43');p.add_argument('--output-base');p.add_argument('--fit-scale',type=int,default=1);a=p.parse_args()
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
base=Path('output/nvos_evidence_next73' if a.fusion=='joint' else ('output/nvos_completion_next73' if a.fusion=='completion' else 'output/nvos_unsupported_next73'));base=Path(a.output_base) if a.output_base else base;base.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',PYTHONPATH='.')
rows=[]
for scene in ['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']:
 for carrier in ['old','new']:
  if (base/carrier/scene/'report.json').exists():continue
  cmd=[PY,'scripts/research/fit_nvos_evidence_next73.py','--scene',scene,'--carrier',carrier,'--fusion',a.fusion,'--output-base',str(base)]
  if a.fusion=='video':cmd=[PY,'scripts/research/fit_nvos_geometry_transfer_next72.py','--scene',scene,'--carrier',carrier,'--output-base',str(base),'--fit-scale',str(a.fit_scale)]
  cmd+=['--tracks-base',a.tracks_base]
  with (base/f'{scene}_{carrier}.log').open('w') as log:
   rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  rows.append(dict(command=cmd,exit_code=rc));(base/'execution.json').write_text(json.dumps(rows,indent=2))
  if rc:raise RuntimeError((scene,carrier,rc))
summary={}
for carrier in ['old','new']:
 records={s.parent.name:json.loads(s.read_text())['metrics'] for s in (base/carrier).glob('*/report.json')}
 summary[carrier]=dict(scenes=records,macro={k:sum(r[k] for r in records.values())/len(records) for k in ['iou','average_precision']})
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
