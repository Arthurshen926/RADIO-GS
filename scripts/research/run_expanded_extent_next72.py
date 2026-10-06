"""Expanded native-resolution runtime grounding and matched extent pilots."""
import json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python'
def run(job):
 gpu,scene,wait_scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.');out=Path('output/expanded_extent_next72');out.mkdir(exist_ok=True)
 # Give the geometry pilot queue ownership of these devices until its final fit.
 while not (Path('output/nvos_transfer_next72/new')/wait_scene/'report.json').exists():
  status=Path('output/nvos_transfer_next72')/(wait_scene+'_execution.json')
  if status.exists() and any(r['exit_code'] for r in json.loads(status.read_text())):raise RuntimeError('NVOS producer failed')
  time.sleep(10)
 authority=str(Path('output/source_coverage_next72')/scene/'coverage64.json');commands=[[SAM,'scripts/research/detect_typed_sources_next44.py','--key',key,'--output-base',str(out),'--source-authority',authority,'--native-resolution'],[PY,'scripts/research/validate_typed_identity_next44.py','--key',key,'--evidence-base',str(out),'--source-authority',authority]]
 rows=[]
 with (out/(scene+'.log')).open('w') as log:
  for c in commands:
   rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=c,exit_code=rc));(out/(scene+'_execution.json')).write_text(json.dumps(rows,indent=2))
   if rc:raise RuntimeError((scene,c[1],rc))
  # Old semantics/new evidence isolates scope improvements; then both expanded.
  for semantic in ['old','coverage64']:
   base='output/joint_semantic_next66/adaptive_allrgb/calibrated' if semantic=='old' else 'output/expanded_semantic_next72/coverage64/calibrated';ready=Path(base)/key/'typed_readout.pt'
   while not ready.exists():time.sleep(10)
   result=str(out/('semantic_'+semantic));c=[PY,'scripts/research/decode_expanded_extent_next72.py','--key',key,'--mode','seeded','--cohort','coverage64','--semantic-base',base,'--output-base',result];rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=c,exit_code=rc));(out/(scene+'_execution.json')).write_text(json.dumps(rows,indent=2))
   if rc:raise RuntimeError((scene,'decode',rc))
   c=[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',result,'--readout','anchored','--pipeline','--calibrated','--element-results',str(Path(result)/key/'typed_readout.pt')];rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=c,exit_code=rc));(out/(scene+'_execution.json')).write_text(json.dumps(rows,indent=2))
   if rc:raise RuntimeError((scene,'evaluate',rc))
if __name__=='__main__':
 with ThreadPoolExecutor(2) as p:list(p.map(run,[(2,'figurines','trex'),(3,'ramen','horns_left')]))
