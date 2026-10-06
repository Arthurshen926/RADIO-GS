import os,subprocess,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PY='/root/miniconda3/envs/cybersim_agent/bin/python';root=Path('output/clip_field_next58')
def run(job):
 gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
 rows=[];base=root/'contrastive'
 with (root/(scene+'_contrastive.log')).open('w') as f:
  for c in [[PY,'scripts/research/read_clip_contrastive_next58.py','--key',key],[PY,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline','--calibrated','--element-results',str(base/key/'typed_readout.pt')]]:
   rc=subprocess.run(c,env=env,stdout=f,stderr=subprocess.STDOUT).returncode;rows.append(dict(command=c,exit_code=rc));print(scene,c[1],rc,flush=True)
   if rc:break
 return rows
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])))
(root/'execution_contrastive.json').write_text(json.dumps(rows,indent=2)+'\n')
