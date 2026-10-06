import os,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path('output/text_video_next63');root.mkdir(parents=True,exist_ok=True)
def run(job):
 gpu,scene=job;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
 with (root/(scene+'.log')).open('w') as log:
  rc=subprocess.run(['/root/miniconda3/envs/sam3/bin/python','scripts/research/track_text_sources_next63.py','--key','lerf_'+scene],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 print(scene,rc,flush=True);return dict(scene=scene,exit_code=rc)
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(run,enumerate(['figurines','ramen','teatime','waldo_kitchen'])))
(root/'tracking_execution.json').write_text(json.dumps(rows,indent=2)+'\n')
