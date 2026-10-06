import os,json,queue,subprocess,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
python='/root/miniconda3/envs/cybersim_agent/bin/python';work=queue.Queue();results=[];lock=threading.Lock();base=Path('output/category_competition_next38');base.mkdir(parents=True,exist_ok=True)
for scene in ['0000','0062','0070','0097','0140','0347','0400','0590']:work.put('scannet_scene'+scene+'_00')
def run(gpu):
 while True:
  try:key=work.get_nowait()
  except queue.Empty:return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
  with (base/(key+'.log')).open('w') as log:
   for command in [[python,'scripts/research/fit_category_competition.py','--key',key],[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline','--calibrated']]:
    rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:break
  with lock:
   row=dict(key=key,exit_code=rc);results.append(row);(base/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(row,flush=True)
with ThreadPoolExecutor(4) as pool:list(pool.map(run,[0,1,2,3]))
