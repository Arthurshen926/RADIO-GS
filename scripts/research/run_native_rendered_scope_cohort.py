import json,os,queue,subprocess,threading,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
python='/root/miniconda3/envs/cybersim_agent/bin/python';keys=['lerf_'+s for s in ['figurines','ramen','teatime','waldo_kitchen']]+['scannet_scene'+s+'_00' for s in ['0000','0062','0070','0097','0140','0347','0400','0590']]
for key in keys:
 s=Path('output/rendered_scope_propagation_next37')/key;out=Path('output/native_rendered_scope_next37')/key;out.mkdir(parents=True,exist_ok=True);d=torch.load(s/'field.pt',weights_only=False);d['scope_native_dictionary']=True;d['native_modalities']=['text'];d['scope_modalities']=['text'];torch.save(d,out/'field.pt');shutil.copyfile(s/'construction.json',out/'construction.json');shutil.copyfile(Path('output/scope_query_modes_next34')/key/'calibration.pt',out/'calibration.pt')
q=queue.Queue();results=[]
for k in keys:q.put(k)
lock=threading.Lock()
def run(gpu):
 while True:
  try:key=q.get_nowait()
  except queue.Empty:return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
  commands=[[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base','output/native_rendered_scope_next37','--readout','anchored','--pipeline','--calibrated']]
  if key.startswith('lerf_'):commands.append([python,'scripts/research/evaluate_scope_reference_images.py','--key',key,'--base','output/native_rendered_scope_next37'])
  with open('/tmp/next37_combined_'+key+'.log','w') as log:
   for command in commands:
    rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:break
  with lock:
   row=dict(key=key,exit_code=rc);results.append(row);Path('/tmp/next37_combined_jobs.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
with ThreadPoolExecutor(2) as pool:list(pool.map(run,[2,5]))
