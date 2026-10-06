import os,json,subprocess,queue,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
python='/root/miniconda3/envs/cybersim_agent/bin/python';q=queue.Queue();results=[];lock=threading.Lock()
keys=['lerf_'+s for s in ['figurines','ramen','teatime','waldo_kitchen']]+['scannet_scene'+s+'_00' for s in ['0000','0062','0070','0097','0140','0347','0400','0590']]+['nvos_'+s for s in ['fern','flower','fortress','horns_center','leaves','orchids','trex']]
for k in keys:q.put(k)
def run(gpu):
 while True:
  try:key=q.get_nowait()
  except queue.Empty:return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
  source=Path('output/rendered_scopes_next37')/key;out=Path('output/rendered_scope_propagation_next37')/key
  commands=[]
  if not (source/'field.pt').exists():commands.append([python,'scripts/research/fit_rendered_scope_memberships.py','--key',key])
  # This explicit control changes only membership construction relative to
  # next36 crossfit_anchored. Same fixed confident unaries, no scope veto.
  prepare="import torch;from pathlib import Path; s=Path("+repr(str(source))+");o=Path("+repr(str(out))+");o.mkdir(parents=True,exist_ok=True);d=torch.load(s/'field.pt',weights_only=False);d['scope_compatible']=False;torch.save(d,o/'field.pt');(o/'construction.json').write_text((s/'construction.json').read_text())"
  commands.append([python,'-c',prepare])
  base=str(out.parent)
  if key.startswith('nvos_'):
   for scene in [key[5:]]+(['horns_left'] if key=='nvos_horns_center' else []):commands.append([python,'scripts/research/evaluate_nvos_scope_field.py','--scene',scene,'--base',base])
  else:
   commands.append([python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',base,'--readout','anchored','--pipeline'])
   if key.startswith('lerf_'):commands.append([python,'scripts/research/evaluate_scope_reference_images.py','--key',key,'--base',base])
  started=time.time();rc=0
  with open('/tmp/next37_full_'+key+'.log','w') as log:
   for c in commands:
    rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:break
  with lock:
   row=dict(key=key,gpu=gpu,exit_code=rc,seconds=time.time()-started);results.append(row);Path('/tmp/next37_full_jobs.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
with ThreadPoolExecutor(5) as pool:list(pool.map(run,[0,1,3,4,5]))
