"""Replay one explicit task-dependent recipe across the frozen benchmark cohort."""
import json,os,queue,shutil,subprocess,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
python='/root/miniconda3/envs/cybersim_agent/bin/python';base=Path('output/taskwise_scope_composition_next38');base.mkdir(parents=True,exist_ok=True);work=queue.Queue();results=[];lock=threading.Lock()
for source in sorted(Path('output/scope_composition_native_next38').glob('*/field.pt')):
 key=source.parent.name;data=torch.load(source,weights_only=False);data['scope_task_policy']='typed'
 category=Path('output/category_competition_next38')/key/'field.pt'
 if category.exists():data['category_competition']=torch.load(category,weights_only=False)['category_competition']
 out=base/key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
 if (source.parent/'calibration.pt').exists():shutil.copyfile(source.parent/'calibration.pt',out/'calibration.pt')
 work.put(key)
(base/'recipe.json').write_text(json.dumps(dict(scope_solver='nonnegative joint composition',fixed_native_cores=True,task_policy={'semantic_text':'scope composition','categorical_text':'scope composition and source partial-label competition','image_category_union':'retained image field','reference_prompt':'scope composition and retained authorized prompt completion'},source_scope_resolution=[96,128],no_per_scene_winner_selection=True,development_recipe=True),indent=2)+'\n')
def run(gpu):
 while True:
  try:key=work.get_nowait()
  except queue.Empty:return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
  if key.startswith('nvos_'):
   commands=[[python,'scripts/research/evaluate_nvos_scope_field.py','--scene',s,'--base',str(base)] for s in [key[5:]]+(['horns_left'] if key=='nvos_horns_center' else [])]
  else:
   commands=[[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline','--calibrated']]
   if key.startswith('lerf_'):commands.append([python,'scripts/research/evaluate_scope_reference_images.py','--key',key,'--base',str(base)])
  with (base/key/'execution.log').open('w') as log:
   for command in commands:
    rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:break
  with lock:
   row=dict(key=key,exit_code=rc);results.append(row);(base/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(row,flush=True)
with ThreadPoolExecutor(4) as pool:list(pool.map(run,[0,1,2,3]))
