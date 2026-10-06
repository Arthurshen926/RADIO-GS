import subprocess
from pathlib import Path
for scene in ['teatime','waldo_kitchen']:
 for script,args,label in [('graph_transfer_20260919_extension.py',[], 'final'),('graph_transfer_20260919_final_selected.py',['--output',f'output/graph_transfer_final_20260919/{scene}/selected'],'final_selected'),('graph_transfer_20260919_relations.py',[],'relations')]:
  if label=='relations':continue
  with open(f'/tmp/graph_transfer_{label}_{scene}.log','w') as f:
   r=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python',f'scripts/audits/{script}','--scene',scene,*args],stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError((scene,label,r.returncode))
  print(scene,label,'complete',flush=True)
