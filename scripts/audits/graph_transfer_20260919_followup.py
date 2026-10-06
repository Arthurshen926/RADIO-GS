import subprocess,time,json
from pathlib import Path
for scene in ['figurines','ramen']:
 root=Path('output/graph_transfer_20260919')/scene
 while not (root/'report.json').exists():time.sleep(5)
 with (root/'selected.log').open('w') as log:
  p=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/audits/graph_transfer_20260919_selected.py','--scene',scene,'--output',str(root/'selected')],stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError((scene,p.returncode))
 print(scene,'selected complete',flush=True)
