import time,subprocess
from pathlib import Path
while not Path('output/lerf_system_refinement_20260919/ramen/selected/report.json').exists():time.sleep(5)
for scene in ['teatime','waldo_kitchen']:
 root=Path('output/lerf_system_refinement_20260919')/scene
 if scene=='figurines':
  while not (root/'report.json').exists():time.sleep(5)
 for label,args in [('build',[]),('source',['--output',str(root/'source')]),('selected',['--output',str(root/'selected')])]:
  if scene=='figurines' and label=='build':continue
  script='lerf_system_refinement_20260919'+(''if label=='build'else'_'+label)+'.py'
  with open(f'/tmp/lerf_system_refinement_{scene}_{label}.log','w')as log:
   r=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/audits/'+script,'--scene',scene,*args],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError((scene,label,r.returncode))
  print(scene,label,'complete',flush=True)
