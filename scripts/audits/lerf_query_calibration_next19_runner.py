import subprocess,time
from pathlib import Path
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 out=Path('output/lerf_query_calibration_next19')/scene
 if scene=='figurines':
  while not (out/'report.json').exists():time.sleep(5)
 for phase,args in [('build',[]),('selected',['--output',str(out/'selected')])]:
  if scene=='figurines'and phase=='build':continue
  script='lerf_query_calibration_next19'+(''if phase=='build'else'_selected')+'.py'
  with open(f'/tmp/lerf_query_calibration_next19_{scene}_{phase}.log','w')as f:
   r=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/audits/'+script,'--scene',scene,*args],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError((scene,phase,r.returncode))
  print(scene,phase,'complete',flush=True)
