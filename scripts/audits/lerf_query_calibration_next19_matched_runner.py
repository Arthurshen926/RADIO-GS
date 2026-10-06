import subprocess
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 with open(f'/tmp/lerf_query_calibration_next19_matched_{scene}.log','w')as f:
  r=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/audits/lerf_query_calibration_next19_matched.py','--scene',scene],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError((scene,r.returncode))
 print(scene,'complete',flush=True)
