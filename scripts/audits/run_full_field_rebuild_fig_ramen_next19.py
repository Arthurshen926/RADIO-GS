"""GPU4 sequential complete cohort runner; no frozen mutation."""
import os,subprocess,json
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-b3fc88ce-5b9d-dfaa-6d9b-2268a1394aab'
root=Path('output/full_field_rebuild_next19');root.mkdir(exist_ok=True);receipt=root/'figurines_ramen_queue.json';rows=[]
for scene in ['figurines','ramen']:
 row=dict(scene=scene,status='running');rows.append(row);receipt.write_text(json.dumps(rows,indent=2))
 with (root/(scene+'_worker.log')).open('x')as log:
  r=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/audits/run_full_field_rebuild_next19.py','--scene',scene],stdout=log,stderr=subprocess.STDOUT)
 row.update(returncode=r.returncode,status='complete'if r.returncode==0 else'failed');receipt.write_text(json.dumps(rows,indent=2))
 if r.returncode:raise RuntimeError((scene,r.returncode))
