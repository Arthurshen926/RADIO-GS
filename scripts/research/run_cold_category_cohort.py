import concurrent.futures,json,os,subprocess
from pathlib import Path
rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes'];py='/root/miniconda3/envs/cybersim_agent/bin/python'
def worker(gpu):
 for r in rows[gpu::6]:
  out=Path('output/cold_category')/r['scene'];out.mkdir(parents=True,exist_ok=True)
  with (out/'run.log').open('w') as log:
   subprocess.run([py,'scripts/research/run_cold_category.py','--scene',r['scene'],'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   for arm in ['categorical','source_exclusive']:
    subprocess.run([py,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/arm/'typed_readout.pt'),'--evaluation',r['evaluation_config'],'--output',str(out/arm/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
  print(r['scene'],'complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
