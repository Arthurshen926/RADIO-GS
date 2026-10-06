import concurrent.futures,json,os,subprocess
from pathlib import Path
rows=json.loads(Path('configs/consolidated/scannet/cohort.json').read_text())['scenes'];py='/root/miniconda3/envs/cybersim_agent/bin/python'
def worker(gpu):
 for r in rows[gpu::6]:
  out=Path('output/cold_category')/r['scene']/'exclusive_scope';out.mkdir(exist_ok=True)
  with (out/'run.log').open('w') as log:
   subprocess.run([py,'scripts/research/run_exclusive_category_scope.py','--scene',r['scene'],'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([py,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'typed_readout.pt'),'--evaluation',r['evaluation_config'],'--output',str(out/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
  print(r['scene'],'complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
