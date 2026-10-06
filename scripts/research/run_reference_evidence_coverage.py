"""Fixed eight-query evidence diagnostic on six GPUs."""
import os,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
scenes=['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']
def worker(gpu):
    for s in scenes[gpu::6]:
        dest=Path('output/reference_evidence_coverage')/s;dest.mkdir(parents=True,exist_ok=True)
        with (dest/'run.log').open('w') as log:subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','scripts/research/diagnose_reference_evidence_coverage.py','--scene',s,'--output',str(dest)],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(s,'done',flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(worker,range(6)))
