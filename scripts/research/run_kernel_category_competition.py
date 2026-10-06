"""Run all eight frozen ScanNet development scenes with one source recipe."""
import argparse,json,os,queue,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


p=argparse.ArgumentParser();p.add_argument('--positive-contract',choices=['any','all'],default='any');p.add_argument('--source',default='output/pipeline_kernel_extent_next40');p.add_argument('--output');p.add_argument('--gpus',default='0,1,4,5');a=p.parse_args()
jobs=queue.Queue();base=Path(a.output or ('output/competitive_kernel_extent_next40' if a.positive_contract=='any' else 'output/all_positive_kernel_extent_next40'));base.mkdir(parents=True,exist_ok=True)
for path in sorted(Path(a.source).glob('scannet_*/field.pt')):jobs.put(path.parent.name)
def run(gpu):
    results=[]
    while True:
        try:key=jobs.get_nowait()
        except queue.Empty:break
        out=base/key;out.mkdir(parents=True,exist_ok=True)
        commands=[['scripts/research/fit_kernel_category_competition.py','--key',key,'--positive-contract',a.positive_contract,'--source',a.source,'--output',str(base)],['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline']]
        with (out/'execution.log').open('w') as log:
            for command in commands:
                rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python']+command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT).returncode
                if rc:break
        row=dict(key=key,exit_code=rc);results.append(row);print(row,flush=True)
    return results
with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:results=sum(pool.map(run,a.gpus.split(',')),[])
(base/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
