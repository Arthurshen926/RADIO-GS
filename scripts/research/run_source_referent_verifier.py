"""Run every source referent verification across available GPU lanes."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',default='output/source_referent_verification')
p.add_argument('--gpus',type=int,nargs='+',default=list(range(6)))
p.add_argument('--python',default='/root/.cache/radio_gs/qwen3_env/bin/python')
p.add_argument('--model-id',default='Qwen/Qwen3-VL-8B-Instruct')
p.add_argument('--batch-size',type=int,default=4);p.add_argument('--context-image',action='store_true');a=p.parse_args();root=Path(a.root)


def worker(job):
    shard,gpu=job
    cmd=[a.python,'scripts/research/verify_source_referents.py','--manifest',str(root/'manifest.json'),
         '--output',str(root/'judgments'),'--shard',str(shard),'--shards',str(len(a.gpus)),
         '--model-id',a.model_id,'--attention','sdpa','--batch-size',str(a.batch_size)]
    if a.context_image:cmd.append('--context-image')
    (root/f'verifier_command_{shard}.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (root/f'verifier_{shard}.log').open('w') as log:
        subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('shard',shard,'complete',flush=True)


with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,enumerate(a.gpus)))
