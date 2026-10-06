"""Transfer the SAME identity policies to positive-reference NVOS inference."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

ROOT=Path('output/reference_referent_verification')
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='+',default=list(range(6)));a=p.parse_args()
manifest=json.loads((ROOT/'manifest.json').read_text())
reports=[json.loads(p.read_text()) for p in sorted((ROOT/'judgments').glob('shard_*.json'))]
decisions={r['key']:r for report in reports for r in report['results']}
counts={}
for arm in ['veto','affirmative']:
    for scene,path in manifest['tracks'].items():
        out=ROOT/arm/scene;out.mkdir(parents=True,exist_ok=True)
        tracks=torch.load(path,map_location='cpu',weights_only=False);removed=[];kept=[]
        for job in manifest['jobs']:
            if job['scene']!=scene:continue
            r=decisions[job['key']];keep=not r['balanced_veto'] if arm=='veto' else r['affirmative_admission']
            if not keep:
                # The entire frame supplies no label, including no complement
                # negatives. The authorized reference is never judged or removed.
                tracks['frames'][job['frame']]['object_logit']=-1.
                removed.append(job['frame'])
            else:kept.append(job['frame'])
        torch.save(tracks,out/'tracks.pt');counts[f'{arm}:{scene}']=dict(kept=kept,unknown=removed)
(ROOT/'verification.json').write_text(json.dumps(dict(counts=counts,candidates=len(decisions),
    image_forwards=sum(r['image_forwards'] for r in reports),seconds=sum(r['seconds'] for r in reports),
    reference_anchor_retained=True,unknown_frame_not_background=True),indent=2)+'\n')


def worker(gpu,arm,scene):
    out=ROOT/arm/scene
    commands=[
        [PY,'scripts/research/run_reference_query.py','--scene',f'configs/consolidated/nvos/{scene}.scene.json',
         '--query',f'configs/consolidated/nvos/{scene}.query.json','--tracks',str(out/'tracks.pt'),
         '--output',str(out/'runtime'),'--gpu','0'],
        [PY,'scripts/research/evaluate_reference_instance.py','--scene',f'configs/consolidated/nvos/{scene}.scene.json',
         '--evaluation',f'configs/consolidated/nvos/{scene}.evaluation.json',
         '--field',str(out/'runtime/typed_readout.pt'),'--typed-readout','--output',str(out/'evaluation')]]
    # run_reference_query's explicit --gpu must use the physical lane because it
    # constructs the subprocess CUDA environment itself.
    commands[0][-1]=str(gpu)
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/'run.log').open('w') as log:
        for cmd in commands:subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(arm,scene,'complete',flush=True)


jobs=[(arm,scene) for arm in ['veto','affirmative'] for scene in manifest['tracks']]
lanes=[[(gpu,*job) for job in jobs[i::len(a.gpus)]] for i,gpu in enumerate(a.gpus)]
def lane(jobs):
    for job in jobs:worker(*job)
with ThreadPoolExecutor(max_workers=len(lanes)) as pool:list(pool.map(lane,lanes))
