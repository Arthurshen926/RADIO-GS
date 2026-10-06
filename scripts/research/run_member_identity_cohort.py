"""One fixed identity-corroboration candidate on the shared full system."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/member_identity_study'))
    p.add_argument('--gpus',default='2,3,4,5');a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path('configs/pipeline/system_v7.json').read_text());gpus=a.gpus.split(',')
    jobs=[j for j in recipe['jobs'] if j['benchmark'] in ['uqis','scannet']]
    def build(lane):
        for job in jobs[lane::len(gpus)]:
            output=root/'maps'/job['benchmark']/job['name'];output.mkdir(parents=True,exist_ok=True)
            if not (output/'build.json').exists():
                command=[PY,'-u','-m','radio_gs.pipeline.member_identity','--scene',job['scene'],'--output',str(output)]
                if job['benchmark']=='scannet':
                    field=torch.load(Path('output/observation_reliability')/job['name']/'observation_field.pt',map_location='cpu',weights_only=False)
                    offsets=output/'region_offsets.npy';np.save(offsets,field['region_offsets'].numpy());del field
                    command.extend(['--offsets',str(offsets)])
                with (output/'compile.log').open('w') as log:
                    subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['benchmark'],job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(build,range(len(gpus))))
    for job in recipe['jobs']:
        base=root/'maps'/job['benchmark']/job['name']
        if job['benchmark']=='uqis':job['scene']=str(base/'map.json')
        elif job['benchmark']=='scannet':job['scene']=str(base/'scene.json')
    recipe.update(default=False,name='Actual-member native identity corroboration',
        validation={'status':'fixed full-system candidate; no fitted labels'},
        comparison='v7 maps and teachers; emitting-view excluded member identity; masks and transfer unchanged')
    # Interleave the new instance task with public controls; no subset is scored
    # until this complete cohort finishes all predictions.
    recipe['jobs'].sort(key=lambda j:({'uqis':0,'scannet':1,'nvos':2,'lerf':3}[j['benchmark']],j['name']))
    path=root/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (root/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),'--output',str(root/'system'),'--gpus',a.gpus],
            env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Complete common-system identity comparison finished.',flush=True)


if __name__=='__main__':main()
