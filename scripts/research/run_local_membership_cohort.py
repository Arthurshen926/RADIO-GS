"""Build both fixed local-membership controls, then predict before label scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/local_membership_study'))
    p.add_argument('--gpus',default='2,3,4,5');a=p.parse_args()
    recipe=json.loads(Path('configs/pipeline/system_v7.json').read_text())
    root=a.output;root.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',')
    jobs=[j for j in recipe['jobs'] if j['benchmark'] in ['uqis','scannet']]
    def build(lane):
        for job in jobs[lane::len(gpus)]:
            out=root/'maps'/job['benchmark']/job['name'];out.mkdir(parents=True,exist_ok=True)
            marker=out/'build.json'
            if not marker.exists():
                command=[PY,'-u','-m','radio_gs.pipeline.local_membership_map','--scene',job['scene'],'--output',str(out)]
                if job['benchmark']=='scannet':
                    provenance=Path('output/observation_reliability')/job['name']/'observation_field.pt'
                    field=torch.load(provenance,map_location='cpu',weights_only=False)
                    offsets=out/'region_offsets.npy';np.save(offsets,field['region_offsets'].numpy());del field
                    command.extend(['--offsets',str(offsets)])
                with (out/'compile.log').open('w') as log:
                    subprocess.run(command,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['benchmark'],job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(build,range(len(gpus))))
    for mode in ['transfer','scope']:
        candidate=copy.deepcopy(recipe);candidate['default']=False
        candidate['name']='Fixed local native membership '+mode
        candidate['validation']={'status':'candidate, same evidence, no fitted labels'}
        candidate['jobs']=[j for j in candidate['jobs'] if mode=='scope' or j['benchmark']=='uqis']
        for job in candidate['jobs']:
            base=root/'maps'/job['benchmark']/job['name']
            if job['benchmark']=='uqis':job['scene']=str(base/mode/'map.json')
            elif job['benchmark']=='scannet':job['scene']=str(base/'scene.json')
        path=root/f'{mode}.recipe.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
        with (root/f'{mode}.run.log').open('w') as log:
            subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),
                '--output',str(root/mode),'--gpus',a.gpus],
                env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(mode,'complete',flush=True)


if __name__=='__main__':main()
