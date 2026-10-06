"""Build a query-free object scope prior and evaluate all independent queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'
SAM='/root/miniconda3/envs/sam3/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/entity_scope_study'))
    p.add_argument('--gpus',default='0,1,4,5');p.add_argument('--build-only',action='store_true')
    p.add_argument('--provider',choices=['sam3','panoptic'],default='sam3')
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis'];gpus=args.gpus.split(',')
    def build(lane):
        for job in jobs[lane::len(gpus)]:
            destination=root/'maps'/job['name'];destination.mkdir(parents=True,exist_ok=True)
            if (destination/'map.json').exists():continue
            with (destination/'compile.log').open('w') as log:
                subprocess.run([SAM if args.provider=='sam3' else PY,'-u','-m','radio_gs.pipeline.entity_scope_map',
                    '--scene',job['scene'],'--mapping','output/scannet_uqis_9_v0_3_1_candidate/mapping/scenes.json',
                    '--output',str(destination),'--provider',args.provider],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(build,range(len(gpus))))
    if args.build_only:return
    for job in jobs:
        job['scene']=str(root/'maps'/job['name']/'map.json')
        job['selection_contract']='offline entity scope, original native identity; two-view geometric support else freshly computed v8 scope'
    recipe.update(default=False,name='Native identity and entity scope prior: '+args.provider,
        validation=dict(scope='all opened-development UQIS queries; public providers unchanged'))
    path=root/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (root/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),
            '--output',str(root/'system'),'--gpus',args.gpus,'--benchmarks','uqis'],
            env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
