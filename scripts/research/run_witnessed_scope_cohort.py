"""Frozen identity, same observations, third-view scope: majority versus union."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/witnessed_scope_study'))
    p.add_argument('--gpus',default='0,1,4,5')
    p.add_argument('--build-only',action='store_true')
    p.add_argument('--mode',choices=['majority','union'])
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    gpus=args.gpus.split(',')
    def build(lane):
        for job in jobs[lane::len(gpus)]:
            destination=root/'maps'/job['name'];destination.mkdir(parents=True,exist_ok=True)
            if (destination/'map.json').exists():continue
            with (destination/'compile.log').open('w') as log:
                subprocess.run([PY,'-u','-m','radio_gs.pipeline.witnessed_scope_map',
                    '--scene',job['scene'],'--output',str(destination)],
                    env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(build,range(len(gpus))))
    if args.build_only:return
    run=root/args.mode;run.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        source=root/'maps'/job['name'];destination=run/'maps'/job['name']
        destination.mkdir(parents=True,exist_ok=True)
        spec=json.loads((source/'map.json').read_text())
        for name in spec['map_files']:
            if not (destination/name).exists():os.link(source/name,destination/name)
        spec['scope_witness_rule']=args.mode
        (destination/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
        job['scene']=str(destination/'map.json')
        job['selection_contract']='third-view witnessed scope '+args.mode+'; fixed before evaluation'
    recipe.update(default=False,name='Native identity and witnessed scope '+args.mode,
        validation=dict(scope='all opened-development UQIS queries; public providers unchanged'))
    path=run/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (run/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),
            '--output',str(run/'system'),'--gpus',args.gpus,'--benchmarks','uqis'],
            env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
