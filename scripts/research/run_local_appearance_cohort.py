"""Same source observations: pooled identity versus native patch correspondence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/local_appearance_study'))
    p.add_argument('--gpus',default='4,5');p.add_argument('--mode',choices=['pooled','chamfer','reciprocal'])
    p.add_argument('--build-only',action='store_true');a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',')
    recipe=json.loads(Path('configs/pipeline/system_v7.json').read_text())
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    def compile(lane):
        for job in jobs[lane::len(gpus)]:
            out=root/'maps'/job['name'];out.mkdir(parents=True,exist_ok=True)
            if (out/'map.json').exists():continue
            with (out/'compile.log').open('w') as log:
                subprocess.run([PY,'-u','-m','radio_gs.pipeline.local_appearance_map','--scene',job['scene'],
                    '--mapping','output/scannet_uqis_9_v0_3_1_candidate/mapping/scenes.json','--output',str(out)],
                    env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('compiled',job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(compile,range(len(gpus))))
    if a.build_only:return
    mode=a.mode;run=root/mode;run.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        source=root/'maps'/job['name'];out=run/'maps'/job['name'];out.mkdir(parents=True,exist_ok=True)
        spec=json.loads((source/'map.json').read_text())
        for name in spec['map_files']:
            if not (out/name).exists():os.link(source/name,out/name)
        if mode!='pooled':spec['image_local_match']=mode
        (out/'map.json').write_text(json.dumps(spec,indent=2)+'\n');job['scene']=str(out/'map.json')
    recipe.update(default=False,name='Native local appearance identity: '+mode,
        comparison='same legal source frames, masks, geometry and native teachers; image identity changes only',
        validation=dict(scope='full opened-development UQIS cohort; public providers retained'))
    path=run/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (run/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),'--output',str(run/'system'),
            '--gpus',a.gpus,'--benchmarks','uqis'],env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
