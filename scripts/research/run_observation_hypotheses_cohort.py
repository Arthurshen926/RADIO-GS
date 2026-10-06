"""All UQIS queries under one fixed, label-free scope-hypothesis mechanism.

The recipe retains all public jobs and their unchanged declared inputs. This
experiment executes only the nine changed registered-map jobs; it does not
represent retained public results as freshly re-evaluated method improvements.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from radio_gs.pipeline.observation_hypotheses import compile_map

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/observation_hypotheses_study'))
    p.add_argument('--gpus',default='4,5');p.add_argument('--tracks',action='store_true')
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path('configs/pipeline/system_v7.json').read_text())
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    def build(job):
        output=root/'maps'/job['name']
        if not (output/'build.json').exists():compile_map(Path(job['scene']),output,a.tracks)
        job['scene']=str(output/'map.json')
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:list(pool.map(build,jobs))
    recipe.update(default=False,name='Native identity over physical scope hypotheses',
        validation={'status':'fixed UQIS mechanism candidate; unchanged public task definitions'},
        comparison='all original observations, all scope hypotheses; fixed physical matching; no new teacher or source data')
    if a.tracks:
        recipe.update(name='View-exclusive physical instance tracks for all four modalities',
            comparison='same observations; mutual correspondence tracks; no source, teacher or target supervision added')
    path=root/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (root/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),'--output',str(root/'system'),
            '--gpus',a.gpus,'--benchmarks','uqis'],env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All nine maps and 268 queries completed before evaluation.',flush=True)


if __name__=='__main__':main()
