"""Sequential reproducible Stage1; outputs stay beneath one supplied new run."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new run; preserve existing outputs')
    out.mkdir(parents=True);commands=[]
    # Recipe is mandatory, supplied from the reviewed fixed pilot, no target fit.
    template=Path('output/query_region_evidence_next42/run_20260928_stage1/recipe.json')
    (out/'recipe.json').write_bytes(template.read_bytes())
    for stage,gpu in [('build',1),('fit',0),('evaluate',0)]:
        argv=[PYTHON,f'scripts/research/{stage}_query_region_evidence_next42.py','--output',str(out)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
        start=time.monotonic()
        with (out/(stage+'.log')).open('w') as log:rc=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        commands.append(dict(stage=stage,argv=argv,seconds=time.monotonic()-start,exit_code=rc))
        (out/'stages.json').write_text(json.dumps(commands,indent=2))
        if rc:raise RuntimeError(stage+' failed; inspect log')
if __name__=='__main__':main()
