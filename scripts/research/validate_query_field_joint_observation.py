"""Cold field-only replay; keep numerical and strict bitwise claims separate."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from scripts.research.run_query_field_joint_observation import ROOT,ARMS,read,write,receipt,check
from scripts.research.validate_query_field_legacy_replay import capture
from scripts.research.validate_query_field_mixture_replay import differences


def child(args):
    forbidden=[Path(p).resolve() for p in args.deny];attempts=[]
    def guard(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes)):return
        path=Path(os.fsdecode(values[0])).resolve()
        if any(path==p or p in path.parents for p in forbidden):
            attempts.append(str(path));raise PermissionError('source path denied: '+str(path))
    sys.addaudithook(guard)
    fixture=torch.load(args.fixture,map_location='cpu',weights_only=True)
    torch.save(capture(args.field,fixture),args.output)
    print(json.dumps(dict(forbidden_open_attempts=attempts)))


def main():
    read(ROOT/'scoring_complete.json')
    out=ROOT/'validation/runtime';out.mkdir(parents=True,exist_ok=False)
    prior=read('output/query_field_20261008/protected_relations_v1/validation/runtime/report.json')
    check(prior['fixture']);fixture=torch.load(prior['fixture']['path'],map_location='cpu',weights_only=True)
    repo=Path.cwd().resolve();forbidden=[repo/'output',repo/'data',Path('/root/RADIO-GS/output'),Path('/root/RADIO-GS/data')]
    records=[]
    for arm in ARMS:
        directory=ROOT/'fits/scene0030_00'/arm/'field'
        expected=capture(directory,fixture);torch.save(expected,out/(arm+'_original.pt'))
        gc.collect();torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='joint-observation-cold-') as temporary:
            isolated=Path(temporary);shutil.copytree(directory,isolated/'field')
            shutil.copy2(prior['fixture']['path'],isolated/'query.pt')
            cmd=[sys.executable,'-m','scripts.research.validate_query_field_joint_observation','--child',
                '--field',str(isolated/'field'),'--fixture',str(isolated/'query.pt'),'--output',str(isolated/'result.pt')]
            for path in forbidden:cmd.extend(['--deny',str(path.resolve())])
            process=subprocess.run(cmd,cwd=isolated,env=dict(os.environ,PYTHONPATH=str(repo)),capture_output=True,text=True,check=True)
            log=json.loads(process.stdout.strip().splitlines()[-1]);actual=torch.load(isolated/'result.pt',map_location='cpu',weights_only=True)
            shutil.copy2(isolated/'result.pt',out/(arm+'_copied.pt'))
        comparisons=differences(expected,actual)
        records.append(dict(arm=arm,comparison=comparisons,field=receipt(directory/'field.pt'),**log))
        write(out/(arm+'.json'),records[-1]);print('cold',arm,comparisons,flush=True)
    strict=all(q['exact_scores'] and q['exact_selections'] for r in records for q in r['comparison'].values())
    numeric=all(q['maximum_absolute_error']<=1e-6 and q['exact_selections'] for r in records for q in r['comparison'].values())
    denied=not any(r['forbidden_open_attempts'] for r in records)
    write(out/'report.json',dict(complete=True,records=records,strict_bitwise_passed=strict,
        numerical_and_selection_passed=numeric,source_denial_passed=denied,tolerance=1e-6,
        coverage='one real scene per arm;4 input modalities plus category menu;one fresh copied process per arm',
        enforcement='Python file-open audit;not OS sandbox',fixture=prior['fixture'],
        preserves_original_operation_sequence=True,model_promotion=False,
        code=[receipt(Path(__file__)),receipt(Path('scripts/research/validate_query_field_legacy_replay.py'))],
        evidence=[receipt(p) for p in out.glob('*.pt')]))
    if not numeric or not denied:raise ValueError('cold numerical/selection or source denial check failed; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--field');p.add_argument('--fixture');p.add_argument('--output')
    p.add_argument('--deny',action='append',default=[]);args=p.parse_args();torch.set_num_threads(4)
    child(args) if args.child else main()
