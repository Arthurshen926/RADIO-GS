"""Copied field, fresh process, complete preencoded menu and denied source paths."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from radio_gs.query_field.runtime import QueryFieldRuntime
from scripts.research.run_query_field_semantic_measure import ROOT,read,write,receipt,design


def worker(field,fixture,output):
    forbidden=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding'];attempts=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.realpath(os.fsdecode(args[0]))
        if any(path==r or path.startswith(r+os.sep) for r in forbidden):
            attempts.append(path);raise PermissionError('source path denied '+path)
    sys.addaudithook(audit)
    runtime=QueryFieldRuntime.load(field,'cuda');queries=torch.load(fixture,weights_only=True).cuda()
    scores,labels=runtime.category_partition(queries)
    torch.save(dict(scores=scores.cpu(),labels=labels.cpu(),forbidden_open_attempts=attempts),output)


def main():
    d=design();read(ROOT/'scoring_complete.json');j=d['jobs'][0]
    root=ROOT/'validation/runtime';root.mkdir(parents=True,exist_ok=False)
    data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    torch.save(data['vectors'],root/'fixture.pt')
    with tempfile.TemporaryDirectory(prefix='radio_gs_semantic_measure_') as t:
        t=Path(t);shutil.copytree(ROOT/'fits'/j['name']/'field',t/'field')
        shutil.copyfile(root/'fixture.pt',t/'fixture.pt')
        with (root/'worker.log').open('x') as f:
            subprocess.run([sys.executable,'-m','scripts.research.validate_query_field_semantic_measure','worker',
                '--field',str(t/'field'),'--fixture',str(t/'fixture.pt'),'--output',str(t/'result.pt')],stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copyfile(t/'result.pt',root/'copied.pt')
    actual=torch.load(root/'copied.pt',weights_only=True);expected=data['scores']['reliable_typed']
    labels=expected.argmax(1).masked_fill(~data['available'],-1)
    error=float((actual['scores']-expected).abs().max());exact=torch.equal(actual['scores'],expected)
    selected=torch.equal(actual['labels'],labels)
    result=dict(complete=True,scene=j['name'],menu=len(data['queries']),strict_scores=exact,
        exact_partition=selected,maximum_error=error,source_denial_passed=not actual['forbidden_open_attempts'],
        forbidden_open_attempts=actual['forbidden_open_attempts'],numerical_passed=error<=1e-6 and selected,
        coverage='one scene full preencoded menu; shared runtime category_partition;not raw encoder or OS sandbox',
        evidence=[receipt(root/p) for p in ['fixture.pt','copied.pt','worker.log']]+[receipt(Path(__file__))])
    write(root/'report.json',result);print(result,flush=True)
    if not result['numerical_passed'] or not result['source_denial_passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker']);p.add_argument('--field');p.add_argument('--fixture');p.add_argument('--output');a=p.parse_args()
    torch.set_num_threads(4)
    worker(a.field,a.fixture,a.output) if a.stage=='worker' else main()
