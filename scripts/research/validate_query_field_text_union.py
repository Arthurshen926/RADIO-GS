"""Full one-scene text union replay from a copied export with denied sources."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from radio_gs.query_field.runtime import QueryFieldRuntime
from scripts.research.run_query_field_text_union import ROOT,design,read,write,receipt


def worker(field,fixture,output):
    forbidden=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding'];attempts=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.realpath(os.fsdecode(args[0]))
        if any(path==r or path.startswith(r+os.sep) for r in forbidden):
            attempts.append(path);raise PermissionError('source path denied '+path)
    sys.addaudithook(audit)
    try:open(Path('output')/'text_union_forbidden_probe','rb')
    except PermissionError:probe=True
    else:raise RuntimeError('denial probe failed')
    attempts.clear();runtime=QueryFieldRuntime.load(field,'cuda');data=torch.load(fixture,weights_only=True)
    results=[runtime.text_union(q.cuda(),data['negative'].cuda()) for q in data['queries']]
    torch.save(dict(scores=torch.stack([r.membership_scores.cpu() for r in results],1),
        selected=torch.stack([r.selected.cpu() for r in results],1),forbidden_open_attempts=attempts,guard_probe_blocked=probe),output)


def main():
    d=design();read(ROOT/'scoring_complete.json');j=d['jobs'][0]
    root=ROOT/'validation/runtime';root.mkdir(parents=True,exist_ok=False)
    data=torch.load(ROOT/'predictions'/(j['name']+'__measure')/'predictions.pt',weights_only=True)
    torch.save(dict(queries=data['native_query_vectors'],negative=data['generic_negative_vectors']),root/'fixture.pt')
    with tempfile.TemporaryDirectory(prefix='radio_gs_text_union_') as t:
        t=Path(t);shutil.copytree(ROOT/'fits'/j['name']/'field',t/'field');shutil.copyfile(root/'fixture.pt',t/'fixture.pt')
        with (root/'worker.log').open('x') as f:
            subprocess.run([sys.executable,'-m','scripts.research.validate_query_field_text_union','worker','--field',str(t/'field'),
                '--fixture',str(t/'fixture.pt'),'--output',str(t/'result.pt')],stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copyfile(t/'result.pt',root/'copied.pt')
    actual=torch.load(root/'copied.pt',weights_only=True);expected=data['scores']['measure_typed']
    error=float((actual['scores']-expected).abs().max());exact=torch.equal(actual['scores'],expected)
    selected=torch.equal(actual['selected'],data['selected_by_control']['measure_typed'])
    result=dict(complete=True,scene=j['name'],queries=len(data['texts']),strict_scores=exact,exact_selection=selected,
        maximum_error=error,source_denial_passed=actual['guard_probe_blocked'] and not actual['forbidden_open_attempts'],
        forbidden_open_attempts=actual['forbidden_open_attempts'],guard_probe_blocked=actual['guard_probe_blocked'],
        numerical_passed=error<=1e-6 and selected,coverage='all preencoded requests in one scene;not OS sandbox/raw encoder or all-scene proof',
        evidence=[receipt(root/p) for p in ['fixture.pt','copied.pt','worker.log']]+[receipt(Path(__file__))])
    write(root/'report.json',result);print(result,flush=True)
    if not result['numerical_passed'] or not result['source_denial_passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker']);p.add_argument('--field');p.add_argument('--fixture');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(4)
    worker(a.field,a.fixture,a.output) if a.stage=='worker' else main()
