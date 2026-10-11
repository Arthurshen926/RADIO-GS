"""Cold four-first-query compatibility after explicit-cutoff warmup repair."""
import argparse
import subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute

ROOT=Path('output/query_field_20261012/text_extent_public_v2')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    old=read('output/query_field_20261012/text_extent_public_v1/design.json')
    write(ROOT/'design.json',dict(jobs=old['jobs'],scope_rule='atomic',
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[
            receipt(__file__),receipt('scripts/research/validate_query_field_extent_runtime.py')],
        query_selection='unchanged first query in each of four scenes',
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        target_labels_used=False,all_queries_validated=False,default_promoted=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']:check(r)
    return d


def worker(index):
    import scripts.research.validate_query_field_extent_runtime as original
    original.ROOT=ROOT
    original.worker(index)


def run(gpu):
    d=design();probes=[]
    for i,j in enumerate(d['jobs']):
        execute(['-m','scripts.research.validate_query_field_extent_runtime_v2','worker','--index',str(i)],ROOT/'logs'/(j['scene']+'.log'),gpu)
        r=read(ROOT/'probes'/j['scene']/'result.json');check(r['output']);check(j['reference'])
        actual=torch.load(r['output']['path'],weights_only=True);ref=torch.load(j['reference']['path'],weights_only=True)
        for key,source in [('membership','membership_by_control'),('selected','selected_by_control')]:
            assert torch.equal(actual[key],ref[source]['native_atomic_abstention'][:,0]),(j['scene'],key)
        probes.append(dict(scene=j['scene'],query=j['text'],member_values_exact=True,selected_bits_exact=True,
            record=receipt(ROOT/'probes'/j['scene']/'result.json')))
        print('cold text compatibility PASS',j['scene'],flush=True);del actual,ref
    result=dict(passed=True,probes=probes,design=receipt(ROOT/'design.json'),target_labels_used=False,
        all_queries_validated=False,whole_five_task=False,default_promoted=False,
        previous_full68_strict_float_gate_repaired=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-text-extent-public-v2-audit.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run']);p.add_argument('--index',type=int);p.add_argument('--gpu',type=int,default=5)
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.index)
    else:run(a.gpu)
