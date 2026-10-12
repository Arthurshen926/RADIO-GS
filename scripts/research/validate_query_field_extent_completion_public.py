"""Validate all completion queries through the public API with separate strict gates.

Separate from the four cold smoke probes; no new target scoring is performed.
Workers receive just a scene field and current legal vector batches. References
are opened only by the controller after the workers seal their predictions.
"""
import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from radio_gs.query_field.runtime import QueryFieldRuntime
from scripts.research.run_query_field_scope_completion import ROOT as PARENT
from scripts.research.run_query_field_scope_completion import CODE as METHOD_CODE
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261012/text_extent_completion_public_v1')
CODE=[__file__,'radio_gs/query_field/runtime.py','radio_gs/query_field/extent_runtime.py',*METHOD_CODE[1:]]


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for j in read(PARENT/'prediction_config.json')['jobs']:
        p=PARENT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        data=torch.load(p/'predictions.pt',weights_only=True)
        fixture=ROOT/'queries'/f"{j['scene']}.pt";fixture.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(native_queries=data['native_query_vectors'],generic_negatives=data['generic_negative_vectors']),fixture)
        jobs.append(dict(scene=j['scene'],field=j['field'],query=receipt(fixture),
            reference=receipt(p/'predictions.pt'),texts=data['texts'],
            manifest=receipt(Path(j['field'])/'manifest.json')))
        del data
    write(ROOT/'design.json',dict(jobs=jobs,code=[receipt(p) for p in CODE],protocol=receipt('docs/experiments/2026-10-12-text-extent-completion-public-protocol.md'),encoder_audit=receipt('docs/experiments/2026-10-12-text-extent-encoder-audit.json'),scope_rule='other_view_positive',queries=68,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        target_labels_used=False,raw_encoder_revalidated=False,default_promoted=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol'],d['encoder_audit']]:check(r)
    return d


def worker(index):
    d=design();j=d['jobs'][index];check(j['query']);check(j['manifest'])
    packet=torch.load(j['query']['path'],weights_only=True)
    assert set(packet)=={'native_queries','generic_negatives'}
    runtime=QueryFieldRuntime.load(j['field']);runtime.prepare_text_extent(scope_device='cuda')
    values=[];rows=[]
    for qi,q in enumerate(packet['native_queries']):
        with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
            result=runtime.text_extent(q,packet['generic_negatives'],scope_rule=d['scope_rule'])
        values.append(result.membership_scores);rows.append(result.diagnostics)
        print(j['scene'],qi,'fallback',result.diagnostics['fallback'],flush=True)
    out=ROOT/'predictions'/j['scene'];out.mkdir(parents=True,exist_ok=False)
    member=torch.stack(values,1);torch.save(dict(membership=member,selected=member>.5),out/'result.pt')
    write(out/'result.json',dict(scene=j['scene'],query=j['query'],field=j['field'],diagnostics=rows,
        output=receipt(out/'result.pt'),cold_load=True,query_file_io_denied=True,queries=len(values),
        worker_loaded_reference_predictions=False,os_sandbox=False))


def run(gpus):
    d=design()
    lanes(list(range(4)),gpus,lambda i,g:execute(['-m','scripts.research.validate_query_field_extent_completion_public','worker','--index',str(i)],ROOT/'logs'/(d['jobs'][i]['scene']+'.log'),g))
    records=[receipt(ROOT/'predictions'/j['scene']/'result.json') for j in d['jobs']]
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,scored=False))
    probes=[]
    for j in d['jobs']:
        r=read(ROOT/'predictions'/j['scene']/'result.json');check(r['output']);check(j['reference'])
        actual=torch.load(r['output']['path'],weights_only=True);ref=torch.load(j['reference']['path'],weights_only=True)
        target=ref['membership_by_control']['other_view_positive_completion']
        assert torch.equal(actual['selected'],actual['membership']>.5)
        delta=(actual['membership']-target).abs();different=delta!=0
        probes.append(dict(scene=j['scene'],queries=r['queries'],member_values_exact=not bool(different.any()),
            selected_bit_differences=int((actual['selected']!=(target>.5)).sum()),max_absolute_difference=float(delta.max()),
            differing_queries=[dict(index=qi,text=j['texts'][qi],fallback=r['diagnostics'][qi]['fallback'],
                differing_values=int(different[:,qi].sum()),max_absolute_difference=float(delta[:,qi].max()))
                for qi in different.any(0).nonzero().flatten().tolist()],
            record=receipt(ROOT/'predictions'/j['scene']/'result.json')))
        print(j['scene'],probes[-1],flush=True);del actual,ref,target,delta,different
    assert sum(p['queries'] for p in probes)==68
    result=dict(complete=True,actual_set_gate_passed=all(p['selected_bit_differences']==0 for p in probes),
        strict_float_gate_passed=all(p['member_values_exact'] for p in probes),probes=probes,
        design=receipt(ROOT/'design.json'),prediction_seal=receipt(ROOT/'predictions_complete.json'),
        target_labels_used=False,all_68_queries_validated=True,raw_encoder_revalidated=False,
        previous_atomic_float_failure_repaired=False,whole_five_task=False,default_promoted=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-text-extent-completion-public-audit.json',result)
    assert result['actual_set_gate_passed'],'actual selected set mismatch; retain all records'
    print('Actual set gate PASS; strict float gate '+('PASS' if result['strict_float_gate_passed'] else 'FAIL'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='3,4')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.index)
    else:run(list(map(int,a.gpus.split(','))))
