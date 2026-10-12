"""Candidate support must survive into finite member proposal decisions."""
import argparse
import copy
import subprocess
from pathlib import Path
import numpy as np
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261012/reference_proposals_v1')
PREVIOUS=Path('output/query_field_20261012/reference_edit_v1')
ARMS=['free','scope','matched','prioronly']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for old in read(PREVIOUS/'design.json')['jobs']:
        check(old['configuration']);cfg=copy.deepcopy(read(old['configuration']['path']));base=cfg['jobs'][0]
        cfg['jobs']=[dict(base,name=old['name']+'__'+a,query_program='reference_extent_'+('edit' if a=='free' else a)) for a in ARMS]
        cfg['prediction_policy']='same edit objective/input; scoped proposals vs matched and prior-only capacity controls'
        path=ROOT/'configs'/(old['name']+'.json');write(path,cfg)
        check(old['scoring']);scoring=read(old['scoring']['path']);scoring['prediction_config']=str(path)
        sp=ROOT/'configs'/(old['name']+'_score.json');write(sp,scoring)
        jobs.append(dict(old,configuration=receipt(path),scoring=receipt(sp)))
    write(ROOT/'design.json',dict(jobs=jobs,arms=ARMS,primary='scope',
        protocol=receipt('docs/experiments/2026-10-12-reference-proposals-protocol.md'),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),receipt('scripts/research/run_query_field_relations.py')],
        previous_summary=receipt(PREVIOUS/'summary.json'),previous_audit=receipt(PREVIOUS/'audit.json'),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_frames=120,new_field_state_bytes=0,training_steps=0,raw_scribble=False,whole_five_task=False,
        development_on_previously_evaluated_cases=True,proposal_shuffle_seed=20261012,observation_weight={a:.5 for a in ARMS},regularizers={a:'edit' for a in ARMS},proposal_policies={'free':None,'scope':'scope','matched':'shuffled','prioronly':'prior_only'}))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol'],d['previous_summary'],d['previous_audit']]:check(r)
    return d


def run(gpus):
    d=design()
    def predict(j,g):
        for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
        for job in read(j['configuration']['path'])['jobs']:
            execute(['-m','radio_gs.query_field.public_nvos','predict','--config',j['configuration']['path'],
                '--job',job['name'],'--output',str(ROOT/'predictions'/j['name'])],ROOT/'logs'/(job['name']+'.log'),g)
        print('predicted',j['name'],flush=True)
    lanes(d['jobs'],gpus,predict);records=[];baseline=[]
    for j in d['jobs']:
        cfg=read(j['configuration']['path']);states={}
        for arm,job in zip(ARMS,cfg['jobs']):
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['extent_state'])
            records.append(receipt(p/'prediction.json'));states[arm]=torch.load(r['extent_state']['path'],weights_only=True)
        old_dir=PREVIOUS/'predictions'/j['name']/(j['name']+'__edit');old_record=read(old_dir/'prediction.json')
        check(dict(path=str(old_dir/'predictions.pt'),sha256=old_record['predictions_sha256']))
        old=torch.load(old_dir/'predictions.pt',weights_only=True)
        new=torch.load(ROOT/'predictions'/j['name']/(j['name']+'__free')/'predictions.pt',weights_only=True)
        errors={k:float((old[k]-new[k]).abs().max()) for k in ['membership','response','retained_response']}
        exact=torch.equal(old['selected'],new['selected']);baseline.append(dict(case=j['name'],selected_exact=exact,max_errors=errors))
        assert exact and max(errors.values())<=1e-6,(j['name'],baseline[-1])
        for arm in ['scope','matched','prioronly']:
            for k in ['initial','prior','cost','reference','domain']:
                assert torch.equal(states[arm][k],states['free'][k]),(j['name'],arm,k)
        initial=states['free']['initial']>.5
        assert torch.equal(states['prioronly']['proposal_domain'],initial)
        assert torch.equal(states['scope']['candidate_domain'],states['scope']['proposal_domain'])
        for group in torch.unique(states['scope']['proposal_strata']):
            mask=(states['scope']['proposal_strata']==group)&~initial
            assert int(states['scope']['proposal_domain'][mask].sum())==int(states['matched']['proposal_domain'][mask].sum())
        del old,new,states
    assert len(records)==32
    write(ROOT/'predictions_complete.json',dict(records=records,baseline=baseline))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main
    main(ROOT,'reference-proposals',document_date='2026-10-12')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='0,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
