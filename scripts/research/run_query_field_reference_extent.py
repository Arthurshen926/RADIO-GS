"""Frozen eight-case transfer of the shared optical decision engine to NVOS."""
import argparse
import copy
import subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261012/reference_extent_v1')
PREVIOUS=Path('output/query_field_20261010/reference_member_readout_v1')
ARMS=['composed','shared_prior','shared_lift','shared_literal']
PROGRAMS=['reference_scope_admitted','reference_extent_prior','reference_extent_lifted','reference_extent_literal']
PROTOCOL=Path('docs/experiments/2026-10-12-reference-extent-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for old in read(PREVIOUS/'design.json')['jobs']:
        check(old['configuration']);cfg=copy.deepcopy(read(old['configuration']['path']))
        base=next(j for j in cfg['jobs'] if j['name'].endswith('__admission'))
        cfg['jobs']=[dict(base,name=old['name']+'__'+a,query_program=p) for a,p in zip(ARMS,PROGRAMS)]
        cfg['prediction_policy']='same current reference/field; shared optics, initializer and declared task cutoff; no promotion'
        path=ROOT/'configs'/(old['name']+'.json');write(path,cfg)
        check(old['scoring']);scoring=read(old['scoring']['path']);scoring['prediction_config']=str(path)
        sp=ROOT/'configs'/(old['name']+'_score.json');write(sp,scoring)
        jobs.append(dict(old,configuration=receipt(path),scoring=receipt(sp)))
    assert len(jobs)==8
    write(ROOT/'design.json',dict(jobs=jobs,arms=ARMS,primary='shared_prior',protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),
             receipt('scripts/research/run_query_field_relations.py')],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_frames=120,queries_used_for_construction=False,new_field_state_bytes=0,training_steps=0,
        reference_input_cost_inherited=True,raw_scribble=False,whole_five_task=False,
        decision_cutoff={'shared_prior':.5,'shared_lift':.5,'shared_literal':10.5/255},
        reference_raster=[192,256],max_passes=4,initial_composition='same fresh reference_scope_admitted'))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d


def run(gpus):
    d=design()
    def predict(j,g):
        for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
        for job in read(j['configuration']['path'])['jobs']:
            execute(['-m','radio_gs.query_field.public_nvos','predict','--config',j['configuration']['path'],
                     '--job',job['name'],'--output',str(ROOT/'predictions'/j['name'])],
                    ROOT/'logs'/(job['name']+'.log'),g)
        print('predicted',j['name'],flush=True)
    lanes(d['jobs'],gpus,predict)
    records=[];baseline=[]
    for j in d['jobs']:
        cfg=read(j['configuration']['path'])
        for job in cfg['jobs']:
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
            if job['query_program']!='reference_scope_admitted':check(r['extent_state'])
            records.append(receipt(p/'prediction.json'))
        old=torch.load(PREVIOUS/'predictions'/j['name']/(j['name']+'__admission')/'predictions.pt',weights_only=True)
        new=torch.load(ROOT/'predictions'/j['name']/(j['name']+'__composed')/'predictions.pt',weights_only=True)
        errors={k:float((old[k]-new[k]).abs().max()) for k in ['membership','response','retained_response']}
        exact=torch.equal(old['selected'],new['selected'])
        baseline.append(dict(case=j['name'],selected_exact=exact,max_errors=errors))
        assert exact and max(errors.values())<=1e-6,(j['name'],baseline[-1])
    assert len(records)==32
    write(ROOT/'baseline_checks.json',dict(records=baseline))
    write(ROOT/'predictions_complete.json',dict(records=records,baseline=baseline))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name'])
        print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main
    main(ROOT,'reference-extent',document_date='2026-10-12')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='0,2,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
