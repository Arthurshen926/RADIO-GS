"""Frozen edit-footprint vs no-footprint study after rejecting incidence equality."""
import argparse
import copy
import subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261012/reference_edit_v1')
PREVIOUS=Path('output/query_field_20261012/reference_extent_v1')
ARMS=['edit','none']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for old in read(PREVIOUS/'design.json')['jobs']:
        check(old['configuration']);cfg=copy.deepcopy(read(old['configuration']['path']))
        base=cfg['jobs'][0]
        cfg['jobs']=[dict(base,name=old['name']+'__'+a,query_program='reference_extent_'+a) for a in ARMS]
        cfg['prediction_policy']='same coupled data function; final-volume bias vs no unary cost vs finite edits'
        path=ROOT/'configs'/(old['name']+'.json');write(path,cfg)
        check(old['scoring']);scoring=read(old['scoring']['path']);scoring['prediction_config']=str(path)
        sp=ROOT/'configs'/(old['name']+'_score.json');write(sp,scoring)
        jobs.append(dict(old,configuration=receipt(path),scoring=receipt(sp)))
    write(ROOT/'design.json',dict(jobs=jobs,arms=ARMS,primary='edit',observation_weight={'edit':.5,'none':.5},regularizers={'edit':'edit','none':'none'},
        protocol=receipt('docs/experiments/2026-10-12-reference-edit-protocol.md'),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),receipt('scripts/research/run_query_field_relations.py')],
        previous_summary=receipt(PREVIOUS/'summary.json'),previous_audit=receipt(PREVIOUS/'audit.json'),
        grouped_summary=receipt('output/query_field_20261012/reference_groups_v1/summary.json'),
        grouped_audit=receipt('output/query_field_20261012/reference_groups_v1/audit.json'),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_frames=120,new_field_state_bytes=0,training_steps=0,raw_scribble=False,whole_five_task=False,
        development_on_previously_evaluated_cases=True,old_controls_rerun=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol'],d['previous_summary'],d['previous_audit'],d['grouped_summary'],d['grouped_audit']]:check(r)
    return d


def run(gpus):
    d=design()
    def predict(j,g):
        for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
        for job in read(j['configuration']['path'])['jobs']:
            execute(['-m','radio_gs.query_field.public_nvos','predict','--config',j['configuration']['path'],
                '--job',job['name'],'--output',str(ROOT/'predictions'/j['name'])],ROOT/'logs'/(job['name']+'.log'),g)
        print('predicted',j['name'],flush=True)
    lanes(d['jobs'],gpus,predict);records=[]
    for j in d['jobs']:
        cfg=read(j['configuration']['path'])
        for job in cfg['jobs']:
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['extent_state'])
            records.append(receipt(p/'prediction.json'))
    assert len(records)==16
    write(ROOT/'predictions_complete.json',dict(records=records))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main
    main(ROOT,'reference-edit',document_date='2026-10-12')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='0,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
