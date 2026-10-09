"""Full nine-scene/four-input factorial capability transfer, no parameter search."""
import argparse
import copy
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.query_program import PROGRAM
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/joint_observation_transfer_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
ARMS=('small_source','small_supervised','full_source','full_supervised')
CHECKPOINT=Path('output/real_direct_membership_20261006/real/model.joblib')
PROTOCOL=Path('docs/experiments/2026-10-09-joint-observation-transfer-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    previous=read(PREVIOUS/'design.json'); prior=read(PREVIOUS/'summary.json')
    cohort=read(PREVIOUS/'confidence_mixture_function/cohort.json')
    package=Path(cohort['uqis_package']); mapping=read(package/'mapping/scenes.json')['scenes']
    legacy=read('configs/pipeline/system_v10.json')
    fields=[]
    for item in previous['fields']:
        if item['family']!='uqis':continue
        job=copy.deepcopy(item); name=job['scene']
        old=next(j for j in legacy['jobs'] if j['benchmark']=='uqis' and j['name']==name)
        job['legacy_map']=old['scene']; spec=read(old['scene'])
        source=read(Path(job['observations'])/'observations.json')
        authority=next(s for s in mapping if s['scene_id']==name)
        allowed=set(map(int,authority['field_frame_ids'])); withheld=set(map(int,authority['withheld_frame_ids']))
        dev=source['splits']['development_frames']; small=source['splits']['train_frames']
        if set(spec['source_frames'])-allowed or set(spec['source_frames'])&withheld:
            raise ValueError('legacy frame budget exceeds current mapping authority')
        if set(small)-set(spec['source_frames']) or set(small)&set(dev):
            raise ValueError('small frame control or development split invalid')
        job.update(budgets=dict(small=small,full=[f for f in spec['source_frames'] if f not in dev]),
                   excluded_frames=dev, legacy_manifest=receipt(old['scene']))
        fields.append(job)
    if len(fields)!=9:raise ValueError('complete UQIS field cohort required')
    arms={}
    for arm in ARMS:
        c=copy.deepcopy(cohort);c['root']=str(ROOT/arm)
        for job in c['uqis']:job['field']=str(ROOT/'fits'/job['name']/arm/'field')
        # This mechanism study has no predictions for the other families.
        for family in ['lerf','nvos','scannet']:c[family]=[]
        write(ROOT/arm/'cohort.json',c);arms[arm]=receipt(ROOT/arm/'cohort.json')
    code=sorted(Path('radio_gs/query_field').glob('*.py'))+[Path(__file__),
        Path('radio_gs/querying/direct_anchor_membership.py'),Path('radio_gs/querying/anchor_observation_evidence.py')]
    write(ROOT/'design.json',dict(fields=fields,arms=arms,protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in code],checkpoint=receipt(CHECKPOINT),prior_summary=receipt(PREVIOUS/'summary.json'),
        mapping=receipt(package/'mapping/scenes.json'),query_program=PROGRAM,
        source_only_arms=['small_source','full_source'],additional_supervision_arms=['small_supervised','full_supervised'],
        scope='all9 UQIS scenes/all268 inputs per arm; not a full five-task study',
        training_steps=0,model_promotion=False,independent_final_test=False,per_task_selection=False,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def frozen_design():
    d=read(ROOT/'design.json')
    for item in [d[k] for k in ['protocol','checkpoint','prior_summary','mapping']]+d['code']+list(d['arms'].values()):check(item)
    for job in d['fields']:check(job['legacy_manifest'])
    return d


def build(design,gpus):
    def action(index,gpu):
        job=design['fields'][index];out=ROOT/'fits'/job['scene']
        if not (out/'construction.json').exists():
            execute(['-m','scripts.research.run_query_field_joint_observation','compile','--index',str(index)],
                    ROOT/'logs/build'/(job['scene']+'.log'),gpu)
        print('constructed',job['scene'],flush=True)
    lanes(list(range(9)),gpus,action)
    records=[receipt(ROOT/'fits'/j['scene']/'construction.json') for j in design['fields']]
    write(ROOT/'construction_complete.json',dict(fields=36,records=records,all_builds_before_prediction=True))


def verify_predictions(design):
    records=[]
    for arm in ARMS:
        cohort=read(design['arms'][arm]['path']);root=ROOT/arm/'predictions/uqis';ids=set()
        for job in cohort['uqis']:
            path=root/'receipts'/(job['name']+'.json');r=read(path)
            for q in r['queries']:
                if q['query_id'] in ids:raise ValueError('duplicate prediction')
                ids.add(q['query_id'])
                for folder,suffix,key in [('predictions','.npy','prediction_sha256'),('selections','.npy','selection_sha256'),('gaussians','.pt','gaussian_sha256')]:
                    check(dict(path=str(root/folder/(q['query_id']+suffix)),sha256=q[key]))
            records.append(receipt(path))
        if len(ids)!=268:raise ValueError('incomplete query cohort')
    return records


def predict(design,gpus):
    read(ROOT/'construction_complete.json')
    tasks=[(arm,job) for arm in ARMS for job in read(design['arms'][arm]['path'])['uqis']]
    def action(task,gpu):
        arm,job=task;c=read(design['arms'][arm]['path']);out=ROOT/arm/'predictions/uqis'
        if (out/'receipts'/(job['name']+'.json')).exists():return
        execute(['-m','radio_gs.query_field.public_uqis','--field',job['field'],'--scene',job['name'],
            '--public',str(Path(c['uqis_package'])/'method'),'--encoder',c['encoder'],
            '--output',str(out),'--query-program',PROGRAM],ROOT/'logs/predict'/(arm+'_'+job['name']+'.log'),gpu)
        print('predicted',arm,job['name'],flush=True)
    lanes(tasks,gpus,action)
    write(ROOT/'predictions_complete.json',dict(records=verify_predictions(design),queries=1072,
        all_arms_before_scoring=True,design=receipt(ROOT/'design.json')))


def score(design,gpus):
    check(read(ROOT/'predictions_complete.json')['design']);verify_predictions(design)
    def action(arm,gpu):
        c=read(design['arms'][arm]['path']);out=ROOT/arm/'predictions/uqis'
        if not (out/'evaluation.json').exists():
            execute(['-m','radio_gs.benchmarks.uqis.evaluate','--package',c['uqis_package'],
                '--predictions',str(out/'predictions'),'--selections',str(out/'selections'),
                '--output',str(out/'evaluation.json')],ROOT/'logs/score'/(arm+'.log'),gpu)
        print('scored',arm,flush=True)
    lanes(list(ARMS),gpus,action)
    write(ROOT/'scoring_complete.json',dict(arms=list(ARMS),queries_per_arm=268,
        records=[receipt(ROOT/a/'predictions/uqis/evaluation.json') for a in ARMS]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','build','predict','score','all'])
    p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,2,3');args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare()
    else:
        design=frozen_design();gpus=list(map(int,args.gpus.split(',')))
        if args.stage=='compile':
            from radio_gs.query_field.compile_joint_observation import compile_scene
            j=design['fields'][args.index]
            compile_scene(j,ROOT/'fits'/j['scene'],j['budgets'],CHECKPOINT,j['excluded_frames'])
        else:
            if args.stage in {'build','all'}:build(design,gpus)
            if args.stage in {'predict','all'}:predict(design,gpus)
            if args.stage in {'score','all'}:score(design,gpus)
