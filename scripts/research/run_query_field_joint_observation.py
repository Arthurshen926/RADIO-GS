"""Full nine-scene/four-input factorial capability transfer, no parameter search."""
import argparse
import copy
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.query_program import PROGRAM
from radio_gs.query_field.storage import field_identity
from radio_gs.query_field.research_storage import storage_binding,verify_storage_binding,runner_storage_args,validate_research_package,export_receipt
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261009/joint_observation_transfer_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
ARMS=('small_source','small_supervised','full_source','full_supervised')
CHECKPOINT=Path('output/real_direct_membership_20261006/real/model.joblib')
PROTOCOL=Path('docs/experiments/2026-10-09-joint-observation-transfer-protocol.md')


def prepare(*,store_root=None,legacy_v1=False):
    binding=storage_binding(store_root=store_root,legacy_v1=legacy_v1)
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
    write(ROOT/'design.json',dict(storage=binding,fields=fields,arms=arms,protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in code],checkpoint=receipt(CHECKPOINT),prior_summary=receipt(PREVIOUS/'summary.json'),
        mapping=receipt(package/'mapping/scenes.json'),query_program=PROGRAM,
        source_only_arms=['small_source','full_source'],additional_supervision_arms=['small_supervised','full_supervised'],
        scope='all9 UQIS scenes/all268 inputs per arm; not a full five-task study',
        training_steps=0,model_promotion=False,independent_final_test=False,per_task_selection=False,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def frozen_design(*,store_root=None,legacy_v1=False):
    storage_binding(store_root=store_root,legacy_v1=legacy_v1)
    d=read(ROOT/'design.json')
    verify_storage_binding(d,store_root=store_root,legacy_v1=legacy_v1)
    for item in [d[k] for k in ['protocol','checkpoint','prior_summary','mapping']]+d['code']+list(d['arms'].values()):check(item)
    for job in d['fields']:check(job['legacy_manifest'])
    return d


def verify_construction(output,*,store_root=None,legacy_v1=False):
    report=read(Path(output)/'construction.json')
    records={row['arm']:row for row in report['records']}
    if set(records)!=set(ARMS):raise ValueError('incomplete construction arms')
    for arm,row in records.items():
        manifest=validate_research_package(Path(output)/arm/'field',store_root=store_root,legacy_v1=legacy_v1)
        if row['export']!=export_receipt(manifest):
            raise ValueError('construction receipt differs from exported field')
    return report


def build(design,gpus,*,store_root=None,legacy_v1=False):
    verify_storage_binding(design,store_root=store_root,legacy_v1=legacy_v1)
    def action(index,gpu):
        job=design['fields'][index];out=ROOT/'fits'/job['scene']
        if not (out/'construction.json').exists():
            execute(['-m','scripts.research.run_query_field_joint_observation','compile','--index',str(index)]+runner_storage_args(ROOT,store_root=store_root,legacy_v1=legacy_v1),
                    ROOT/'logs/build'/(job['scene']+'.log'),gpu)
        verify_construction(out,store_root=store_root,legacy_v1=legacy_v1)
        print('constructed',job['scene'],flush=True)
    lanes(list(range(9)),gpus,action)
    records=[receipt(ROOT/'fits'/j['scene']/'construction.json') for j in design['fields']]
    write(ROOT/'construction_complete.json',dict(fields=36,records=records,all_builds_before_prediction=True))


def verify_prediction_binding(job,report,*,store_root=None,legacy_v1=False):
    manifest=validate_research_package(job['field'],store_root=store_root,legacy_v1=legacy_v1)
    if report.get('field_sha256')!=field_identity(manifest):
        raise ValueError('prediction receipt field identity differs from current arm')
    if report.get('field_schema',manifest['schema'] if legacy_v1 else None)!=manifest['schema']:
        raise ValueError('prediction receipt field schema differs')
    if report.get('scene')!=job['name'] or report.get('query_program')!=PROGRAM:
        raise ValueError('prediction receipt scene or query program differs')


def verify_predictions(design,*,store_root=None,legacy_v1=False):
    records=[]
    for arm in ARMS:
        cohort=read(design['arms'][arm]['path']);root=ROOT/arm/'predictions/uqis';ids=set()
        for job in cohort['uqis']:
            path=root/'receipts'/(job['name']+'.json');r=read(path)
            verify_prediction_binding(job,r,store_root=store_root,legacy_v1=legacy_v1)
            for q in r['queries']:
                if q['query_id'] in ids:raise ValueError('duplicate prediction')
                ids.add(q['query_id'])
                for folder,suffix,key in [('predictions','.npy','prediction_sha256'),('selections','.npy','selection_sha256'),('gaussians','.pt','gaussian_sha256')]:
                    check(dict(path=str(root/folder/(q['query_id']+suffix)),sha256=q[key]))
            records.append(receipt(path))
        if len(ids)!=268:raise ValueError('incomplete query cohort')
    return records


def predict(design,gpus,*,store_root=None,legacy_v1=False):
    verify_storage_binding(design,store_root=store_root,legacy_v1=legacy_v1)
    read(ROOT/'construction_complete.json')
    for job in design['fields']:
        verify_construction(ROOT/'fits'/job['scene'],store_root=store_root,legacy_v1=legacy_v1)
    tasks=[(arm,job) for arm in ARMS for job in read(design['arms'][arm]['path'])['uqis']]
    def action(task,gpu):
        arm,job=task;c=read(design['arms'][arm]['path']);out=ROOT/arm/'predictions/uqis'
        if (out/'receipts'/(job['name']+'.json')).exists():
            verify_prediction_binding(job,read(out/'receipts'/(job['name']+'.json')),store_root=store_root,legacy_v1=legacy_v1)
            return
        execute(['-m','radio_gs.query_field.public_uqis','--field',job['field'],'--scene',job['name'],
            '--public',str(Path(c['uqis_package'])/'method'),'--encoder',c['encoder'],
            '--output',str(out),'--query-program',PROGRAM]+([] if legacy_v1 else ['--store-root',str(store_root)]),ROOT/'logs/predict'/(arm+'_'+job['name']+'.log'),gpu)
        print('predicted',arm,job['name'],flush=True)
    lanes(tasks,gpus,action)
    write(ROOT/'predictions_complete.json',dict(records=verify_predictions(design,store_root=store_root,legacy_v1=legacy_v1),queries=1072,
        all_arms_before_scoring=True,design=receipt(ROOT/'design.json')))


def score(design,gpus,*,store_root=None,legacy_v1=False):
    verify_storage_binding(design,store_root=store_root,legacy_v1=legacy_v1)
    check(read(ROOT/'predictions_complete.json')['design']);verify_predictions(design,store_root=store_root,legacy_v1=legacy_v1)
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


def main(argv=None):
    global ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','compile','build','predict','score','all'])
    p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,2,3')
    p.add_argument('--root',type=Path,help='Fresh run output root; required for shared v2')
    p.add_argument('--store-root',type=Path,help='Explicit shared v2 store for inputs and outputs')
    p.add_argument('--legacy-v1',action='store_true',help='Explicit historical full-package replay')
    args=p.parse_args(argv)
    if args.legacy_v1 and args.store_root is not None:p.error('--legacy-v1 and --store-root are mutually exclusive')
    if not args.legacy_v1 and (args.root is None or args.store_root is None):p.error('new research runs require --root and --store-root before computation')
    if args.root is not None:ROOT=args.root
    options=dict(store_root=args.store_root,legacy_v1=args.legacy_v1)
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare(**options)
    else:
        design=frozen_design(**options);gpus=list(map(int,args.gpus.split(',')))
        if args.stage=='compile':
            from radio_gs.query_field.compile_joint_observation import compile_scene
            j=design['fields'][args.index]
            compile_scene(j,ROOT/'fits'/j['scene'],j['budgets'],CHECKPOINT,j['excluded_frames'],**options)
        else:
            if args.stage in {'build','all'}:build(design,gpus,**options)
            if args.stage in {'predict','all'}:predict(design,gpus,**options)
            if args.stage in {'score','all'}:score(design,gpus,**options)


if __name__=='__main__':main()
