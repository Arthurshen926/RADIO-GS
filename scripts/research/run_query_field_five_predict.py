"""Complete every family before scoring the fixed five-benchmark architecture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from radio_gs.query_field.storage import sha256
from scripts.research.prepare_query_field_five_benchmarks import ROOT, CONFIG, write


PY=sys.executable


def execute(command, log, environment=None):
    with Path(log).open('x') as handle:
        subprocess.run([PY,*command],env=environment,stdout=handle,stderr=subprocess.STDOUT,check=True)


def verify_predictions(cohort):
    records=[]
    for job in cohort['lerf']:
        root=Path(job['predictions']);record=json.loads((root/'prediction.json').read_text())
        if sha256(root/'predictions.pt')!=record['predictions_sha256']:raise ValueError('retained LERF payload changed')
        if not (root/'evaluation/evaluation.json').exists():raise ValueError('retained LERF scoring absent')
        records.append(dict(family='lerf',scene=job['scene'],reused=True,predictions_sha256=record['predictions_sha256']))
    for family in ['nvos','scannet']:
        for job in cohort[family]:
            root=ROOT/'predictions'/family/job['name']
            if family=='nvos':root=root/job['name']
            record=json.loads((root/'prediction.json').read_text())
            if sha256(root/'predictions.pt')!=record['predictions_sha256']:raise ValueError('new prediction payload changed')
            if family=='nvos':
                expected=json.loads(Path(job['prediction_config']).read_text())
                if record['configuration']!=expected:raise ValueError('NVOS configuration changed')
                if job['reference_prediction']:
                    a=torch.load(root/'predictions.pt',map_location='cpu',weights_only=True)
                    b=torch.load(Path(job['reference_prediction'])/'predictions.pt',map_location='cpu',weights_only=True)
                    if not all(torch.equal(a[k],b[k]) for k in a):raise ValueError('retained fern did not replay')
            records.append(dict(family=family,scene=job['name'],predictions_sha256=record['predictions_sha256']))
    root=ROOT/'predictions/uqis'
    public=Path(cohort['uqis_package'])/'method'
    expected={q['query_id'] for m in ['text','image','point_2d','point_3d']
              for q in json.loads((public/'queries'/f'{m}.json').read_text())['queries']}
    if {p.stem for p in (root/'predictions').glob('*.npy')}!=expected:raise ValueError('incomplete UQIS scores')
    if {p.stem for p in (root/'selections').glob('*.npy')}!=expected:raise ValueError('incomplete UQIS actual sets')
    for job in cohort['uqis']:
        record=json.loads((root/'receipts'/f"{job['name']}.json").read_text())
        for row in record['queries']:
            for folder,key in [('predictions','prediction_sha256'),('selections','selection_sha256')]:
                if sha256(root/folder/f"{row['query_id']}.npy")!=row[key]:raise ValueError('UQIS payload changed')
            if sha256(root/'gaussians'/f"{row['query_id']}.pt")!=row['gaussian_sha256']:
                raise ValueError('UQIS Gaussian output changed')
        records.append(dict(family='uqis',scene=job['name'],queries=record['query_count'],unsupported=record['unsupported']))
    return records


def predict(cohort,gpus):
    for job in cohort['construction']:
        if not (Path(job['output'])/'build_complete.json').exists():raise ValueError('construction incomplete: '+job['name'])
    tasks=[(family,job) for family in ['scannet','uqis','nvos'] for job in cohort[family]]
    lanes=[tasks[i::len(gpus)] for i in range(len(gpus))]
    logs=ROOT/'prediction_logs';logs.mkdir(exist_ok=True)
    def lane(index):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[index]),PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
        for family,job in lanes[index]:
            name=job['name'];output=ROOT/'predictions'/family
            if family=='scannet':
                output=output/name
                if (output/'prediction.json').exists():continue
                command=['-m','radio_gs.query_field.public_scannet','predict','--field',job['field'],'--queries',job['queries'],
                         '--encoder',cohort['encoder'],'--output',str(output)]
            elif family=='uqis':
                if (output/'receipts'/f'{name}.json').exists():continue
                command=['-m','radio_gs.query_field.public_uqis','--field',job['field'],'--scene',name,
                         '--public',str(Path(cohort['uqis_package'])/'method'),'--encoder',cohort['encoder'],'--output',str(output)]
            else:
                output=output/name
                if (output/name/'prediction.json').exists():continue
                command=['-m','radio_gs.query_field.public_nvos','predict','--config',job['prediction_config'],
                         '--job',name,'--output',str(output)]
            execute(command,logs/f'{family}_{name}.log',env)
            print('predicted',family,name,'GPU',gpus[index],flush=True)
    with ThreadPoolExecutor(len(lanes)) as pool:list(pool.map(lane,range(len(lanes))))
    write(ROOT/'predictions_complete.json',dict(records=verify_predictions(cohort),
          all_new_cohorts_before_new_scoring=True,cohort_sha256=sha256(CONFIG/'cohort.json')))


def score(cohort):
    barrier=json.loads((ROOT/'predictions_complete.json').read_text())
    if barrier['cohort_sha256']!=sha256(CONFIG/'cohort.json'):raise ValueError('cohort changed')
    verify_predictions(cohort)
    logs=ROOT/'scoring_logs';logs.mkdir(exist_ok=True)
    jobs=[]
    for family in ['scannet','nvos']:
        for job in cohort[family]:
            root=ROOT/'predictions'/family/job['name']
            if (root/'evaluation.json').exists():continue
            command=(['-m','radio_gs.query_field.public_scannet','score','--predictions',str(root),
                      '--evaluation',job['evaluation'],'--output',str(root/'evaluation.json')]
                     if family=='scannet' else ['-m','radio_gs.query_field.public_nvos','score',
                          '--config',job['scoring_config'],'--output',str(root)])
            jobs.append((command,logs/f"{family}_{job['name']}.log"))
    def run(job):
        execute(*job,dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.'))
    with ThreadPoolExecutor(3) as pool:list(pool.map(run,jobs))
    root=ROOT/'predictions/uqis'
    if not (root/'evaluation.json').exists():
        run((['-m','radio_gs.benchmarks.uqis.evaluate','--package',cohort['uqis_package'],
              '--predictions',str(root/'predictions'),'--selections',str(root/'selections'),
              '--output',str(root/'evaluation.json')],logs/'uqis.log'))
    write(ROOT/'scoring_complete.json',dict(cohort_sha256=sha256(CONFIG/'cohort.json'),
                                           tasks=['lerf2d','lerf3d_local_projection','scannet_ovs','nvos','uqis']))
    print('all five benchmark families scored',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['predict','score']);parser.add_argument('--gpus',nargs='+',type=int,default=[1,2,3,4,5])
    args=parser.parse_args();cohort=json.loads((CONFIG/'cohort.json').read_text())
    predict(cohort,args.gpus) if args.stage=='predict' else score(cohort)
