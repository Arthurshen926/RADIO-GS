"""Complete fixed positive-ray function study on all eight NVOS requests."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261009/positive_rays_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
ARMS=['parent','positive_ray_shared','positive_ray_local']
PROTOCOL=Path('docs/experiments/2026-10-09-positive-ray-protocol.md')

def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);cohort=read(PREVIOUS/'confidence_mixture_function/cohort.json');jobs=[]
    for item in cohort['nvos']:
        cfg=copy.deepcopy(read(item['prediction_config']));base=cfg['jobs'][0];cfg['jobs']=[]
        cfg['prediction_policy']='fixed positive ray energy mixtures; same negative branch, no training or default promotion'
        for arm in ARMS:cfg['jobs'].append(dict(base,name=item['name']+'__'+arm,query_program=arm))
        config=ROOT/'configs'/(item['name']+'.json');write(config,cfg)
        scoring=read(item['scoring_config']);scoring['prediction_config']=str(config)
        scorepath=ROOT/'configs'/(item['name']+'_score.json');write(scorepath,scoring)
        jobs.append(dict(name=item['name'],configuration=receipt(config),scoring=receipt(scorepath),
            field=receipt(Path(base['field'])/'field.pt'),manifest=receipt(Path(base['field'])/'manifest.json'),
            inputs=[receipt(base[k]) for k in ['camera_fixture','positive','negative']]))
    if len(jobs)!=8:raise ValueError('incomplete NVOS cohort')
    write(ROOT/'design.json',dict(jobs=jobs,primary='positive_ray_local',arms=ARMS,protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__))],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        whole_five_task=False,additional_source_state=False,training_steps=0))

def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['protocol']]:check(r)
    return d

def run(gpus):
    d=design()
    def predict(j,gpu):
        for r in [j['configuration'],j['field'],j['manifest']]+j['inputs']:check(r)
        cfg=read(j['configuration']['path'])
        for job in cfg['jobs']:
            execute(['-m','radio_gs.query_field.public_nvos','predict','--config',j['configuration']['path'],
                '--job',job['name'],'--output',str(ROOT/'predictions'/j['name'])],ROOT/'logs'/(job['name']+'.log'),gpu)
        print('predicted',j['name'],flush=True)
    lanes(d['jobs'],gpus,predict)
    records=[]
    for j in d['jobs']:
        cfg=read(j['configuration']['path'])
        for job in cfg['jobs']:
            p=ROOT/'predictions'/j['name']/job['name'];r=read(p/'prediction.json')
            check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
            if r['configuration']!=cfg:raise ValueError('config changed')
            records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records))
    from radio_gs.query_field.public_nvos import score
    for j in d['jobs']:
        check(j['scoring']);score(read(j['scoring']['path']),ROOT/'predictions'/j['name']);print('scored',j['name'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation.json') for j in d['jobs']]))
    from scripts.research.summarize_query_field_signed_regions import main as summarize
    summarize(ROOT,'positive-ray')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
