"""Complete fixed signed-region function study on all eight NVOS requests."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261009/signed_regions_v1')
PREVIOUS=Path('output/query_field_20261009/confidence_mixture_functions_v1')
ARMS=['parent','joint_unsigned','joint_signed']
PROTOCOL=Path('docs/experiments/2026-10-09-signed-region-protocol.md')

def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);cohort=read(PREVIOUS/'confidence_mixture_function/cohort.json');jobs=[]
    for item in cohort['nvos']:
        cfg=copy.deepcopy(read(item['prediction_config']));base=cfg['jobs'][0];cfg['jobs']=[]
        cfg['prediction_policy']='fixed signed region functions; no training or default promotion'
        for arm in ARMS:cfg['jobs'].append(dict(base,name=item['name']+'__'+arm,query_program=arm))
        config=ROOT/'configs'/(item['name']+'.json');write(config,cfg)
        scoring=read(item['scoring_config']);scoring['prediction_config']=str(config)
        scorepath=ROOT/'configs'/(item['name']+'_score.json');write(scorepath,scoring)
        jobs.append(dict(name=item['name'],configuration=receipt(config),scoring=receipt(scorepath),
            field=receipt(Path(base['field'])/'field.pt'),manifest=receipt(Path(base['field'])/'manifest.json'),
            inputs=[receipt(base[k]) for k in ['camera_fixture','positive','negative']]))
    if len(jobs)!=8:raise ValueError('incomplete NVOS cohort')
    write(ROOT/'design.json',dict(jobs=jobs,primary='joint_signed',arms=ARMS,protocol=receipt(PROTOCOL),
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
    per={a:{} for a in ARMS}
    for j in d['jobs']:
        for row in read(ROOT/'predictions'/j['name']/'evaluation.json')['results']:
            arm=row['job'].split('__')[-1];per[arm][j['name']]=row
    macro={a:{key:sum((v['retained_response']['iou'] if key=='response_iou' else v[key]) for v in per[a].values())/8
        for key in ['response_iou','iou','precision','recall']} for a in ARMS}
    report=dict(complete=True,primary='joint_signed',cases=8,macro=macro,per_case=per,whole_five_task=False,model_promotion=False,
        source_frames=120,source_frame_resolution_unmodified=True,prior_labels_seen=True,implementation_commit=d['implementation_commit'])
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-signed-region-results.json',report);print(macro,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpus',default='2,3');a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(list(map(int,a.gpus.split(','))))
