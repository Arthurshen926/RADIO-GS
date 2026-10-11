"""Post-hoc 2x2 actual-set intervention; not a new inference candidate."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_member_energy import ROOT as PARENT,read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261011/member_change_factorial_v1')
ARMS=['semantic_initial','member_energy','removal_only','addition_only']


def prepare():
    read(PARENT/'predictions_complete.json')
    cfg=copy.deepcopy(read(PARENT/'prediction_config.json'));cfg['semantic_controls']=ARMS
    cfg['status']='post-hoc member-change factorial diagnostic; not deployable candidate'
    write(ROOT/'prediction_config.json',cfg)
    inputs=[]
    for j in cfg['jobs']:
        inputs.extend([receipt(PARENT/'predictions'/j['name']/'predictions.pt'),receipt(PARENT/'predictions'/j['name']/'prediction.json')])
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),inputs=inputs,code=receipt(__file__),
        intervention='initial, final, intersection(initial,final), union(initial,final)',
        post_hoc=True,trigger='figurines score seen: member_energy41.9015 vs frozen55.7839; no_footprint43.4708',
        model_promotion=False,whole_five_task=False,implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    records=[]
    for j in cfg['jobs']:
        old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        initial=old['selected_by_control']['semantic_initial'];final=old['selected_by_control']['member_energy']
        sets=dict(semantic_initial=initial,member_energy=final,removal_only=initial&final,addition_only=initial|final)
        old.update(selected_by_control=sets,selected=final,membership_by_control={a:s.float() for a,s in sets.items()},
            scores={a:old['scores']['semantic_initial'] for a in ARMS})
        out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(old,out/'predictions.pt')
        write(out/'prediction.json',dict(configuration=cfg,job=j,post_hoc=True,queries=len(old['texts']),
            predictions_sha256=receipt(out/'predictions.pt')['sha256'],target_labels_opened=False,source_raster_access=False))
        records.append(receipt(out/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=4))


def score(gpus):
    d=read(ROOT/'design.json')
    for r in d['inputs']+[d['configuration'],d['code']]:check(r)
    for r in read(ROOT/'predictions_complete.json')['records']:
        check(r);v=read(r['path']);check(dict(path=str(Path(r['path']).parent/'predictions.pt'),sha256=v['predictions_sha256']))
    cfg=read(d['configuration']['path'])
    def action(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
        '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,action)
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            if a in ARMS[:2]:assert m[a]['selected']==old[a]['selected'] or all(m[a]['selected'][k]==old[a]['selected'][k] for k in ['miou','acc025','acc05','count','missing'])
            per[a][j['scene']]=m[a]['selected']['miou']
    macro={a:sum(v.values())/4 for a,v in per.items()}
    write(ROOT/'summary.json',dict(complete=True,macro=macro,per_scene=per,post_hoc=True,model_promotion=False,
        addition_effect_on_initial=macro['addition_only']-macro['semantic_initial'],
        removal_effect_on_initial=macro['removal_only']-macro['semantic_initial'],
        interaction=macro['member_energy']-macro['removal_only']-macro['addition_only']+macro['semantic_initial']))
    print(macro,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score']);p.add_argument('--gpus',default='2');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    else:score(list(map(int,a.gpus.split(','))))
