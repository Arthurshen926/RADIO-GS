"""Exploratory repair after full v1 failure; same sealed field/query evidence."""
import argparse
import copy
from pathlib import Path
import subprocess
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute
from radio_gs.query_field.query_program import admission_preserving_support

ROOT=Path('output/query_field_20261009/union_admission_v1')
SOURCE=Path('output/query_field_20261009/text_union_functions_v1')
PROTOCOL=Path('docs/experiments/2026-10-09-union-admission-protocol.md')


def prepare():
    read(SOURCE/'scoring_complete.json');ROOT.mkdir(exist_ok=False,parents=True)
    cfg=copy.deepcopy(read(SOURCE/'prediction_config.json'))
    cfg['jobs']=[j for j in cfg['jobs'] if j['name'].endswith('__measure')]
    cfg['status']='exploratory admission repair after v1 labels;not independent validation'
    write(ROOT/'prediction_config.json',cfg)
    sources=[]
    for j in cfg['jobs']:
        for name in [j['name'],j['scene']+'__parent']:
            sources.extend(receipt(SOURCE/'predictions'/name/p) for p in ['prediction.json','predictions.pt'])
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        sources=sources,code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(Path(__file__))],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primary='admission_preserving',exploratory=True,previous_labels_seen=True,new_training=False,new_source_state=False))


def run(gpu):
    d=read(ROOT/'design.json')
    for r in d['sources']+d['code']+[d['configuration'],d['protocol']]:check(r)
    cfg=read(d['configuration']['path']);records=[]
    for j in cfg['jobs']:
        source=SOURCE/'predictions'/j['name'];old=read(source/'prediction.json')
        data=torch.load(source/'predictions.pt',weights_only=True)
        parent=torch.load(SOURCE/'predictions'/(j['scene']+'__parent')/'predictions.pt',weights_only=True)
        scores=admission_preserving_support(data['scores']['measure_prior'],parent['scores']['shared_membership'])
        selected=scores>.5
        if not torch.equal(selected,data['selected_by_control']['measure_prior']):raise ValueError('admission was changed')
        data.update(scores={'admission_preserving':scores},selected=selected,selected_by_control={'admission_preserving':selected})
        path=ROOT/'predictions'/j['name'];path.mkdir(parents=True,exist_ok=False);torch.save(data,path/'predictions.pt')
        write(path/'prediction.json',dict(job=j,configuration=cfg,predictions_sha256=receipt(path/'predictions.pt')['sha256'],
            queries=old['queries'],source_prediction=receipt(source/'prediction.json'),runtime_source_access=False,
            previous_labels_seen=True,predictor_reads_current_labels=False,exact_semantic_admission=True))
        records.append(receipt(path/'prediction.json'));print('predicted',j['scene'],flush=True)
    write(ROOT/'predictions_complete.json',dict(records=records))
    for j in cfg['jobs']:
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
            '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/(j['name']+'.log'),gpu)
        print('scored',j['scene'],flush=True)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    previous=read(SOURCE/'summary.json');per={}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['admission_preserving']
        per[j['scene']]=dict(response_iou=m['iou'],ap=m['ap'],selection_iou=m['selected']['miou'])
        if m['selected']['miou']!=previous['per_scene']['measure_prior']['selection_iou'][j['scene']]:raise ValueError('same sets scored differently')
    report=dict(complete=True,exploratory=True,previous_labels_seen=True,scenes=4,per_scene=per,
        macro={k:sum(v[k] for v in per.values())/4 for k in next(iter(per.values()))},
        exact_semantic_admission=True,conditional_extent_transferred=False,whole_five_task=False,model_promotion=False,
        implementation_commit=d['implementation_commit'],design=receipt(ROOT/'design.json'))
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-union-admission-results.json',report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else run(a.gpu)
