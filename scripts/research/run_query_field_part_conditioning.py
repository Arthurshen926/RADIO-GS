"""Full LERF query-conditioned part acceptance, all predictions before scoring."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import condition_parts

ROOT=Path('output/query_field_20261010/part_conditioning_v1')
SOURCE=Path('output/query_field_20261009/text_union_functions_v1')
BASE=Path('output/query_field_20261009/union_admission_v1')
ARMS=['admission','parts','protected_parts']
PROTOCOL=Path('docs/experiments/2026-10-10-part-conditioning-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(BASE/'prediction_config.json'));cfg['status']='fixed all-case part conditioning; prior query fixtures; no promotion'
    write(ROOT/'prediction_config.json',cfg)
    records=[]
    for j in cfg['jobs']:
        for p in [Path(j['field'])/'field.pt',Path(j['field'])/'manifest.json',Path(j['queries']),
                  SOURCE/'predictions'/j['name']/'prediction.json',SOURCE/'predictions'/j['name']/'predictions.pt',
                  BASE/'predictions'/j['name']/'prediction.json',BASE/'predictions'/j['name']/'predictions.pt']:
            records.append(receipt(p))
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        inputs=records,code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],
        arms=ARMS,primary=ARMS[-1],implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        current_query_fixtures_reused=True,training_steps=0,new_state_bytes=0,whole_five_task=False))


def design(verify_inputs=True):
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]+(d['inputs'] if verify_inputs else []):check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design(verify_inputs=False);cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    current={str(Path(j['field'])/'manifest.json'),j['queries']}
    current.update(str(root/'predictions'/j['name']/name) for root in [SOURCE,BASE] for name in ['prediction.json','predictions.pt'])
    for r in d['inputs']:
        if r['path'] in current:check(r)
    # load_field below verifies this scene's tensor against the frozen manifest;
    # a worker need not repeatedly hash all unrelated scene tensors.
    data=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
    baseline=torch.load(BASE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
    field,_=load_field(j['field'],'cuda');available=field.observed|field.semantic_measure.supported
    assert data['texts']==[x['text'] for x in read(j['queries'])['queries']]
    for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k).cpu())
    scores={'admission':baseline['scores']['admission_preserving']};diagnostics=[]
    for arm in ARMS[1:]:scores[arm]=[]
    for qi,text in enumerate(data['texts']):
        prior=data['scores']['measure_prior'][:,qi].cuda();details={}
        for arm in ARMS[1:]:
            values,details[arm]=condition_parts(field.evidence,prior,available,protect=arm=='protected_parts')
            scores[arm].append(values.cpu())
        diagnostics.append(dict(text=text,controls=details));print('predicted',j['scene'],qi,flush=True)
    for arm in ARMS[1:]:scores[arm]=torch.stack(scores[arm],1)
    selected={a:scores[a]>.5 for a in ARMS}
    assert torch.equal(selected['admission'],baseline['selected'])
    assert torch.equal(selected['admission'],data['selected_by_control']['measure_prior'])
    data.update(scores=scores,selected=selected['protected_parts'],selected_by_control=selected)
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(data,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(data['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],runtime_source_access=False,
        target_labels_opened=False,evaluation_scene_opened_by_predictor=False,
        prior_query_fixture=receipt(SOURCE/'predictions'/j['name']/'predictions.pt'),baseline_reused=True))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_part_conditioning','predict','--index',str(i)],ROOT/'logs'/('predict_'+str(i)+'.log'),g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        assert r['configuration']==cfg and r['job']==j and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=3))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),'--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        result=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(BASE/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['admission_preserving']
        current=result['admission']
        for key in ['iou','ap','localization','observations']:assert current[key]==old[key]
        for key in ['miou','acc025','acc05','count','missing']:assert current['selected'][key]==old['selected'][key]
        # Prediction artifact paths necessarily change; compare all scored
        # objects by their bound labels and values, not their output filenames.
        keys=['frame','query','gt_path','iou','missing']
        assert [[x[k] for k in keys] for x in current['selected']['objects']]==[[x[k] for k in keys] for x in old['selected']['objects']]
        for a in ARMS:
            m=result[a];per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,per_scene=per,macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},
        baseline_scores_exact=True,implementation_commit=d['implementation_commit'],whole_five_task=False,model_promotion=False)
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-part-conditioning-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
