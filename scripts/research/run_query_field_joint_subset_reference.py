"""Fixed-source joint-subset functional reference; no deployment path."""
import argparse,copy,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.conditional_subset_reference import solve,unoccluded_footprint
from radio_gs.query_field.relation_training import SourceRelations

ROOT=Path('output/query_field_20261010/joint_subset_reference_v1')
REFERENCE=Path('output/query_field_20261010/source_member_readout_v1')
ARMS=['source_parts','joint_no_footprint','joint_footprint']
PROTOCOL=Path('docs/experiments/2026-10-10-joint-subset-reference-protocol.md')


def prepare():
    read(REFERENCE/'predictions_complete.json');ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(REFERENCE/'prediction_config.json'));cfg['status']='source-assisted transient joint-fit capability reference only'
    write(ROOT/'prediction_config.json',cfg);records=[]
    for j in cfg['jobs']:
        p=REFERENCE/'predictions'/j['name'];r=read(p/'prediction.json')
        paths=[p/'prediction.json',p/'predictions.pt',Path(r['teacher_selection']['path']),Path(j['queries']),
            Path(j['field'])/'manifest.json',Path(j['construction']),Path(j['observations'])/'observations.json']
        paths += [Path(j['observations'])/f"view_{v['frame']:05d}.pt" for v in read(j['construction'])['source_records']]
        records.extend(receipt(p) for p in paths)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),inputs=records,
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),receipt('scripts/research/conditional_subset_reference.py')],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arms=ARMS,primary='joint_footprint',
        steps=240,source_assisted_diagnostic=True,transient_query_fitting=True,persistent_field_training=False,
        whole_five_task=False,model_promotion=False,earlier_development_scores_seen=True))


def design(verify_inputs=True):
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]+(d['inputs'] if verify_inputs else []):check(r)
    return d


def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index];parent=REFERENCE/'predictions'/j['name'];r=read(parent/'prediction.json')
    data=torch.load(parent/'predictions.pt',weights_only=True);teacher=torch.load(r['teacher_selection']['path'],weights_only=True)
    assert data['texts']==teacher['texts']==[x['text'] for x in read(j['queries'])['queries']]
    field=SimpleNamespace(**{k:v.cuda() for k,v in data['geometry'].items()},observed=data['observed'].cuda())
    scale=read(Path(j['field'])/'manifest.json')['spec']['world_units_per_metre']
    source=SourceRelations(j['observations'],field.observed)
    assert source.manifest['world_units_per_metre']==scale
    construction=read(j['construction']);assert len(source.views)==len(construction['source_records'])
    offset=0
    for vi,(v,original) in enumerate(zip(source.views,construction['source_records'])):
        assert v['frame']==original['frame'] and v['original_regions']==original['regions']
        count=len(v['positive']);v['accepted']=teacher['probability'][offset:offset+count]>.5
        assert torch.equal(teacher['views'][offset:offset+count],torch.full((count,),vi))
        # Existing partial targets were built at the fixed observation raster.
        raw=torch.load(Path(j['observations'])/f"view_{v['frame']:05d}.pt",weights_only=True)
        v['height'],v['width']=raw['masks'].shape[-2:];del raw
        v['footprint']=unoccluded_footprint(field,v['camera'],v['height'],v['width'],scale)
        offset+=count
    assert offset==len(teacher['views'])
    scores={'source_parts':data['scores']['source_parts'],'joint_no_footprint':[],'joint_footprint':[]};diagnostics=[]
    for qi,text in enumerate(data['texts']):
        views=[];accepted=0
        for v in source.views:
            ids=v['accepted'][:,qi];accepted+=int(ids.sum())
            if not bool(ids.any()):continue
            views.append(dict(camera=v['camera'],height=v['height'],width=v['width'],footprint=v['footprint'],
                positive=v['positive'][ids].any(0).cuda(),negative=v['negative'][ids].all(0).cuda()))
        assert accepted==r['diagnostics'][qi]['controls']['source_parts']['admitted_regions'],'region eligibility changed'
        initial=data['scores']['source_parts'][:,qi].cuda();details={}
        for a,penalty in [('joint_no_footprint',False),('joint_footprint',True)]:
            result,detail=solve(field,initial,views,scale,footprint=penalty,steps=d['steps'])
            scores[a].append(result.cpu());details[a]=detail
            print('fitted',j['scene'],qi,a,detail,flush=True)
        diagnostics.append(dict(text=text,controls=details))
    for a in ARMS[1:]:scores[a]=torch.stack(scores[a],1)
    for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k).cpu()),'geometry changed'
    selected={a:scores[a]>.5 for a in ARMS};data.update(scores=scores,selected=selected['joint_footprint'],selected_by_control=selected)
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(data,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(data['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],source_assisted_diagnostic=True,runtime_source_access=True,
        transient_query_fitting=True,persistent_field_updated=False,geometry_exact=True,
        target_labels_opened=False,evaluation_scene_opened_by_predictor=False,teacher_selection=r['teacher_selection']))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_joint_subset_reference','predict','--index',str(i)],ROOT/'logs'/('predict_'+str(i)+'.log'),g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json');assert r['configuration']==cfg and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=3))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),'--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        result=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(REFERENCE/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['source_parts'];current=result['source_parts']
        for k in ['iou','ap','localization','observations']:assert current[k]==old[k]
        for k in ['miou','acc025','acc05','count','missing']:assert current['selected'][k]==old['selected'][k]
        for a,m in result.items():per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,per_scene=per,macro={a:{k:sum(v[k] for v in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},
        source_assisted_diagnostic=True,transient_query_fitting=True,persistent_field_updated=False,model_promotion=False,whole_five_task=False,
        implementation_commit=d['implementation_commit'])
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-joint-subset-reference-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
