"""Frozen four-scene conditional Gaussian region scope experiment."""
import argparse,copy,subprocess,time
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.region_carrier import query_text_region_scope
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261011/binary_region_scope_v1')
FIELDS=Path('output/query_field_20261011/binary_region_carrier_v1')
PARENT=Path('output/query_field_20261011/member_energy_v2')
ARMS=['semantic_initial','frozen_visibility','seed_addition','conditional_addition','conditional_replace']
PROTOCOL=Path('docs/experiments/2026-10-11-binary-region-scope-protocol.md')


def prepare():
    cfg=copy.deepcopy(read(PARENT/'prediction_config.json'));cfg['semantic_controls']=ARMS
    cfg.update(status='binary Gaussian region quality transfer; adaptive development',member_readout='native initial and frozen visibility; seed or conditional scope intervention')
    records=[]
    for j in cfg['jobs']:
        assert read(FIELDS/j['scene']/'compilation.json')['complete']
        j['field']=str(FIELDS/j['scene']/'field')
        for p in [Path(j['field'])/'manifest.json',Path(j['queries']),PARENT/'predictions'/j['name']/'predictions.pt']:
            records.append(receipt(p))
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),inputs=records,
        protocol=receipt(PROTOCOL),code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primary='conditional_addition',earlier_development_scores_seen=True,training=False,default_promoted=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['inputs']+d['code']+[d['configuration'],d['protocol']]:check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    field,_=load_field(j['field']);decoder=field.region_carrier.decoder(field.xyz);exclusion=JointExclusionDecoder(field)
    assert old['texts']==[q['text'] for q in read(j['queries'])['queries']]
    for k,v in old['geometry'].items():assert torch.equal(v,getattr(field,k))
    members={a:[] for a in ARMS};diagnostics=[]
    for qi,text in enumerate(old['texts']):
        start=time.monotonic();v,detail=query_text_region_scope(field,old['native_query_vectors'][qi],old['generic_negative_vectors'],region_decoder=decoder,exclusion_decoder=exclusion)
        for a in ARMS[:2]:
            assert torch.equal(v[a]>.5,old['selected_by_control'][a][:,qi]),(text,a)
            assert float((v[a]-old['membership_by_control'][a][:,qi]).abs().max())<=2e-6
        for a in ARMS:members[a].append(v[a])
        changes={a:dict(added=int(((v[a]>.5)&~(v['frozen_visibility']>.5)).sum()),removed=int(((v['frozen_visibility']>.5)&~(v[a]>.5)).sum())) for a in ARMS[2:]}
        diagnostics.append(dict(text=text,seconds=time.monotonic()-start,**detail,changes=changes))
        print('predicted',j['scene'],qi,text,diagnostics[-1],flush=True)
    members={a:torch.stack(v,1) for a,v in members.items()};sets={a:v>.5 for a,v in members.items()}
    old.update(scores={a:old['scores']['semantic_initial'] for a in ARMS},membership_by_control=members,
        selected_by_control=sets,selected=sets['conditional_addition'])
    torch.save(old,out/'predictions.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,diagnostics=diagnostics,queries=len(old['texts']),
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],target_labels_opened=False,source_access=False,
        evaluation_scene_opened_by_predictor=False,all_controls_recomputed_from_field=True))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def pred(i,_):execute(['-m','scripts.research.run_query_field_region_scope','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log')
    lanes(list(range(4)),[0,1],pred)
    write(ROOT/'predictions_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'prediction.json') for j in cfg['jobs']],queries=68,controls=5))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),'--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        metrics=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        previous=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a,m in metrics.items():
            for k in ['iou','ap','localization','observations']:assert m[k]==previous['semantic_initial'][k]
            if a in ARMS[:2]:
                for k in ['miou','count','missing']:assert m['selected'][k]==previous[a]['selected'][k]
            per[a][j['scene']]=m['selected']['miou']
    report=dict(complete=True,macro={a:sum(v.values())/4 for a,v in per.items()},per_scene=per,
        predictions_before_scoring=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'summary.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='4');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
