"""Fixed-measure scope-exclusion intervention with explicit semantic/member roles."""
import argparse,copy,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.conditional_subset_reference import unoccluded_footprint
from scripts.research.fixed_initializer_subset_reference import solve_fixed
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.scope_exclusion import union_exclusion

ROOT=Path('output/query_field_20261010/scope_exclusion_v1')
REFERENCE=Path('output/query_field_20261010/fixed_initializer_reference_v1')
FINITE=Path('output/query_field_20261010/compositional_atoms_v2')
NORMALIZER=Path('output/query_field_20261010/joint_subset_reference_v1')
SEMANTIC=Path('output/query_field_20261009/text_union_functions_v1')
ARMS=['partial_member','partial_typed','scope_member','scope_typed']
PROTOCOL=Path('docs/experiments/2026-10-10-scope-exclusion-protocol.md')


def prepare():
    read(REFERENCE/'scoring_complete.json');ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(REFERENCE/'prediction_config.json'))
    cfg.update(status='source-assisted fixed exclusion and typed-readout diagnostic',runtime_source_access=True,
        semantic_controls=ARMS,selection='each declared membership >.5, independent of semantic probability',
        member_readout='fixed-energy 240-step research reference; no deployed query optimizer')
    write(ROOT/'prediction_config.json',cfg);records=[]
    for j in cfg['jobs']:
        r=read(REFERENCE/'predictions'/j['name']/'prediction.json')
        paths=[Path(r['teacher_selection']['path']),Path(j['queries']),Path(j['field'])/'manifest.json',
            Path(j['construction']),Path(j['observations'])/'observations.json']
        paths += [root/'predictions'/j['name']/name for root in [REFERENCE,FINITE,NORMALIZER,SEMANTIC] for name in ['prediction.json','predictions.pt']]
        paths += [Path(j['observations'])/f"view_{v['frame']:05d}.pt" for v in read(j['construction'])['source_records']]
        records.extend(receipt(p) for p in paths)
    code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]
    code += [receipt(__file__),receipt('scripts/research/conditional_subset_reference.py'),receipt('scripts/research/fixed_initializer_subset_reference.py'),receipt('radio_gs/evaluation/lerf_readouts.py')]
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),inputs=records,
        code=code,implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arms=ARMS,
        primary='scope_typed',steps=240,source_assisted_diagnostic=True,transient_query_fitting=True,
        persistent_field_training=False,whole_five_task=False,model_promotion=False,earlier_development_scores_seen=True))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]+d['inputs']:check(r)
    return d


def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    parent=REFERENCE/'predictions'/j['name'];r=read(parent/'prediction.json')
    data=torch.load(parent/'predictions.pt',weights_only=True)
    teacher=torch.load(r['teacher_selection']['path'],weights_only=True)
    finite=torch.load(FINITE/'predictions'/j['name']/'predictions.pt',weights_only=True)
    normalizer=torch.load(NORMALIZER/'predictions'/j['name']/'predictions.pt',weights_only=True)
    semantic=torch.load(SEMANTIC/'predictions'/j['name']/'predictions.pt',weights_only=True)
    assert data['texts']==teacher['texts']==[x['text'] for x in read(j['queries'])['queries']]
    for cached in [finite,normalizer,semantic]:
        assert cached['texts']==data['texts'] and torch.equal(cached['observed'],data['observed'])
        for k,v in data['geometry'].items():assert torch.equal(v,cached['geometry'][k])
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
        raw=torch.load(Path(j['observations'])/f"view_{v['frame']:05d}.pt",weights_only=True)
        v['masks']=raw['masks'].bool();v['all_scales']=raw['scales_m'];v['height'],v['width']=raw['masks'].shape[-2:]
        v['visible']=raw['raw_mass'].reshape(v['height'],v['width'])>0
        v['footprint']=unoccluded_footprint(field,v['camera'],v['height'],v['width'],scale);offset+=count
    assert offset==len(teacher['views'])
    new=[];diagnostics=[]
    for qi,text in enumerate(data['texts']):
        views=[];permissions=[]
        for v in source.views:
            ids=v['accepted'][:,qi]
            if not bool(ids.any()):continue
            pos=v['positive'][ids].any(0);old=v['negative'][ids].all(0)
            accepted=torch.zeros(len(v['masks']),dtype=torch.bool)
            accepted[torch.tensor(v['original_regions'])[ids]]=True
            neg,permitted=union_exclusion(v['masks'],v['all_scales'],accepted,v['visible'],old.reshape(v['height'],v['width']))
            neg=neg.flatten();assert not bool((pos&neg).any()) and bool((neg|~old).all())
            if int(ids.sum())==1:assert torch.equal(neg,old),'singleton changed'
            views.append(dict(camera=v['camera'],height=v['height'],width=v['width'],footprint=v['footprint'],positive=pos.cuda(),negative=neg.cuda()))
            permissions.append(dict(frame=v['frame'],accepted_regions=int(ids.sum()),permitted_regions=int(permitted.sum()),
                positive_pixels=int(pos.sum()),old_negative_pixels=int(old.sum()),new_negative_pixels=int(neg.sum()),added_negative_pixels=int((neg&~old).sum())))
        old_details=r['diagnostics'][qi]['controls']['joint_finite_initializer']
        assert sum(bool(v['positive'].any()) for v in views)==old_details['active_views']
        result,details=solve_fixed(field,finite['scores']['atom_parts'][:,qi].cuda(),views,scale,
            normalizer=normalizer['scores']['source_parts'][:,qi].cuda(),steps=d['steps'])
        new.append(result.cpu());diagnostics.append(dict(text=text,fitting=details,permissions=permissions))
        print('fitted',j['scene'],qi,details,flush=True)
    member=torch.stack(new,1);prior=semantic['scores']['measure_prior'];old=data['scores']['joint_finite_initializer']
    scores=dict(partial_member=old,partial_typed=prior,scope_member=member,scope_typed=prior)
    sets=dict(partial_member=old>.5,partial_typed=old>.5,scope_member=member>.5,scope_typed=member>.5)
    for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k).cpu())
    data.update(scores=scores,selected=sets['scope_typed'],selected_by_control=sets,membership_by_control=dict(partial=old,scope=member))
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(data,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(data['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],source_assisted_diagnostic=True,runtime_source_access=True,
        transient_query_fitting=True,persistent_field_updated=False,target_labels_opened=False,evaluation_scene_opened_by_predictor=False,
        teacher_selection=r['teacher_selection'],semantic_parent=receipt(SEMANTIC/'predictions'/j['name']/'predictions.pt'),
        readout_contract='semantic measure probability is independent of actual shared 2D/3D member set'))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_scope_exclusion','predict','--index',str(i)],ROOT/'logs'/('predict_'+str(i)+'.log'),g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json');assert r['configuration']==cfg and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=len(ARMS)))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),'--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        result=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(REFERENCE/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['joint_finite_initializer']
        native=read(SEMANTIC/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['measure_prior']
        for k in ['iou','ap','localization','observations']:assert result['partial_member'][k]==old[k]
        for a in ['partial_typed','scope_typed']:
            for k in ['iou','ap','localization','observations']:assert result[a][k]==native[k]
        for a,b in [('partial_member','partial_typed'),('scope_member','scope_typed')]:
            for k in ['miou','acc025','acc05','count','missing']:assert result[a]['selected'][k]==result[b]['selected'][k]
        for a,m in result.items():per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,per_scene=per,macro={a:{k:sum(v[k] for v in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},
        source_assisted_diagnostic=True,transient_query_fitting=True,persistent_field_updated=False,model_promotion=False,whole_five_task=False,
        implementation_commit=d['implementation_commit'])
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-scope-exclusion-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
