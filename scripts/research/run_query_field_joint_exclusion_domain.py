"""Full4/68 matched set-decoder intervention; all predictions before scores."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from radio_gs.query_field.part_conditioning import native_region_probability
from radio_gs.query_field.scope_confidence import query_text_scope_confident
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.compile_query_field_available_coverage import ROOT as FIELDS
ROOT=Path('output/query_field_20261011/joint_exclusion_domain_v1')
OLD_JOINT=Path('output/query_field_20261010/joint_exclusion_prior_v1')
ADMISSION=Path('output/query_field_20261009/union_admission_v1')
REFERENCE=Path('output/query_field_20261010/scope_confidence_v1')
ARMS=['semantic_initial','observed_joint','observed_fixed','frozen_visibility','joint_coverage']
PROTOCOL=Path('docs/experiments/2026-10-11-available-coverage-protocol.md')


def prepare():
    read(REFERENCE/'scoring_complete.json');ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'design.json').exists():raise ValueError('frozen design already exists')
    cfg=copy.deepcopy(read(REFERENCE/'prediction_config.json'));cfg.update(status='matched actual member-domain correction; fixed strong initialization',semantic_controls=ARMS,
        member_readout='native semantic initialization then unchanged one-pass exclusion; five domain controls',runtime_source_access=False)
    records=[]
    for j in cfg['jobs']:
        assert read(FIELDS/j['scene']/'validation.json')['passed']
        assert read(ROOT/'validation'/(j['scene']+'.json'))['passed']
        records += [receipt(FIELDS/j['scene']/'compilation.json'),receipt(FIELDS/j['scene']/'validation.json'),receipt(ROOT/'validation'/(j['scene']+'.json'))]
        j['field']=str(FIELDS/j['scene']/'field')
        for p in [Path(j['field'])/'manifest.json',Path(j['queries'])]+[REFERENCE/'predictions'/j['name']/n for n in ['prediction.json','predictions.pt']]:records.append(receipt(p))
        # Freeze the earlier actual-set controls separately from semantic inputs.
    for j in cfg['jobs']:
        for root in [OLD_JOINT,ADMISSION]:
            for name in ['prediction.json','predictions.pt']:records.append(receipt(root/'predictions'/j['name']/name))
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),inputs=records,protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),receipt('scripts/research/validate_query_field_joint_exclusion.py')],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),primary='joint_coverage',
        earlier_development_scores_seen=True,model_promotion=False,whole_five_task=False,training=False,query_parameter_fitting=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+d['inputs']+[d['configuration'],d['protocol']]:check(r)
    return d


def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    for job in cfg['jobs']:assert read(ROOT/'validation'/(job['scene']+'.json'))['passed']
    old=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
    historical=torch.load(OLD_JOINT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    admission=torch.load(ADMISSION/'predictions'/j['name']/'predictions.pt',weights_only=True)
    assert old['texts']==historical['texts']==admission['texts']
    field,_=load_field(j['field'],'cuda');decoder=JointExclusionDecoder(field)
    available=field.observed|field.semantic_measure.supported
    assert old['texts']==[q['text'] for q in read(j['queries'])['queries']] and torch.equal(old['observed'],available.cpu())
    for k,v in old['geometry'].items():assert torch.equal(v,getattr(field,k).cpu())
    members={a:[] for a in ARMS};diagnostics=[]
    for qi,text in enumerate(old['texts']):
        native=query_text_scope_confident(field,old['native_query_vectors'][qi].cuda(),old['generic_negative_vectors'].cuda())
        initial=native.semantic_probability;accepted=native.accepted_regions
        assert float((native.semantic_probability.cpu()-old['scores']['confidence_scope'][:,qi]).abs().max())<=2e-6
        assert torch.equal(initial>.5,admission['selected'][:,qi].cuda())
        members['semantic_initial'].append(initial.cpu())
        members['observed_joint'].append(historical['membership_by_control']['joint_coverage'][:,qi])
        members['observed_fixed'].append(historical['membership_by_control']['frozen_visibility'][:,qi]);controls={}
        for joint,a in [(False,'frozen_visibility'),(True,'joint_coverage')]:
            member,detail=decoder.decode(initial,accepted,joint=joint);members[a].append(member.cpu());controls[a]=detail
        diagnostics.append(dict(text=text,controls=controls));print('predicted',j['scene'],qi,controls,flush=True)
    members={a:torch.stack(v,1) for a,v in members.items()};scores={a:old['scores']['confidence_scope'] for a in ARMS};sets={a:members[a]>.5 for a in ARMS}
    old.update(scores=scores,membership_by_control=members,selected=sets['joint_coverage'],selected_by_control=sets)
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(old,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(old['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],target_labels_opened=False,evaluation_scene_opened_by_predictor=False,
        runtime_source_raster_access=False,teacher_acceptance_used=False,transient_query_fitting=False,persistent_field_updated=False))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    for j in cfg['jobs']:assert read(ROOT/'validation'/(j['scene']+'.json'))['passed']
    def action(i,g):execute(['-m','scripts.research.run_query_field_joint_exclusion_domain','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json');assert r['configuration']==cfg and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=5))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),'--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        metrics=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics'];old=read(REFERENCE/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a,m in metrics.items():
            for k in ['iou','ap','localization','observations']:assert m[k]==old['confidence_scope'][k]
            if a in ['semantic_initial','observed_joint','observed_fixed']:
                r=ADMISSION if a=='semantic_initial' else OLD_JOINT
                key='admission_preserving' if a=='semantic_initial' else ('joint_coverage' if a=='observed_joint' else 'frozen_visibility')
                baseline=read(r/'predictions'/j['name']/'evaluation/evaluation.json')['metrics'][key]
                for k in ['miou','acc025','acc05','count','missing']:assert m['selected'][k]==baseline['selected'][k]
            per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},per_scene=per,
        teacher_acceptance_used=False,query_fitting=False,training=False,model_promotion=False,whole_five_task=False)
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-11-joint-exclusion-domain-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='3');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='predict':torch.cuda.set_per_process_memory_fraction(.45)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
