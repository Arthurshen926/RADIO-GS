"""Four-scene native/teacher controls for field-owned union constraints."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import native_region_probability,read_parts
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.compile_query_field_scope_constraints import ROOT as FIELDS,CONFIG

ROOT=Path('output/query_field_20261010/scope_constraint_readout_v1')
SOURCE=Path('output/query_field_20261009/text_union_functions_v1')
REFERENCE=Path('output/query_field_20261010/scope_exclusion_v1')
ARMS=['native_partial','native_scope','teacher_partial','teacher_scope']
PROTOCOL=Path('docs/experiments/2026-10-10-scope-constraints-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);cfg=copy.deepcopy(read(CONFIG))
    cfg.update(status='compiled union-constraint controls; teacher arms are diagnostic',semantic_controls=ARMS,
        selection='each own finite evidence member>.5, same actual set for 2D/3D',runtime_source_access=False)
    for j in cfg['jobs']:j['field']=str(FIELDS/j['scene']/'field')
    write(ROOT/'prediction_config.json',cfg);records=[]
    for j in cfg['jobs']:
        assert read(FIELDS/j['scene']/'validation.json')['passed']
        teacher=read(REFERENCE/'predictions'/j['name']/'prediction.json')['teacher_selection']
        for p in [Path(j['field'])/'manifest.json',FIELDS/j['scene']/'compilation.json',FIELDS/j['scene']/'validation.json',
                  SOURCE/'predictions'/j['name']/'prediction.json',SOURCE/'predictions'/j['name']/'predictions.pt',Path(j['queries']),Path(teacher['path'])]:records.append(receipt(p))
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),inputs=records,
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],arms=ARMS,primary='native_scope',
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_steps=0,
        earlier_development_scores_seen=True,model_promotion=False,whole_five_task=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+d['inputs']+[d['configuration'],d['protocol']]:check(r)
    return d


def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    fixture=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
    teacher_receipt=read(REFERENCE/'predictions'/j['name']/'prediction.json')['teacher_selection'];check(teacher_receipt)
    teacher=torch.load(teacher_receipt['path'],weights_only=True)
    field,_=load_field(j['field'],'cuda');available=field.observed|field.semantic_measure.supported
    assert fixture['texts']==teacher['texts']==[q['text'] for q in read(j['queries'])['queries']]
    assert torch.equal(teacher['views'],field.evidence.views.cpu()) and torch.equal(fixture['observed'],available.cpu())
    for k,v in fixture['geometry'].items():assert torch.equal(v,getattr(field,k).cpu())
    mass=torch.sparse.mm(field.evidence.matrix('positive').T,available.float()[:,None]).flatten()
    members={a:[] for a in ARMS};diagnostics=[]
    for qi,text in enumerate(fixture['texts']):
        prior=fixture['scores']['measure_prior'][:,qi].cuda()
        native=native_region_probability(field.evidence,fixture['native_query_vectors'][qi].cuda(),fixture['generic_negative_vectors'].cuda())
        providers=dict(native=(native>.5)&(mass>0),teacher=(teacher['probability'][:,qi].cuda()>.5)&(mass>0));details={}
        for provider,accepted in providers.items():
            prepared={}
            for scope in [False,True]:
                arm=provider+('_scope' if scope else '_partial')
                prepared[scope]=field.scope_constraints(accepted,field.scope_permissions,expanded=scope)
                score,detail=read_parts(prior,available,prepared[scope]);members[arm].append(score.cpu());details[arm]=detail
            assert torch.equal(prepared[False][0],prepared[True][0])
            assert bool((prepared[True][1]>=prepared[False][1]-2e-6).all())
        diagnostics.append(dict(text=text,controls=details));print('predicted',j['scene'],qi,flush=True)
    members={a:torch.stack(v,1) for a,v in members.items()};sets={a:v>.5 for a,v in members.items()}
    scores={a:fixture['scores']['measure_prior'] for a in ARMS}
    fixture.update(scores=scores,membership_by_control=members,selected=sets['native_scope'],selected_by_control=sets)
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(fixture,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(fixture['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],runtime_source_raster_access=False,
        cached_current_query_vectors_and_semantic_response=True,teacher_acceptance_used_only_for_named_teacher_controls=True,
        teacher_selection=teacher_receipt,transient_query_fitting=False,persistent_field_updated=False,
        target_labels_opened=False,evaluation_scene_opened_by_predictor=False))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_scope_constraints','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g)
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
        metrics=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(SOURCE/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']['measure_prior']
        for a,m in metrics.items():
            for k in ['iou','ap','localization','observations']:assert m[k]==old[k]
            per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,per_scene=per,macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},
        primary='native_scope',teacher_controls_are_privileged=True,new_training=False,model_promotion=False,whole_five_task=False)
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-scope-constraints-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
