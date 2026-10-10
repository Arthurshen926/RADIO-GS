"""Full LERF query-conditioned part acceptance, all predictions before scoring."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import prepare_parts,read_parts,native_region_probability
from radio_gs.query_field.region_union import prepare_union_parts
from scripts.research.teacher_part_acceptance import select_parts

ROOT=Path('output/query_field_20261010/compositional_atoms_v2')
REFERENCE=Path('output/query_field_20261010/teacher_acceptance_v1')
SOURCE=Path('output/query_field_20261009/text_union_functions_v1')
BASE=Path('output/query_field_20261009/union_admission_v1')
ARMS=['admission','teacher_protected','teacher_union_protected','atom_parts','atom_protected']
PROTOCOL=Path('docs/experiments/2026-10-10-compositional-atoms-protocol.md')


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    read(REFERENCE/'scoring_complete.json')
    cfg=copy.deepcopy(read(REFERENCE/'prediction_config.json'));cfg['status']='fixed same-view marginal union; no promotion'
    legacy={j['name']:j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark']=='lerf'}
    for j in cfg['jobs']:j['teacher_detections']=legacy[j['scene']]['source_evidence']
    cfg['status']='source-assisted functional intervention; never a deployed-model result'
    write(ROOT/'prediction_config.json',cfg)
    records=[]
    for j in cfg['jobs']:
        for p in [Path(j['field'])/'manifest.json',Path(j['queries']),Path(j['construction']),Path(j['observations'])/'observations.json',
                  REFERENCE/'predictions'/j['name']/'prediction.json',REFERENCE/'predictions'/j['name']/'predictions.pt',REFERENCE/'predictions'/j['name']/'teacher_selection.pt',
                  SOURCE/'predictions'/j['name']/'prediction.json',SOURCE/'predictions'/j['name']/'predictions.pt',
                  BASE/'predictions'/j['name']/'prediction.json',BASE/'predictions'/j['name']/'predictions.pt']:
            records.append(receipt(p))
        for row in read(j['construction'])['source_records']:
            frame=row['frame']
            records.extend([receipt(Path(j['observations'])/f'view_{frame:05d}.pt'),
                receipt(Path(j['teacher_detections'])/f'identity_{frame}.pt')])
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        inputs=records,code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__),receipt('scripts/research/compile_query_field_atoms.py')],
        arms=ARMS,primary='atom_protected',implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        current_query_fixtures_reused=True,training_steps=0,new_query_independent_atom_state=True,whole_five_task=False,
        source_assisted_diagnostic=True,query_conditioned_teacher_input=True))


def design(verify_inputs=True):
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]+(d['inputs'] if verify_inputs else []):check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design(verify_inputs=False);cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    current={str(Path(j['field'])/'manifest.json'),j['queries'],j['construction'],str(Path(j['observations'])/'observations.json')}
    for row in read(j['construction'])['source_records']:
        frame=row['frame'];current.update([str(Path(j['observations'])/f'view_{frame:05d}.pt'),str(Path(j['teacher_detections'])/f'identity_{frame}.pt')])
    current.update(str(root/'predictions'/j['name']/name) for root in [SOURCE,BASE,REFERENCE] for name in ['prediction.json','predictions.pt'])
    for r in d['inputs']:
        if r['path'] in current:check(r)
    # load_field below verifies this scene's tensor against the frozen manifest;
    # a worker need not repeatedly hash all unrelated scene tensors.
    data=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
    baseline=torch.load(BASE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
    reference=torch.load(REFERENCE/'predictions'/j['name']/'predictions.pt',map_location='cpu',weights_only=True)
    from scripts.research.compile_query_field_atoms import compile_scene
    from radio_gs.query_field.compositional_evidence import CompositionalEvidence
    field,_=load_field(j['field'],'cpu');available=(field.observed|field.semantic_measure.supported)
    assert data['texts']==[x['text'] for x in read(j['queries'])['queries']]
    for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k))
    # Query-independent compilation finishes before acceptance is read.
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    state,compiled=compile_scene(field,j)
    torch.save(state,out/'atoms.pt');write(out/'compilation.json',dict(**compiled,artifact=receipt(out/'atoms.pt')))
    model=CompositionalEvidence(state,'cuda')
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    record=read(REFERENCE/'predictions'/j['name']/'prediction.json');check(record['teacher_selection'])
    selection=torch.load(REFERENCE/'predictions'/j['name']/'teacher_selection.pt',weights_only=True)
    assert selection['texts']==data['texts'] and torch.equal(selection['views'],field.evidence.views)
    accepted=((selection['probability']>.5)&(mass[:,None]>0)).cuda()
    del field,state;available=available.cuda()
    scores={a:reference['scores'][a] for a in ARMS[:3]}
    for arm in ARMS[3:]:scores[arm]=[]
    diagnostics=[]
    for qi,text in enumerate(data['texts']):
        prior=data['scores']['measure_prior'][:,qi].cuda();prepared=model.prepare(accepted[:,qi])
        prepared[2].update(acceptance_provider='same_frozen_source_teacher',query_conditioned_source_access=True)
        plain,dp=read_parts(prior,available,prepared)
        protected,ds=read_parts(prior,available,prepared,protect=True)
        assert bool((protected>=plain).all())
        scores['atom_parts'].append(plain.cpu());scores['atom_protected'].append(protected.cpu())
        diagnostics.append(dict(text=text,controls=dict(atom_parts=dp,atom_protected=ds)))
        print('predicted',j['scene'],qi,flush=True)
    for a in ARMS[3:]:scores[a]=torch.stack(scores[a],1)
    selected={a:scores[a]>.5 for a in ARMS}
    data.update(scores=scores,selected=selected['atom_protected'],selected_by_control=selected)
    torch.save(data,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(data['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],runtime_source_access=True,source_assisted_diagnostic=True,
        teacher_selection=record['teacher_selection'],compilation=receipt(out/'compilation.json'),
        field_unchanged=True,extra_query_independent_state=True,integrated_deployed_field=False,
        target_labels_opened=False,evaluation_scene_opened_by_predictor=False,
        prior_query_fixture=receipt(SOURCE/'predictions'/j['name']/'predictions.pt'),baseline_reused=True))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_compositional_atoms','predict','--index',str(i)],ROOT/'logs'/('predict_'+str(i)+'.log'),g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        assert r['configuration']==cfg and r['job']==j and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=5))
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
        baseline_scores_exact=True,implementation_commit=d['implementation_commit'],whole_five_task=False,model_promotion=False,source_assisted_diagnostic=True)
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-compositional-atoms-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
