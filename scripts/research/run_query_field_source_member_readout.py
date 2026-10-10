"""Fixed-evidence source-normalized member readout, complete LERF controls."""
import argparse,copy,subprocess
from pathlib import Path
import torch
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.compositional_evidence import CompositionalEvidence
from radio_gs.query_field.source_member_readout import read_source_members

ROOT=Path('output/query_field_20261010/source_member_readout_v1')
REFERENCE=Path('output/query_field_20261010/compositional_atoms_v2')
SOURCE=Path('output/query_field_20261009/text_union_functions_v1')
BASE=Path('output/query_field_20261009/union_admission_v1')
ARMS=['admission','atom_protected','source_parts','source_protected']
PROTOCOL=Path('docs/experiments/2026-10-10-source-member-readout-protocol.md')


def prepare():
    read(REFERENCE/'predictions_complete.json');ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(REFERENCE/'prediction_config.json'))
    cfg['status']='fixed source-normalized member readout; privileged acceptance diagnostic'
    write(ROOT/'prediction_config.json',cfg);records=[]
    for j in cfg['jobs']:
        paths=[Path(j['field'])/'manifest.json',Path(j['queries'])]
        paths += [root/'predictions'/j['name']/name for root in [SOURCE,BASE,REFERENCE] for name in ['prediction.json','predictions.pt']]
        paths += [REFERENCE/'predictions'/j['name']/name for name in ['compilation.json','atoms.pt']]
        paths += [Path(read(REFERENCE/'predictions'/j['name']/'prediction.json')['teacher_selection']['path'])]
        records.extend(receipt(p) for p in paths)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        inputs=records,code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arms=ARMS,
        primary='source_protected',source_assisted_diagnostic=True,training_steps=0,new_state_bytes=0,
        reused_atom_tensor_bytes=sum(read(REFERENCE/'predictions'/j['name']/'compilation.json')['tensor_bytes'] for j in cfg['jobs']),
        whole_five_task=False,earlier_development_scores_seen=True))


def design(verify_inputs=True):
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]+(d['inputs'] if verify_inputs else []):check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    parent=REFERENCE/'predictions'/j['name'];record=read(parent/'prediction.json')
    data=torch.load(parent/'predictions.pt',weights_only=True)
    fixture=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
    field,_=load_field(j['field']);available=field.observed|field.semantic_measure.supported
    for k,v in data['geometry'].items():assert torch.equal(v,getattr(field,k))
    assert data['texts']==fixture['texts']==[x['text'] for x in read(j['queries'])['queries']]
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    teacher=torch.load(record['teacher_selection']['path'],weights_only=True)
    assert teacher['texts']==data['texts'] and torch.equal(teacher['views'],field.evidence.views)
    accepted=((teacher['probability']>.5)&(mass[:,None]>0)).cuda();available=available.cuda();del field
    model=CompositionalEvidence(torch.load(parent/'atoms.pt',weights_only=True),'cuda')
    scores={a:data['scores'][a] for a in ARMS[:2]};scores.update(source_parts=[],source_protected=[]);diagnostics=[]
    for qi,text in enumerate(data['texts']):
        prior=fixture['scores']['measure_prior'][:,qi].cuda();prepared=model.prepare(accepted[:,qi])
        prepared[2].update(acceptance_provider='same_frozen_source_teacher',query_conditioned_source_access=True)
        plain,dp=read_source_members(prior,available,prepared)
        protected,ds=read_source_members(prior,available,prepared,protect=True)
        assert bool((protected>=plain).all())
        scores['source_parts'].append(plain.cpu());scores['source_protected'].append(protected.cpu())
        diagnostics.append(dict(text=text,controls=dict(source_parts=dp,source_protected=ds)))
        print('predicted',j['scene'],qi,flush=True)
    for a in ARMS[2:]:scores[a]=torch.stack(scores[a],1)
    selected={a:scores[a]>.5 for a in ARMS};data.update(scores=scores,selected=selected['source_protected'],selected_by_control=selected)
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False);torch.save(data,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(data['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],runtime_source_access=True,source_assisted_diagnostic=True,
        teacher_selection=record['teacher_selection'],compilation=record['compilation'],
        target_labels_opened=False,evaluation_scene_opened_by_predictor=False,field_unchanged=True,
        integrated_deployed_field=False,atom_state_unchanged=True))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_source_member_readout','predict','--index',str(i)],ROOT/'logs'/('predict_'+str(i)+'.log'),g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        assert r['configuration']==cfg and r['job']==j and not r['target_labels_opened']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=4))
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
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-10-source-member-readout-results.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='5');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
