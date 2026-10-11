"""Frozen four-scene real member-function study, predictions before GT scoring."""
import argparse,copy,subprocess,time
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder,CUTOFF
from radio_gs.query_field.member_energy import solve_member_energy
from radio_gs.query_field.scope_confidence import native_text_evidence,native_scope_confidence
from radio_gs.query_field.part_conditioning import native_region_probability
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
ROOT=Path('output/query_field_20261011/member_energy_v2')
PARENT=Path('output/query_field_20261011/joint_exclusion_domain_v1')
FIELDS=Path('output/query_field_20261011/compiled_support_footprint_v1')
ARMS=['semantic_initial','frozen_visibility','joint_coverage','member_energy','no_exchange','no_footprint','unit_confidence']
PROTOCOL=Path('docs/experiments/2026-10-11-member-energy-v2-protocol.md')


def prepare():
    assert read(FIELDS/'audit.json')['complete']
    assert read('output/query_field_20261011/member_energy_runtime_v1/compiled_runtime.json')['selected_exact']
    cfg=copy.deepcopy(read(PARENT/'prediction_config.json'))
    cfg.update(semantic_controls=ARMS,status='bounded member energy with explicit ability ablations',runtime_source_access=False)
    records=[receipt(FIELDS/'audit.json')]
    for j in cfg['jobs']:
        j['field']=str(FIELDS/j['scene']/'field')
        for p in [Path(j['field'])/'manifest.json',Path(j['queries']),PARENT/'predictions'/j['name']/'prediction.json',PARENT/'predictions'/j['name']/'predictions.pt']:
            records.append(receipt(p))
    for j in cfg['jobs'][:2]:
        previous=Path('output/query_field_20261011/member_energy_v1/predictions')/j['name']
        records.extend([receipt(previous/'prediction.json'),receipt(previous/'predictions.pt')])
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),inputs=records,
        code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))]+[receipt(__file__)],
        protocol=receipt(PROTOCOL),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        primary='member_energy',earlier_development_scores_seen=True,training=False,model_promotion=False,whole_five_task=False,
        policy=dict(cutoff=10.5/255,support_weight=.1,prior_weight=.1,max_passes=8,exchange_candidates=8,engine='compiled')))


def design():
    d=read(ROOT/'design.json')
    for r in d['inputs']+d['code']+[d['configuration'],d['protocol']]:check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design();cfg=read(d['configuration']['path']);j=cfg['jobs'][index]
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    field,_=load_field(j['field'],'cuda');decoder=JointExclusionDecoder(field)
    assert old['texts']==[q['text'] for q in read(j['queries'])['queries']]
    assert torch.equal(old['observed'],(field.observed|field.semantic_measure.supported).cpu())
    for k,v in old['geometry'].items():assert torch.equal(v,getattr(field,k).cpu())
    members={a:[] for a in ARMS};diagnostics=[]
    for qi,text in enumerate(old['texts']):
        start=time.monotonic();q=old['native_query_vectors'][qi].cuda();neg=old['generic_negative_vectors'].cuda()
        initial,probability=native_text_evidence(field,q,neg)
        accepted,confidence=native_scope_confidence(field,initial,probability)
        assert torch.equal(initial>.5,old['selected_by_control']['semantic_initial'][:,qi].cuda())
        assert float((initial.cpu()-old['scores']['semantic_initial'][:,qi]).abs().max())<=2e-6
        for a in ARMS[:3]:members[a].append(old['membership_by_control'][a][:,qi])
        positive,negative=decoder.labels(accepted);r=np.repeat(confidence.cpu().numpy(),decoder.slices)
        active=torch.stack([accepted[c.regions].any() for c in field.scope_constraints.views])
        footprint=field.support_footprint(active).cpu().numpy();prior=initial.cpu().numpy();controls={}
        for a in ARMS[3:]:
            selected,detail=solve_member_energy(decoder.matrix,prior,positive,negative,
                (r>0).astype(float) if a=='unit_confidence' else r,footprint,
                cutoff=-np.log1p(-CUTOFF),support_weight=0. if a=='no_footprint' else .1,
                prior_weight=.1,max_passes=8,exchange_candidates=0 if a=='no_exchange' else 8,engine='compiled')
            member=initial.cpu().clone();s=torch.from_numpy(selected);member[s & ~(member>.5)]=1.;member[~s & (member>.5)]=0.
            members[a].append(member);controls[a]=detail
            before=np.asarray(decoder.matrix@(prior>.5).astype(float)).ravel()
            after=np.asarray(decoder.matrix@selected.astype(float)).ravel();tau=-np.log1p(-CUTOFF)
            detail.update(sampled_positive_losses=int((positive&(before>tau)&(after<=tau)).sum()),
                sampled_positive_gains=int((positive&(before<=tau)&(after>tau)).sum()))
        diagnostics.append(dict(text=text,controls=controls,seconds=time.monotonic()-start))
        print('predicted',j['scene'],qi,text,'seconds',diagnostics[-1]['seconds'],
              'primary',controls['member_energy'],flush=True)
    members={a:torch.stack(v,1) for a,v in members.items()};sets={a:v>.5 for a,v in members.items()}
    previous=Path('output/query_field_20261011/member_energy_v1/predictions')/j['name']/'predictions.pt'
    if previous.exists():
        v1=torch.load(previous,weights_only=True)
        for a in ARMS:assert torch.equal(sets[a],v1['selected_by_control'][a]),a
        print('all seven arms exact v1',j['scene'],flush=True)
    old.update(scores={a:old['scores']['semantic_initial'] for a in ARMS},membership_by_control=members,
        selected=sets['member_energy'],selected_by_control=sets)
    torch.save(old,out/'predictions.pt')
    write(out/'prediction.json',dict(job=j,configuration=cfg,queries=len(old['texts']),diagnostics=diagnostics,
        predictions_sha256=receipt(out/'predictions.pt')['sha256'],target_labels_opened=False,
        evaluation_scene_opened_by_predictor=False,runtime_source_raster_access=False,
        teacher_acceptance_used=False,transient_query_fitting=False,persistent_field_updated=False))


def run(gpus):
    d=design();cfg=read(d['configuration']['path'])
    def action(i,g):execute(['-m','scripts.research.run_query_field_member_energy','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g)
    lanes(list(range(4)),gpus,action)
    records=[]
    for j in cfg['jobs']:
        out=ROOT/'predictions'/j['name'];r=read(out/'prediction.json')
        assert r['configuration']==cfg and not r['target_labels_opened']
        check(dict(path=str(out/'predictions.pt'),sha256=r['predictions_sha256']));records.append(receipt(out/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=len(ARMS)))
    def score(j,g):execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
        '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score)
    write(ROOT/'scoring_complete.json',dict(records=[receipt(ROOT/'predictions'/j['name']/'evaluation/evaluation.json') for j in cfg['jobs']]))
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        metrics=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        previous=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a,m in metrics.items():
            for k in ['iou','ap','localization','observations']:assert m[k]==previous['semantic_initial'][k]
            if a in ARMS[:3]:
                for k in ['miou','acc025','acc05','count','missing']:assert m['selected'][k]==previous[a]['selected'][k]
            per[a][j['scene']]=dict(response_iou=m['iou'],selection_iou=m['selected']['miou'],ap=m['ap'])
    report=dict(complete=True,macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS},
        per_scene=per,training=False,model_promotion=False,whole_five_task=False)
    write(ROOT/'summary.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='2');a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='predict':torch.cuda.set_per_process_memory_fraction(.60)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
