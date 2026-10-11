"""Correct one raw-N intervention by retaining the originating view's permission."""
import argparse,copy,subprocess,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from radio_gs.query_field.storage import load_field,sha256
from radio_gs.query_field.scope_topology import assemble_scope
from radio_gs.query_field.scope_activation import admitted_negative
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research.run_query_field_scope_topology import ROOT as PARENT,PARENT as REFERENCE,CANONICAL
from scripts.research.run_query_field_signed_scope import ROOT as RAW,hybrid_scope
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261012/scope_permission_v1')
ARM='native_positive_admitted_reference_negative';BASE='native_complete_scope';ARMS=[BASE,ARM]
PROTOCOL=Path('docs/experiments/2026-10-12-scope-permission-protocol.md')
CODE=[__file__,'radio_gs/query_field/scope_activation.py','scripts/research/run_query_field_signed_scope.py',
      'radio_gs/query_field/scope_topology.py','radio_gs/query_field/native_text_device.py',
      'radio_gs/query_field/extent_preparation.py','radio_gs/query_field/extent_decision.py',
      'radio_gs/query_field/extent_decision_linked.py','radio_gs/query_field/field.py','radio_gs/query_field/storage.py']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(PARENT/'prediction_config.json'));cfg.update(semantic_controls=ARMS,status='originating negative permission intervention; privileged only')
    for j in cfg['jobs']:j['field']=str(PARENT/'fields'/j['scene']/'field')
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in CODE],parent_design=receipt(PARENT/'design.json'),raw_design=receipt(RAW/'design.json'),
        raw_predictions_complete_before_design=(RAW/'predictions_complete.json').exists(),
        raw_scores_exist_before_design=bool(list((RAW/'predictions').glob('*/evaluation/evaluation.json'))),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),max_passes=4,
        correction_reason='raw reference N includes inactive views; permission must survive component substitution',
        privileged=True,default_promoted=False,whole_five_task=False))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d[k] for k in ['configuration','protocol','parent_design','raw_design']]:check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design();cfg=read(ROOT/'prediction_config.json');j=cfg['jobs'][index];scene=j['scene'];print('loading',scene,flush=True)
    oldroot=PARENT/'predictions'/j['name'];record=read(oldroot/'prediction.json')
    check(dict(path=str(oldroot/'predictions.pt'),sha256=record['predictions_sha256']));check(record['research_scopes'])
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    field,_=load_field(j['field']);data=torch.load(oldroot/'predictions.pt',weights_only=True)
    previous=torch.load(record['research_scopes']['path'],weights_only=True)
    fixed=torch.load(REFERENCE/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
    prior_reference=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
    native=NativeTextDevice(field,'cuda');topology=field.scope_topology.cuda();prep=field.extent_preparation.cuda()
    engine=LinkedExtentDecision(field.extent_factors,len(field.xyz));result=torch.zeros_like(prior_reference);rows=[];prepared=[]
    for qi,text in enumerate(data['texts']):
        began=time.monotonic();ss=[];permission=[]
        with patch('builtins.open',side_effect=AssertionError('IO')),patch('io.open',side_effect=AssertionError('IO')),patch('torch.load',side_effect=AssertionError('IO')):
            prior,_=native(data['native_query_vectors'][qi],data['generic_negative_vectors']);prior=prior.cuda()
            for vi,(v,pv) in enumerate(zip(topology.views,prep.views)):
                a,c,_=v.native_admission(data['native_query_vectors'][qi],data['generic_negative_vectors'])
                original=assemble_scope(v,pv,prior,a,c)
                for k,x in original.items():assert torch.equal(x.cpu(),previous[BASE][qi][vi][k])
                reference={k:fixed['views'][vi][k][:,qi].cuda() if k!='confidence' else fixed['views'][vi][k][qi].cuda()
                           for k in ['target','positive','negative','confidence']}
                negative=admitted_negative(reference);corrected,conflict=hybrid_scope(original,negative,c);ss.append(corrected)
                raw,raw_conflict=hybrid_scope(original,reference['negative'],c)
                active=bool(c>0 and original['positive'].any());ref_active=bool(reference['confidence']>0 and reference['positive'].any())
                permission.append(dict(reference_active=ref_active,native_active=active,conflicts=conflict,
                    removed_unpermitted_rays=int((raw['negative']&~corrected['negative']).sum()),
                    effective_removed_rays=int((raw['negative']&~corrected['negative']).sum()) if active else 0))
                # The other two crossed interventions retain exactly the same
                # EFFECTIVE evidence; their original all-query outputs remain valid.
                matched={k:x.cuda() for k,x in previous['matched_positive_scope'][qi][vi].items()}
                if bool(c>0 and matched['positive'].any()):
                    for neg_origin in [original,reference]:
                        raw_m,_=hybrid_scope(matched,neg_origin['negative'],c)
                        admitted_m,_=hybrid_scope(matched,admitted_negative(neg_origin),c)
                        assert torch.equal(raw_m['negative'],admitted_m['negative'])
            initial,cost,info=prep(prior,ss)
        error=float((prior.cpu()-prior_reference[:,qi]).abs().max());assert error<=2e-6
        assert torch.equal(prior.cpu()>.5,prior_reference[:,qi]>.5)
        assert info['active_views']==record['diagnostics'][qi]['arms'][BASE]['active_views']
        cpu=[{k:x.cpu() for k,x in s.items()} for s in ss];confidence=np.zeros(len(cpu))
        for vi in info['active_views']:confidence[vi]=float(cpu[vi]['confidence'])
        value=initial.cpu()
        if info['active_views']:
            selected,report=engine.solve((value>.5).numpy(),torch.cat([s['target'] for s in cpu]).numpy(),
                torch.cat([s['positive'] for s in cpu]).numpy(),torch.cat([s['negative'] for s in cpu]).numpy(),
                confidence,cost.cpu().numpy(),max_passes=d['max_passes']);value=torch.from_numpy(selected).float();info.update(report)
        assert not (value>.5)[~data['observed']].any();result[:,qi]=value
        prepared.append(dict(initial=initial.cpu(),cost=cost.cpu(),confidence=torch.from_numpy(confidence)))
        info.update(query=text,native_prior_max_error=error,permission=permission,other_factorial_arms_effective_scope_exact=True,
                    seconds=time.monotonic()-began);rows.append(info);write(out/'queries'/f'{qi:03d}.json',info);print(scene,qi,text,info['seconds'],flush=True)
    data.update(scores={a:data['scores'][BASE] for a in ARMS},membership_by_control={BASE:data['membership_by_control'][BASE],ARM:result},
                selected_by_control={BASE:data['selected_by_control'][BASE],ARM:result>.5},selected=result>.5)
    torch.save(data,out/'predictions.pt');torch.save(prepared,out/'prepared.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,queries=len(data['texts']),diagnostics=rows,
        predictions_sha256=sha256(out/'predictions.pt'),prepared=receipt(out/'prepared.pt'),
        parent_predictions=receipt(oldroot/'predictions.pt'),parent_scopes=record['research_scopes'],
        reference_scope=receipt(REFERENCE/'predictions'/j['name']/'privileged_scope.pt'),field_manifest=receipt(Path(j['field'])/'manifest.json'),
        target_labels_opened=False,new_arm_privileged=True,default_promoted=False))


def complete(gpus):
    design();cfg=read(ROOT/'prediction_config.json');records=[]
    for j in cfg['jobs']:
        out=ROOT/'predictions'/j['name'];r=read(out/'prediction.json')
        check(dict(path=str(out/'predictions.pt'),sha256=r['predictions_sha256']));assert not r['target_labels_opened'];records.append(receipt(out/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=ARMS))
    def score(j,g):
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
                 '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),g)
    lanes(cfg['jobs'],gpus,score);per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics'][BASE]
        for key in ['miou','count','missing']:assert m[BASE]['selected'][key]==old['selected'][key]
        for a in ARMS:
            for key in ['iou','ap','localization','observations']:assert m[a][key]==old[key]
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
         new_arm_privileged=True,default_promoted=False,whole_five_task=False,predictions_before_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','complete','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,3,4,5')
    a=p.parse_args();torch.set_num_threads(4);gpus=list(map(int,a.gpus.split(',')))
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    elif a.stage=='complete':complete(gpus)
    else:
        lanes(list(range(4)),gpus,lambda i,g:execute(['-m','scripts.research.run_query_field_scope_permission','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g))
        complete(gpus)
