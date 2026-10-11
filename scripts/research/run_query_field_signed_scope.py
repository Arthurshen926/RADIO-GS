"""Separate positive scope and negative permission; test atomic abstention.

Three explicitly privileged interventions plus one source-free candidate.
Prediction code never opens evaluation labels. All four scenes precede scoring.
"""
import argparse
import copy
import subprocess
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from radio_gs.query_field.storage import load_field, sha256
from radio_gs.query_field.scope_topology import assemble_scope
from radio_gs.query_field.atomic_scope import assemble_atomic_scope
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research.run_query_field_scope_topology import ROOT as PARENT, PARENT as REFERENCE, CANONICAL, ARMS as OLD_ARMS
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261012/signed_scope_v1')
PROTOCOL = Path('docs/experiments/2026-10-12-signed-scope-protocol.md')
CANDIDATE = 'native_atomic_abstention'
# All three hybrid conditions use the identical NATIVE scalar confidence.
HYBRIDS = ['matched_positive_native_negative', 'native_positive_reference_negative', 'matched_positive_reference_negative']
NEW_ARMS = [CANDIDATE, *HYBRIDS]
ARMS = [*OLD_ARMS, *NEW_ARMS]
CODE = [__file__, 'radio_gs/query_field/atomic_scope.py', 'radio_gs/query_field/scope_topology.py',
        'radio_gs/query_field/native_text_device.py', 'radio_gs/query_field/scope_confidence.py',
        'radio_gs/query_field/extent_preparation.py', 'radio_gs/query_field/extent_decision.py',
        'radio_gs/query_field/extent_decision_linked.py', 'radio_gs/query_field/field.py', 'radio_gs/query_field/storage.py']


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT/'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='signed scope: fixed operational interventions and native atomic candidate',
               arm_runtime_source_access={a:a not in ['seed_addition','native_complete_scope',CANDIDATE] for a in ARMS})
    inputs=[]
    for j in cfg['jobs']:
        j['field']=str(PARENT/'fields'/j['scene']/'field')
        inputs += [receipt(PARENT/'predictions'/j['name']/'prediction.json'),
                   receipt(REFERENCE/'predictions'/j['name']/'privileged_scope.pt'),
                   receipt(CANONICAL/'inputs'/j['scene']/'response.pt'), receipt(Path(j['field'])/'manifest.json')]
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        code=[receipt(x) for x in CODE],inputs=inputs,parent_design=receipt(PARENT/'design.json'),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        max_passes=4,default_promoted=False,full_five=False,parent_results_known=True,
        new_target_labels_used_before_scoring=False,atomic_probability_threshold=.5,
        confidence_source='same native maximum positive margin for all four new arms'))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol'],d['parent_design']]:check(r)
    return d


def hybrid_scope(positive_source, negative_source, confidence):
    """Replace source functions and make introduced contradictions unknown.

    Positive target takes precedence solely for coherence of this intervention;
    this is not a claim that positive evidence should win in a deployed method.
    """
    conflict=negative_source & (positive_source['target']>.5)
    return dict(target=positive_source['target'],positive=positive_source['positive'],
                negative=negative_source & ~conflict,confidence=confidence),int(conflict.sum())


@torch.no_grad()
def predict(index):
    d=design();cfg=read(ROOT/'prediction_config.json');j=cfg['jobs'][index];scene=j['scene']
    print('loading',scene,flush=True)
    for r in d['inputs']:
        if scene in r['path']:check(r)
    parentroot=PARENT/'predictions'/j['name'];record=read(parentroot/'prediction.json')
    check(dict(path=str(parentroot/'predictions.pt'),sha256=record['predictions_sha256']));check(record['research_scopes'])
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    field,manifest=load_field(j['field'])
    data=torch.load(parentroot/'predictions.pt',weights_only=True)
    previous_scopes=torch.load(record['research_scopes']['path'],weights_only=True)
    fixed=torch.load(REFERENCE/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
    prior_reference=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
    assert data['texts']==fixed['texts']
    native=NativeTextDevice(field,'cuda');topology=field.scope_topology.cuda();prep=field.extent_preparation.cuda()
    engine=LinkedExtentDecision(field.extent_factors,len(field.xyz))
    results={a:torch.zeros_like(prior_reference) for a in NEW_ARMS};rows=[];prepared=[];atomic_scopes=[]
    for qi,text in enumerate(data['texts']):
        began=time.monotonic();ss={a:[] for a in NEW_ARMS};atomic_report=[];conflicts={a:[] for a in HYBRIDS}
        with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
            prior,_=native(data['native_query_vectors'][qi],data['generic_negative_vectors']);prior=prior.cuda()
            for vi,(view,pv) in enumerate(zip(topology.views,prep.views)):
                accepted,c,probability=view.native_admission(data['native_query_vectors'][qi],data['generic_negative_vectors'])
                original=assemble_scope(view,pv,prior,accepted,c)
                for k,v in original.items():assert torch.equal(v.cpu(),previous_scopes['native_complete_scope'][qi][vi][k]),(scene,qi,vi,k)
                atom,info=assemble_atomic_scope(view,pv,prior,probability,c)
                assert not (atom['positive'] & ~original['positive']).any()
                assert not (atom['negative'] & ~original['negative']).any() or not accepted.any()
                ss[CANDIDATE].append(atom);atomic_report.append(info)
                matched={k:v.cuda() for k,v in previous_scopes['matched_positive_scope'][qi][vi].items()}
                ref_negative=fixed['views'][vi]['negative'][:,qi].cuda()
                for arm,pos,neg in [(HYBRIDS[0],matched,original['negative']),
                                    (HYBRIDS[1],original,ref_negative),(HYBRIDS[2],matched,ref_negative)]:
                    mixed,conflict=hybrid_scope(pos,neg,c);ss[arm].append(mixed);conflicts[arm].append(conflict)
        error=float((prior.cpu()-prior_reference[:,qi]).abs().max())
        assert error<=2e-6 and torch.equal(prior.cpu()>.5,prior_reference[:,qi]>.5)
        row=dict(query=text,native_prior_max_error=error,native_scope_exact=True,atomic=atomic_report,conflicts=conflicts,arms={})
        pquery={}
        for arm in NEW_ARMS:
            with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
                initial,cost,info=prep(prior,ss[arm])
            cpu=[{k:v.cpu() for k,v in s.items()} for s in ss[arm]]
            confidence=np.zeros(len(cpu))
            for vi in info['active_views']:confidence[vi]=float(cpu[vi]['confidence'])
            value=initial.cpu()
            if info['active_views']:
                selected,report=engine.solve((value>.5).numpy(),torch.cat([s['target'] for s in cpu]).numpy(),
                    torch.cat([s['positive'] for s in cpu]).numpy(),torch.cat([s['negative'] for s in cpu]).numpy(),
                    confidence,cost.cpu().numpy(),max_passes=d['max_passes'])
                value=torch.from_numpy(selected).float();info.update(report)
            assert not (value>.5)[~data['observed']].any()
            results[arm][:,qi]=value;row['arms'][arm]=info
            pquery[arm]=dict(initial=initial.cpu(),cost=cost.cpu(),confidence=torch.from_numpy(confidence))
            if arm==CANDIDATE:atomic_scopes.append(cpu)
        prepared.append(pquery);row['seconds']=time.monotonic()-began;rows.append(row)
        write(out/'queries'/f'{qi:03d}.json',row);print(scene,qi,text,row['seconds'],{a:len(row['arms'][a]['active_views']) for a in NEW_ARMS},flush=True)
    for arm in NEW_ARMS:
        data['membership_by_control'][arm]=results[arm];data['selected_by_control'][arm]=results[arm]>.5
        data['scores'][arm]=data['scores']['native_complete_scope']
    data['selected']=results[CANDIDATE]>.5
    torch.save(data,out/'predictions.pt');torch.save(prepared,out/'prepared.pt');torch.save(atomic_scopes,out/'atomic_scopes.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,queries=len(data['texts']),diagnostics=rows,
        predictions_sha256=sha256(out/'predictions.pt'),prepared=receipt(out/'prepared.pt'),atomic_scopes=receipt(out/'atomic_scopes.pt'),
        parent_predictions=receipt(parentroot/'predictions.pt'),parent_scopes=record['research_scopes'],
        reference_scope=receipt(REFERENCE/'predictions'/j['name']/'privileged_scope.pt'),field_manifest=receipt(Path(j['field'])/'manifest.json'),
        target_labels_opened=False,candidate_source_free=True,hybrid_arms_privileged=HYBRIDS,new_field_state_bytes=0,default_promoted=False))


def complete(gpus):
    design();cfg=read(ROOT/'prediction_config.json');records=[]
    for j in cfg['jobs']:
        out=ROOT/'predictions'/j['name'];r=read(out/'prediction.json')
        check(dict(path=str(out/'predictions.pt'),sha256=r['predictions_sha256']))
        assert not r['target_labels_opened'];records.append(receipt(out/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=ARMS,new_controls=NEW_ARMS))
    def score(j,gpu):
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
                 '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),gpu)
    lanes(cfg['jobs'],gpus,score)
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            if a in OLD_ARMS:
                for k in ['miou','count','missing']:assert m[a]['selected'][k]==old[a]['selected'][k]
            for k in ['iou','ap','localization','observations']:assert m[a][k]==old['native_complete_scope'][k]
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
        new_candidate_source_free=True,hybrids_privileged=HYBRIDS,default_promoted=False,whole_five_task=False,predictions_before_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','complete','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,3,4,5')
    a=p.parse_args();torch.set_num_threads(4);gpus=list(map(int,a.gpus.split(',')))
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    elif a.stage=='complete':complete(gpus)
    else:
        lanes(list(range(4)),gpus,lambda i,g:execute(['-m','scripts.research.run_query_field_signed_scope','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g))
        complete(gpus)
