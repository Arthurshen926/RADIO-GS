"""Signed abstention ablations and one source-free spatial completion candidate."""
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
from radio_gs.query_field.scope_completion import crossed_scope, complete_other_view_positive
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research.run_query_field_signed_scope import ROOT as PARENT, PARENT as TOPOLOGY, CANONICAL
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261012/scope_completion_v1')
PROTOCOL = Path('docs/experiments/2026-10-12-scope-completion-protocol.md')
BASE = 'native_atomic_abstention'
OLD_ARMS = ['seed_addition', 'native_complete_scope', BASE]
NEW_ARMS = ['atomic_positive_native_negative', 'native_positive_atomic_negative', 'other_view_positive_completion']
CANDIDATE = NEW_ARMS[-1]
ARMS = OLD_ARMS + NEW_ARMS
CODE = [__file__, 'radio_gs/query_field/scope_completion.py', 'radio_gs/query_field/scope_activation.py',
        'radio_gs/query_field/atomic_scope.py', 'radio_gs/query_field/scope_topology.py',
        'radio_gs/query_field/native_text_device.py', 'radio_gs/query_field/scope_confidence.py',
        'radio_gs/query_field/extent_preparation.py', 'radio_gs/query_field/extent_decision.py',
        'radio_gs/query_field/extent_decision_linked.py', 'radio_gs/query_field/field.py', 'radio_gs/query_field/storage.py']


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT/'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='signed abstention ablations and other-view spatial completion',
               arm_runtime_source_access={a:False for a in ARMS})
    write(ROOT/'prediction_config.json', cfg)
    write(ROOT/'design.json', dict(configuration=receipt(ROOT/'prediction_config.json'), protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in CODE], parent_design=receipt(PARENT/'design.json'),
        parent_audit=receipt(PARENT/'audit.json'),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        max_passes=4, default_promoted=False, whole_five_task=False, all_new_arms_source_free=True,
        parent_results_known=True, new_target_labels_used_before_scoring=False,
        primary=CANDIDATE, comparator=BASE, spatial_passes=1, float64=True,
        witness_mass_fraction=.5, numerical_epsilon=1e-8))


def design():
    d = read(ROOT/'design.json')
    for r in d['code'] + [d[k] for k in ['configuration','protocol','parent_design','parent_audit']]: check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design(); cfg=read(ROOT/'prediction_config.json'); j=cfg['jobs'][index]; scene=j['scene']
    print('loading',scene,flush=True)
    parent=PARENT/'predictions'/j['name']; record=read(parent/'prediction.json')
    for key in ['parent_scopes','atomic_scopes','field_manifest']: check(record[key])
    check(dict(path=str(parent/'predictions.pt'),sha256=record['predictions_sha256']))
    out=ROOT/'predictions'/j['name']; out.mkdir(parents=True,exist_ok=False)
    field,manifest=load_field(j['field'])
    data=torch.load(parent/'predictions.pt',weights_only=True)
    previous=torch.load(record['parent_scopes']['path'],weights_only=True)['native_complete_scope']
    previous_atomic=torch.load(record['atomic_scopes']['path'],weights_only=True)
    prior_reference=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
    native=NativeTextDevice(field,'cuda'); topology=field.scope_topology.cuda(); preparation=field.extent_preparation.cuda()
    engine=LinkedExtentDecision(field.extent_factors,len(field.xyz))
    members={a:torch.zeros_like(prior_reference) for a in NEW_ARMS}; rows=[]; prepared=[]; all_scopes=[]
    for qi,text in enumerate(data['texts']):
        began=time.monotonic(); standard=[]; atomic=[]
        with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
            prior,_=native(data['native_query_vectors'][qi],data['generic_negative_vectors']); prior=prior.cuda()
            for vi,(view,pv) in enumerate(zip(topology.views,preparation.views)):
                accepted,c,probability=view.native_admission(data['native_query_vectors'][qi],data['generic_negative_vectors'])
                ns=assemble_scope(view,pv,prior,accepted,c)
                atom,_=assemble_atomic_scope(view,pv,prior,probability,c)
                for k in ns:
                    assert torch.equal(ns[k].cpu(),previous[qi][vi][k]),(scene,qi,vi,k)
                    assert torch.equal(atom[k].cpu(),previous_atomic[qi][vi][k]),(scene,qi,vi,k)
                standard.append(ns); atomic.append(atom)
            scopes={a:[] for a in NEW_ARMS}; conflicts={a:[] for a in NEW_ARMS[:2]}
            for ns,atom in zip(standard,atomic):
                for arm,p,n in [(NEW_ARMS[0],atom,ns),(NEW_ARMS[1],ns,atom)]:
                    s,k=crossed_scope(p,n,ns['confidence']); scopes[arm].append(s); conflicts[arm].append(k)
            scopes[CANDIDATE],completion=complete_other_view_positive(preparation,atomic,standard)
        error=float((prior.cpu()-prior_reference[:,qi]).abs().max())
        assert error<=2e-6 and torch.equal(prior.cpu()>.5,prior_reference[:,qi]>.5)
        row=dict(query=text,native_prior_max_error=error,parent_scopes_exact=True,completion=completion,conflicts=conflicts,arms={})
        pquery={}; squery={}
        for arm in NEW_ARMS:
            with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
                initial,cost,info=preparation(prior,scopes[arm])
            ss=[{k:v.cpu() for k,v in s.items()} for s in scopes[arm]]
            confidence=np.zeros(len(ss))
            for vi in info['active_views']:confidence[vi]=float(ss[vi]['confidence'])
            value=initial.cpu()
            if info['active_views']:
                selected,report=engine.solve((value>.5).numpy(),torch.cat([s['target'] for s in ss]).numpy(),
                    torch.cat([s['positive'] for s in ss]).numpy(),torch.cat([s['negative'] for s in ss]).numpy(),
                    confidence,cost.cpu().numpy(),max_passes=d['max_passes'])
                value=torch.from_numpy(selected).float(); info.update(report)
            assert not (value>.5)[~data['observed']].any()
            members[arm][:,qi]=value;row['arms'][arm]=info
            pquery[arm]=dict(initial=initial.cpu(),cost=cost.cpu(),confidence=torch.from_numpy(confidence))
            squery[arm]=ss
        row['seconds']=time.monotonic()-began; rows.append(row);prepared.append(pquery);all_scopes.append(squery)
        write(out/'queries'/f'{qi:03d}.json',row)
        print(scene,qi,text,row['seconds'],{a:len(row['arms'][a]['active_views']) for a in NEW_ARMS},flush=True)
    data['membership_by_control']={**{a:data['membership_by_control'][a] for a in OLD_ARMS},**members}
    data['selected_by_control']={a:v>.5 for a,v in data['membership_by_control'].items()}
    data['scores']={a:data['scores'][BASE] for a in ARMS};data['selected']=data['selected_by_control'][CANDIDATE]
    torch.save(data,out/'predictions.pt');torch.save(prepared,out/'prepared.pt');torch.save(all_scopes,out/'scopes.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,queries=len(data['texts']),diagnostics=rows,
        predictions_sha256=sha256(out/'predictions.pt'),prepared=receipt(out/'prepared.pt'),scopes=receipt(out/'scopes.pt'),
        parent_predictions=receipt(parent/'predictions.pt'),parent_scopes=record['parent_scopes'],
        parent_atomic=record['atomic_scopes'],field_manifest=receipt(Path(j['field'])/'manifest.json'),
        target_labels_opened=False,all_new_arms_source_free=True,new_field_state_bytes=0,default_promoted=False))


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
    lanes(cfg['jobs'],gpus,score);per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            if a in OLD_ARMS:
                for k in ['miou','count','missing']:assert m[a]['selected'][k]==old[a]['selected'][k]
            for k in ['iou','ap','localization','observations']:assert m[a][k]==old[BASE][k]
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
        all_new_arms_source_free=True,default_promoted=False,whole_five_task=False,predictions_before_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','complete','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,3,4,5')
    a=p.parse_args();torch.set_num_threads(4);gpus=list(map(int,a.gpus.split(',')))
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    elif a.stage=='complete':complete(gpus)
    else:
        lanes(list(range(4)),gpus,lambda i,g:execute(['-m','scripts.research.run_query_field_scope_completion','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',g))
        complete(gpus)
