"""One fixed intervention: confidence-relative data and member-cost units."""
import argparse,copy,time,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from radio_gs.query_field.storage import load_field,sha256
from radio_gs.query_field.scope_topology import assemble_scope
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.relative_extent_decision import RelativeEvidenceExtentDecision
from scripts.research.run_query_field_scope_topology import ROOT as PARENT,ARMS as PREVIOUS,CANONICAL
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes

ROOT=Path('output/query_field_20261011/scope_relative_evidence_v1')
PROTOCOL=Path('docs/experiments/2026-10-11-relative-evidence-protocol.md')
ARM='native_relative_evidence';ARMS=[*PREVIOUS,ARM]
CODE=[__file__,'radio_gs/query_field/relative_extent_decision.py','radio_gs/query_field/extent_decision_linked.py',
      'radio_gs/query_field/extent_preparation.py','radio_gs/query_field/scope_topology.py',
      'radio_gs/query_field/native_text_device.py','radio_gs/query_field/field.py','radio_gs/query_field/storage.py']


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    cfg=copy.deepcopy(read(PARENT/'prediction_config.json'));cfg.update(semantic_controls=ARMS,status='fixed relative evidence intervention')
    write(ROOT/'prediction_config.json',cfg)
    write(ROOT/'design.json',dict(configuration=receipt(ROOT/'prediction_config.json'),protocol=receipt(PROTOCOL),
        code=[receipt(x) for x in CODE],parent_design=receipt(PARENT/'design.json'),
        parent_scored_before_design=(PARENT/'predictions_complete.json').exists(),
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        max_passes=4,default_promoted=False,scope_bits_must_match=True))


def design():
    d=read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol'],d['parent_design']]:check(r)
    return d


@torch.no_grad()
def predict(index):
    d=design();cfg=read(ROOT/'prediction_config.json');j=cfg['jobs'][index];scene=j['scene']
    oldroot=PARENT/'predictions'/j['name']
    # No parent evaluation or ground-truth file is read here.
    parent=read(oldroot/'prediction.json');check(dict(path=str(oldroot/'predictions.pt'),sha256=parent['predictions_sha256']));check(parent['research_scopes'])
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    field,manifest=load_field(PARENT/'fields'/scene/'field')
    data=torch.load(oldroot/'predictions.pt',weights_only=True)
    reference=torch.load(oldroot/'research_scopes.pt',weights_only=True)['native_complete_scope']
    prior_reference=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
    native=NativeTextDevice(field,'cuda');topology=field.scope_topology.cuda();preparation=field.extent_preparation.cuda()
    engine=RelativeEvidenceExtentDecision(field.extent_factors,len(field.xyz));result=torch.zeros_like(prior_reference);rows=[];prepared=[]
    for qi,text in enumerate(data['texts']):
        began=time.monotonic()
        with patch('builtins.open',side_effect=AssertionError('query IO')),patch('io.open',side_effect=AssertionError('query IO')),patch('torch.load',side_effect=AssertionError('query IO')):
            prior,_=native(data['native_query_vectors'][qi],data['generic_negative_vectors']);ss=[]
            prior=prior.cuda()
            for vi,(v,pv) in enumerate(zip(topology.views,preparation.views)):
                accepted,c,_=v.native_admission(data['native_query_vectors'][qi],data['generic_negative_vectors'])
                s=assemble_scope(v,pv,prior,accepted,c);ss.append(s)
                for k,value in s.items():assert torch.equal(value.cpu(),reference[qi][vi][k]),(scene,qi,vi,k)
            initial,cost,info=preparation(prior,ss)
        error=float((prior.cpu()-prior_reference[:,qi]).abs().max());assert error<=2e-6
        assert torch.equal(prior.cpu()>.5,prior_reference[:,qi]>.5)
        active=info['active_views'];confidence=np.zeros(len(ss))
        for vi in active:confidence[vi]=float(ss[vi]['confidence'])
        value=initial.cpu()
        if active:
            selected,report=engine.solve((value>.5).numpy(),torch.cat([s['target'] for s in ss]).cpu().numpy(),
                torch.cat([s['positive'] for s in ss]).cpu().numpy(),torch.cat([s['negative'] for s in ss]).cpu().numpy(),
                confidence,cost.cpu().numpy(),max_passes=d['max_passes'])
            value=torch.from_numpy(selected).float();info.update(report)
        assert not (value>.5)[~data['observed']].any()
        result[:,qi]=value;prepared.append(dict(initial=initial.cpu(),cost=cost.cpu(),confidence=torch.from_numpy(confidence)))
        info.update(query=text,scope_exact=True,native_prior_max_error=error,seconds=time.monotonic()-began)
        rows.append(info);write(out/'queries'/f'{qi:03d}.json',info);print(scene,qi,text,info['seconds'],flush=True)
    data['membership_by_control'][ARM]=result;data['selected_by_control'][ARM]=result>.5
    data['scores'][ARM]=data['scores']['native_complete_scope'];data['selected']=result>.5
    torch.save(data,out/'predictions.pt');torch.save(prepared,out/'prepared.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,queries=len(data['texts']),diagnostics=rows,
        predictions_sha256=sha256(out/'predictions.pt'),prepared=receipt(out/'prepared.pt'),
        parent_predictions=receipt(oldroot/'predictions.pt'),parent_scopes=parent['research_scopes'],
        field_manifest=receipt(PARENT/'fields'/scene/'field/manifest.json'),target_labels_opened=False,
        new_scope_recomputed_from_field=True,scope_exact=True,default_promoted=False))


def complete(gpus):
    design();cfg=read(ROOT/'prediction_config.json');records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));assert r['scope_exact'] and not r['target_labels_opened']
        records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=ARMS))
    def score(j,gpu):
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
            '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),gpu)
    lanes(cfg['jobs'],gpus,score)
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            if a!=ARM:assert m[a]==old[a]
            for k in ['iou','ap','localization','observations']:assert m[a][k]==old['native_complete_scope'][k]
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
        source_free_new_arm=True,default_promoted=False,whole_five_task=False,predictions_before_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','complete']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,5')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict(a.index)
    else:complete(list(map(int,a.gpus.split(','))))
