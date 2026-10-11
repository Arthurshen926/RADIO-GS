"""Migrate observation lift and normalized footprint; retain fixed scope controls."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.query_field.extent_preparation import ExtentPreparation, compile_extent_preparation_view
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from scripts.research.compile_query_field_extent_factors import phase_pixels, ROOT as PARENT_FIELDS
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261011/extent_preparation_v1')
FIELDS = ROOT/'fields'
PARENT = Path('output/query_field_20261011/extent_decision_v3')
CANONICAL = Path('output/query_field_20261011/canonical_subset_reference_v1')
PROTOCOL = Path('docs/experiments/2026-10-11-extent-preparation-protocol.md')
CODE = [__file__, 'radio_gs/query_field/extent_preparation.py', 'radio_gs/query_field/field.py',
        'radio_gs/query_field/storage.py', 'radio_gs/query_field/support_footprint.py',
        'radio_gs/query_field/extent_decision.py', 'radio_gs/query_field/extent_decision_linked.py',
        'radio_gs/rendering/contribution_compositor.py', 'radio_gs/querying/typed_evidence.py',
        'radio_gs/pipeline/subset.py']


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT/'prediction_config.json'))
    write(ROOT/'prediction_config.json', cfg)
    inputs = [receipt(PARENT/'summary.json'), receipt(PARENT/'audit.json')]
    for j in cfg['jobs']:
        p = PARENT/'predictions'/j['name']
        inputs += [receipt(x) for x in [p/'predictions.pt', p/'prediction.json', p/'privileged_scope.pt',
                   CANONICAL/'predictions'/j['name']/'admitted_optimization_state.pt',
                   CANONICAL/'inputs'/j['scene']/'response.pt',
                   PARENT_FIELDS/j['scene']/'field/manifest.json', Path(j['construction']),
                   Path(j['observations'])/'observations.json']]
    write(ROOT/'design.json', dict(configuration=receipt(ROOT/'prediction_config.json'), protocol=receipt(PROTOCOL),
          code=[receipt(p) for p in CODE], inputs=inputs,
          implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          privileged_scope=True, privileged_initializer=False, privileged_cost=False,
          max_passes=4, default_promoted=False, initializer_atol=2e-6, cost_rtol=2e-5,
          required_final_selection_disagreements=0))


def design():
    d = read(ROOT/'design.json')
    for r in d['code']+[d['configuration'],d['protocol']]: check(r)
    return d


def compile_scene(index):
    d = design(); j = read(ROOT/'prediction_config.json')['jobs'][index]
    out = FIELDS/j['scene']; out.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    field, parent = load_field(PARENT_FIELDS/j['scene']/'field')
    domain = field.observed | field.semantic_measure.supported
    geo = SimpleNamespace(**{k: getattr(field,k).cuda() for k in ['xyz','quaternions','scales','opacities']},
                          observed=domain.cuda(), world_units_per_metre=field.world_units_per_metre)
    construction = read(j['construction']); observations = read(Path(j['observations'])/'observations.json')
    frames = [r['frame'] for r in construction['source_records']]
    assert len(frames) == 18 and set(frames) <= set(observations['splits']['train_frames'])
    assert not set(frames) & set(observations['splits']['development_frames'])
    state = {}; records = []
    for vi, frame in enumerate(frames):
        path = Path(j['observations'])/f'view_{frame:05d}.pt'
        digest = sha256(path); entry = next(r for r in observations['frames'] if r['frame'] == frame)
        assert digest == entry['compiled']['sha256']
        raw = torch.load(path, weights_only=True); h,w = raw['original_hw']
        camera = SimpleNamespace(intrinsic=raw['intrinsic'], world_to_camera=raw['world_to_camera'], height=h, width=w)
        del raw
        values = compile_extent_preparation_view(geo,camera,384,512,phase_pixels(vi),member_domain=geo.observed)
        assert len(values['valid']) == len(field.extent_factors.views[vi].atom)
        assert not values['footprint'][~domain].any()
        state.update({f'views.{vi}.'+k:v for k,v in values.items()})
        records.append(dict(frame=frame, source_sha256=digest, rays=len(values['valid']),
                            hits=len(values['weight']), valid=int(values['valid'].sum())))
        print('compiled',j['scene'],vi,records[-1],flush=True)
    field.extent_preparation = ExtentPreparation(state,len(field.xyz))
    added = sum(v.numel()*v.element_size() for v in state.values())
    manifest = save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        queries_used=False,target_labels_used=False,cameras_in_export=False,query_scopes_in_export=False,
        query_initializers_in_export=False,source_records=records,default_decoder_changed=False))
    assert manifest['tensor_bytes'] == parent['tensor_bytes']+added
    write(out/'compilation.json',dict(complete=True,scene=j['scene'],added_bytes=added,checks=records,
        field_sha256=manifest['files']['field.pt']['sha256'],seconds=time.monotonic()-began,
        protocol=d['protocol'],queries_used=False,target_labels_used=False))


def query_scopes(evidence, qi):
    return [{k:v[k][:,qi] if k != 'confidence' else v[k][qi] for k in
             ('target','positive','negative','confidence')} for v in evidence['views']]


def predict(index):
    d=design();j=read(ROOT/'prediction_config.json')['jobs'][index]; scene=j['scene']
    out=ROOT/'predictions'/j['name'];out.mkdir(parents=True,exist_ok=False)
    previous=PARENT/'predictions'/j['name']
    field,manifest=load_field(FIELDS/scene/'field')
    evidence=torch.load(previous/'privileged_scope.pt',weights_only=True)
    response=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)
    old=torch.load(previous/'predictions.pt',weights_only=True)
    fixed=torch.load(CANONICAL/'predictions'/j['name']/'admitted_optimization_state.pt',weights_only=True)
    assert response['texts']==evidence['texts']==old['texts']
    assert evidence['source_raster']==[384,512]
    assert evidence['source_frames']==[r['frame'] for r in read(j['construction'])['source_records']]
    assert torch.equal(response['observed'],field.observed|field.semantic_measure.supported)
    for k,v in old['geometry'].items():assert torch.equal(v,getattr(field,k))
    module=field.extent_preparation.cuda()
    decoder=LinkedExtentDecision(field.extent_factors,len(field.xyz))
    result=response['semantic_probability'].clone();details=[];prepared=[]
    for qi,text in enumerate(old['texts']):
        began=time.monotonic();scopes=query_scopes(evidence,qi)
        prior=response['semantic_probability'][:,qi].cuda()
        # This function has no source file access, optimizer or saved query state.
        with patch('builtins.open',side_effect=AssertionError('preparation file access')), patch('torch.load',side_effect=AssertionError('preparation load')):
            initial,cost,info=module(prior,scopes)
        initial=initial.cpu();cost=cost.cpu();active=info['active_views']
        if qi in fixed:
            ref=fixed[qi];assert active==ref['view_indices']
            initial_error=float((initial-ref['start_membership']).abs().max())
            init_diff=int(((initial>.5)!=(ref['start_membership']>.5)).sum())
            refcost=ref['support_cost'];assert torch.equal(cost==0,refcost==0)
            cost_error=float(((cost-refcost).abs()/refcost.abs().clamp_min(1e-30)).max())
            info.update(initial_max_error=initial_error,initial_selection_disagreements=init_diff,cost_max_relative_error=cost_error)
            write(out/'latest_check.json',dict(query=text,**info))
            assert initial_error<=d['initializer_atol'] and init_diff==0 and cost_error<=d['cost_rtol'],info
            confidence=np.zeros(len(scopes))
            for vi in active:confidence[vi]=float(scopes[vi]['confidence'])
            selected,report=decoder.solve((initial>.5).numpy(),
                torch.cat([s['target'] for s in scopes]).numpy(),torch.cat([s['positive'] for s in scopes]).numpy(),
                torch.cat([s['negative'] for s in scopes]).numpy(),confidence,cost.numpy(),max_passes=d['max_passes'])
            result[:,qi]=torch.from_numpy(selected).float()
            info.update(report)
        else:
            assert not active and torch.equal(initial,prior.cpu()) and not cost.any()
        diff=int(((result[:,qi]>.5)!=old['selected_by_control']['extent_discrete'][:,qi]).sum())
        info.update(query=text,final_selection_disagreements=diff,seconds=time.monotonic()-began)
        details.append(info);prepared.append(dict(initial=initial,cost=cost,active_views=active))
        write(out/'progress.json',details)
        print(scene,qi,text,'difference',diff,'seconds',info['seconds'],flush=True)
        assert diff==d['required_final_selection_disagreements'],info
    torch.save(dict(texts=old['texts'],membership=result,selected=result>.5,prepared=prepared),out/'predictions.pt')
    write(out/'prediction.json',dict(scene=scene,queries=len(old['texts']),diagnostics=details,
        predictions=receipt(out/'predictions.pt'),field_manifest=receipt(FIELDS/scene/'field/manifest.json'),
        scope=receipt(previous/'privileged_scope.pt'),preparation_file_access_denied=True,
        scope_still_privileged=True,source_initializer_loaded_for_assertions_only=True,
        target_labels_opened=False,standard_deployment=False))


def run(gpus):
    def job(i,gpu):
        execute(['-m','scripts.research.run_query_field_extent_preparation','compile','--index',str(i)],ROOT/'logs'/f'compile_{i}.log',gpu)
        execute(['-m','scripts.research.run_query_field_extent_preparation','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',gpu)
    lanes(list(range(4)),gpus,job)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','predict','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,5')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='compile':compile_scene(a.index)
    elif a.stage=='predict':predict(a.index)
    else:run(list(map(int,a.gpus.split(','))))
