"""Cold field/input audit and independent float64 mass/cost reconstruction."""
import gc
import argparse
import time
from pathlib import Path
import torch
from unittest.mock import patch
from scripts.research.run_query_field_extent_preparation import ROOT, FIELDS, PARENT, PARENT_FIELDS, CANONICAL, design, query_scopes
from scripts.research.run_query_field_relations import read, write, receipt, check
from radio_gs.query_field.storage import load_field


def main(wait_seconds=0):
    began=time.monotonic();d=design();cfg=read(ROOT/'prediction_config.json')
    deadline=began+wait_seconds
    for record in d['inputs']:check(record)
    records=[]
    for j in cfg['jobs']:
        scene=j['scene'];out=ROOT/'predictions'/j['name']
        while not (out/'prediction.json').exists() and time.monotonic()<deadline:time.sleep(5)
        pred=read(out/'prediction.json');check(pred['predictions'])
        field,manifest=load_field(FIELDS/scene/'field')
        old_manifest=read(PARENT_FIELDS/scene/'field/manifest.json')
        old_state=torch.load(PARENT_FIELDS/scene/'field/field.pt',weights_only=True)
        current=field.state_dict()
        assert manifest['external_scene_dependencies']==[]
        assert manifest['provenance']['parent_field_sha256']==old_manifest['files']['field.pt']['sha256']
        assert set(current)-set(old_state)=={'extent_preparation.'+k for k in field.extent_preparation.state_dict()}
        for k,v in old_state.items():assert torch.equal(v,current[k]),(scene,k)
        parent_tensors=len(old_state);del old_state,current
        comp=read(FIELDS/scene/'compilation.json');assert comp['complete']
        assert comp['field_sha256']==manifest['files']['field.pt']['sha256']
        assert not comp['queries_used'] and not comp['target_labels_used']
        assert comp['added_bytes']==manifest['tensor_bytes']-old_manifest['tensor_bytes']
        observations=read(Path(j['observations'])/'observations.json')
        for r in comp['checks']:
            assert r['source_sha256']==next(x for x in observations['frames'] if x['frame']==r['frame'])['compiled']['sha256']
            check(dict(path=str(Path(j['observations'])/f"view_{r['frame']:05d}.pt"),sha256=r['source_sha256']))
        new=torch.load(out/'predictions.pt',weights_only=True)
        old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert new['texts']==old['texts'] and torch.equal(new['selected'],old['selected_by_control']['extent_discrete'])
        assert torch.equal(new['membership'],old['membership_by_control']['extent_discrete'])
        assert not new['selected'][~old['observed']].any()
        prior=torch.load(CANONICAL/'inputs'/scene/'response.pt',weights_only=True)['semantic_probability']
        scope=torch.load(PARENT/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
        probe=next(qi for qi,p in enumerate(new['prepared']) if p['active_views'])
        with patch('builtins.open',side_effect=AssertionError('file access')), patch('io.open',side_effect=AssertionError('file access')), patch('torch.load',side_effect=AssertionError('state access')):
            probe_initial,probe_cost,_=field.extent_preparation(prior[:,probe],query_scopes(scope,probe))
        assert float((probe_initial-new['prepared'][probe]['initial']).abs().max())<=d['initializer_atol']
        assert float(((probe_cost-new['prepared'][probe]['cost']).abs()/probe_cost.abs().clamp_min(1e-30)).max())<=d['cost_rtol']
        # Audit all queries together, independently in float64 on CPU. The
        # original prediction function was scalar float32 CUDA per query.
        q=len(new['texts']);positive=torch.zeros(len(field.xyz),q,dtype=torch.float64)
        known=torch.zeros_like(positive);raw_cost=torch.zeros_like(positive)
        active=[]
        for vi,(view,s) in enumerate(zip(field.extent_preparation.views,scope['views'])):
            enabled=(s['confidence']>0)&s['positive'].any(0)
            active.append(enabled)
            weight=(s['positive'].double()+s['negative'].double())*s['confidence'].double()[None]*enabled[None]
            op=torch.sparse_coo_tensor(view.index,view.weight.double(),(len(view.valid),len(field.xyz))).coalesce()
            known+=torch.sparse.mm(op.T,weight)
            positive+=torch.sparse.mm(op.T,weight*s['target'].double())
            raw_cost+=view.footprint.double()[:,None]*s['confidence'].double()[None]*enabled[None]
        reconstructed=torch.where(known>1e-8,positive/known.clamp_min(1e-8),prior.double())
        normalized=raw_cost/(raw_cost*reconstructed).sum(0).clamp_min(1e-8)[None]
        errors=[]
        for qi,p in enumerate(new['prepared']):
            expected=[vi for vi,a in enumerate(active) if bool(a[qi])];assert expected==p['active_views']
            ie=float((p['initial'].double()-reconstructed[:,qi]).abs().max())
            ce=float(((p['cost'].double()-normalized[:,qi]).abs()/normalized[:,qi].abs().clamp_min(1e-30)).max())
            assert ie<=d['initializer_atol'] and ce<=d['cost_rtol'],(scene,qi,ie,ce)
            assert torch.equal(p['initial'][known[:,qi]<=1e-8],prior[:,qi][known[:,qi]<=1e-8])
            assert not p['cost'][~old['observed']].any()
            if expected:assert abs(float(p['cost'].double()@p['initial'].double())-1)<2e-5
            else:assert torch.equal(new['membership'][:,qi],prior[:,qi]) and not p['cost'].any()
            errors.append(dict(initial_max_absolute=ie,cost_max_relative=ce))
        record=dict(scene=scene,queries=q,parent_tensors_exact=parent_tensors,final_membership_and_selection_exact=True,
                    added_bytes=comp['added_bytes'],rays=sum(r['rays'] for r in comp['checks']),
                    hits=sum(r['hits'] for r in comp['checks']),compilation_seconds=comp['seconds'],
                    independent_float64_errors=errors,field=receipt(FIELDS/scene/'field/manifest.json'),
                    predictions=pred['predictions'],preparation_file_access_denied=pred['preparation_file_access_denied'],
                    cold_probe_builtin_io_open_and_load_denied=True,
                    active=sum(bool(p['active_views']) for p in new['prepared']),fallback=sum(not p['active_views'] for p in new['prepared']))
        records.append(record);print('audit',scene,q,'PASS',flush=True)
        del field,new,old,prior,scope,positive,known,raw_cost,reconstructed,normalized,op,weight
        gc.collect()
    previous=read(PARENT/'summary.json')
    report=dict(complete=True,audit='PASS',records=records,queries=sum(r['queries'] for r in records),
        active=sum(r['active'] for r in records),fallback=sum(r['fallback'] for r in records),
        final_selection_disagreements=0,added_bytes=sum(r['added_bytes'] for r in records),
        inherited_metrics_by_exact_prediction_identity=previous,
        privileged_scope=True,privileged_initializer=False,privileged_cost=False,
        whole_five_task=False,standard_deployment=False,default_promoted=False,new_ground_truth_reads=False,
        audit_code=receipt(__file__),prediction_design=receipt(ROOT/'design.json'),seconds=time.monotonic()-began)
    write(ROOT/'audit.json',report)
    write('docs/experiments/2026-10-11-extent-preparation-audit.json',report)
    write(ROOT/'summary.json',dict(complete=True,queries=report['queries'],final_selection_disagreements=0,
        added_bytes=report['added_bytes'],macro=previous['macro'],per_scene=previous['per_scene'],
        metrics_inherited_by_exact_prediction_identity=True,privileged_scope=True,
        privileged_initializer=False,privileged_cost=False,standard_deployment=False,default_promoted=False,
        audit=receipt(ROOT/'audit.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4);main(a.wait_seconds)
