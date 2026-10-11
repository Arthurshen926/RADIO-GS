"""Independent cold field, scope intervention and actual-mask evaluation audit."""
import gc
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.scope_topology import assemble_scope
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_scope_topology import ROOT,PARENT,PARENT_FIELDS,ARMS,design
from scripts.research.run_query_field_relations import read,write,check,receipt
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(ROOT/'prediction_config.json');summary=read(ROOT/'summary.json')
    for r in d['inputs']:check(r)
    for r in read(ROOT/'predictions_complete.json')['records']:check(r)
    scenes={}
    for j in cfg['jobs']:
        scene=j['scene'];out=ROOT/'predictions'/j['name'];record=read(out/'prediction.json')
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']));check(record['research_scopes'])
        field,manifest=load_field(ROOT/'fields'/scene/'field')
        parent_manifest=read(PARENT_FIELDS/scene/'field/manifest.json')
        parent=torch.load(PARENT_FIELDS/scene/'field/field.pt',weights_only=True)
        state=field.state_dict()
        for k,v in parent.items():assert torch.equal(v,state[k]),k
        assert set(state)-set(parent)=={'scope_topology.'+k for k in field.scope_topology.state_dict()}
        parent_count=len(parent);del parent,state
        assert manifest['provenance']['parent_field_sha256']==parent_manifest['files']['field.pt']['sha256']
        assert manifest['external_scene_dependencies']==[]
        comp=read(ROOT/'fields'/scene/'compilation.json')
        assert not comp['queries_used'] and not comp['target_labels_used'] and comp['cold_reload_exact']
        assert comp['added_bytes']==manifest['tensor_bytes']-parent_manifest['tensor_bytes']
        assert comp['field_sha256']==manifest['files']['field.pt']['sha256']
        for r in comp['bindings']:
            for key in ['region_cache','sam_cache','compiled']:check(r[key])
            assert r['exact_composition_ray_tests']==10*r['rays'] and r['rays']==49152
        data=torch.load(out/'predictions.pt',weights_only=True)
        old=torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        fixed=torch.load(PARENT/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
        scopes=torch.load(out/'research_scopes.pt',weights_only=True)
        for a in ARMS[:2]:assert torch.equal(data['membership_by_control'][a],old['membership_by_control'][a])
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
            assert not data['selected_by_control'][a][~data['observed']].any()
            assert torch.equal(data['scores'][a],old['scores']['seed_addition'])
        # Independent NumPy atom composition for every privileged intervention.
        # Full-raster validation before source disposal is separately recorded above.
        conflicts=0;capacity=[]
        for qi,row in enumerate(record['diagnostics']):
            assert row['native_prior_max_error']<=2e-6 and row['native_prior_selection_disagreements']==0
            for vi,(v,pv) in enumerate(zip(field.scope_topology.views,field.extent_preparation.views)):
                cap=row['capacity'][vi];accepted=np.array(cap['accepted'],dtype=bool)
                assert accepted.shape==(v.incidence.shape[1],)
                occupied=np.any(v.incidence.numpy()[:,accepted],axis=1)
                neighbor=occupied[v.neighborhood.numpy()]
                target=neighbor[:,4];positive=neighbor.all(axis=1)&pv.valid.numpy()
                s=scopes['matched_positive_scope'][qi][vi];ref=fixed['views'][vi]
                assert np.array_equal(target,s['target'].numpy())
                assert np.array_equal(positive,s['positive'].numpy())
                expected_negative=ref['negative'][:,qi].numpy()&~target
                assert np.array_equal(expected_negative,s['negative'].numpy())
                assert float(s['confidence'])==float(ref['confidence'][qi])
                removed=int((ref['negative'][:,qi].numpy()&target).sum())
                assert removed==cap['replacement_target_negative_conflicts'];conflicts+=removed
                truth=(ref['target'][:,qi].numpy()>.5)&pv.valid.numpy()
                if cap['full']['greedy_iou'] is not None:
                    pred=target&pv.valid.numpy();iou=int((truth&pred).sum())/max(int((truth|pred).sum()),1)
                    assert abs(iou-cap['full']['greedy_iou'])<=1e-6
                    capacity.append(cap)
                for a in ARMS[2:]:
                    scope=scopes[a][qi][vi]
                    assert not (scope['positive']&scope['negative']).any()
                    assert not ((scope['positive']|scope['negative'])&~pv.valid).any()
            for a in ARMS[2:]:
                active=[vi for vi,s in enumerate(scopes[a][qi]) if bool(s['confidence']>0 and s['positive'].any())]
                assert active==row['arms'][a]['active_views']
                if active:assert np.all(np.diff(row['arms'][a]['energy_trace'])<=1e-8)
                else:
                    actual=data['membership_by_control'][a][:,qi]
                    reference=old['membership_by_control']['semantic_initial'][:,qi]
                    assert float((actual-reference).abs().max())<=2e-6
                    assert torch.equal(actual>.5,reference>.5)
        confidence_rows=[]
        for qi,row in enumerate(record['diagnostics']):
            active={a:row['arms'][a]['active_views'] for a in ARMS[2:]}
            common=sorted(set(active[ARMS[2]]) & set(active[ARMS[3]]))
            confidence_rows.append(dict(query=row['query'],common_views=len(common),
                mean_active={a:float(np.mean([float(scopes[a][qi][vi]['confidence']) for vi in ids])) if ids else None
                             for a,ids in active.items()},
                mean_common={a:float(np.mean([float(scopes[a][qi][vi]['confidence']) for vi in common])) if common else None
                             for a in ARMS[2:]}))
        # A fresh native scope is produced with only the cold field and legal
        # vectors. No privileged target, query-scoped detector or old prior input.
        probe=0;device_native=NativeTextDevice(field,'cuda')
        topology=field.scope_topology.cuda();preparation=field.extent_preparation.cuda()
        with patch('builtins.open',side_effect=AssertionError('cold native scope IO')), patch('io.open',side_effect=AssertionError('cold native scope IO')), patch('torch.load',side_effect=AssertionError('cold native scope IO')):
            prior,_=device_native(data['native_query_vectors'][probe],data['generic_negative_vectors'])
            for vi,(v,pv) in enumerate(zip(topology.views,preparation.views)):
                accepted,confidence,_=v.native_admission(data['native_query_vectors'][probe],data['generic_negative_vectors'])
                actual=assemble_scope(v,pv,prior.cuda(),accepted,confidence)
                for k,x in actual.items():assert torch.equal(x.cpu(),scopes['native_complete_scope'][probe][vi][k]),(scene,vi,k)
        del device_native,topology,preparation,field,old,scopes,prior,actual;gc.collect();torch.cuda.empty_cache()
        evaluation=read(out/'evaluation/evaluation.json');metrics={}
        for a in ARMS:
            rows=evaluation['metrics'][a]['selected']['objects'];ious=[];precisions=[];recalls=[]
            assert len(rows)==len({(r['frame'],r['query']) for r in rows})
            for r in rows:
                assert not r['missing']
                gt=load_binary_mask(Path(r['gt_path']),threshold=10)
                pred=load_binary_mask(Path(r['pred_path']),threshold=10,grayscale=True)
                tp=int((gt&pred).sum());iou=tp/max(int((gt|pred).sum()),1)
                assert abs(iou-r['iou'])<1e-12
                ious.append(iou);precisions.append(tp/max(int(pred.sum()),1));recalls.append(tp/max(int(gt.sum()),1))
            assert abs(float(np.mean(ious))-summary['per_scene'][a][scene])<1e-12
            metrics[a]=dict(iou=float(np.mean(ious)),precision=float(np.mean(precisions)),recall=float(np.mean(recalls)))
        mean=lambda values:float(np.mean(values)) if values else None
        scenes[scene]=dict(metrics=metrics,parent_tensors_exact=parent_count,compilation=comp,
            capacity_nonempty_views=len(capacity),capacity=dict(
                full_single=mean([r['full']['single_iou'] for r in capacity]),
                full_greedy=mean([r['full']['greedy_iou'] for r in capacity]),
                retained_greedy=mean([r['retained']['greedy_iou'] for r in capacity]),
                native=mean([r['native_iou'] for r in capacity])),
            replaced_negative_conflicts=conflicts,cold_native_scope_io_denied=True,
            confidence=confidence_rows,
            native_positive_views_without_reference_scope=sum(
                c['native_accepted']>0 and c['source_confidence']==0
                for r in record['diagnostics'] for c in r['capacity']),
            reference_views_without_native_admission=sum(
                c['native_accepted']==0 and c['source_confidence']>0
                for r in record['diagnostics'] for c in r['capacity']),
            active={a:sum(bool(r['arms'][a]['active_views']) for r in record['diagnostics']) for a in ARMS[2:]})
        print('audited',scene,metrics,flush=True)
        del data,fixed;gc.collect()
    macro={a:{k:sum(s['metrics'][a][k] for s in scenes.values())/4 for k in ['iou','precision','recall']} for a in ARMS}
    comparisons={a:{b:paired({s:r['metrics'][b]['iou'] for s,r in scenes.items()},
                            {s:r['metrics'][a]['iou'] for s,r in scenes.items()},'actual selected-set IoU')
                    for b in ARMS[:2]} for a in ARMS[2:]}
    result=dict(passed=True,scenes=scenes,macro=macro,comparisons=comparisons,code=receipt(__file__),
        design=receipt(ROOT/'design.json'),all_actual_mask_ious_replayed=True,matched_scope_privileged=True,
        native_scope_source_free=True,default_promoted=False,whole_five_task=False,
        scope_capacity_is_not_global_upper_bound=True)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-11-scope-topology-audit.json',result)
    print(macro,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
