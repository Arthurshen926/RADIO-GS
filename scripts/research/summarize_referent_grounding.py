"""Report frozen referring learners separately from native controls and oracles."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def main():
    root=Path('output/referent_grounding_20261005')
    baseline=Path('output/identity_bottleneck_20261005/baseline_system')
    read=lambda p:json.loads(p.read_text())
    systems={'native':baseline,**{name:root/name/'system' for name in ['local','scene']}}
    results={name:read(path/'uqis/identity_scope/result.json') for name,path in systems.items()}
    recipe=read(root/'scene_recipe.json')
    maps={j['name']:Path(j['scene']).parent for j in recipe['jobs'] if j['benchmark']=='uqis'}
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    dataset=read(package/'targets.json')
    report=dict(conditions={},fits={},independent_final_test=False,default_changed=False,
        extra_supervision='NR3D full expressions and official3D instances in disjoint training rooms',
        native_coordinates_changed=False,scope_family_changed=False,
        full_candidate_execution=read(root/'scene/system/summary.json')['macro'],
        fresh_scene_jobs=sum(read(systems[n]/'predictions_complete.json')['jobs'] for n in ['local','scene']))
    report['real_observation_control']=read(root/'real_observation_control/comparison.json')
    design=read(root/'design.json');capacity={}
    for split in ['fit','calibration','validation']:
        rooms=[]
        for scene in design[split]:
            folder=root/'maps'/scene
            if not (folder/'targets.npz').exists():continue
            target=np.load(folder/'targets.npz');membership=sparse.load_npz(folder/'membership.npz')
            labels=np.load(root/'labels'/(scene+'.npy'));observed=membership.getnnz(axis=1)>0
            area=np.asarray(membership.sum(0)).ravel()
            purity=target['region_intersection']/np.maximum(area[:,None],1)
            rooms.append(dict(scene=scene,regions=membership.shape[1],targets=len(target['instance_ids']),
                target_mean_covered=float(np.mean([observed[labels==i].mean() for i in target['instance_ids']])),
                majority_candidate_targets=int(np.count_nonzero(purity.max(0)>.5)),
                finite_scope_oracle=float(target['scope_iou'].max(0).mean())))
        capacity[split]=dict(rooms=len(rooms),targets=sum(r['targets'] for r in rooms),
            majority_candidate_targets=sum(r['majority_candidate_targets'] for r in rooms),
            scene_mean_target_coverage=float(np.mean([r['target_mean_covered'] for r in rooms])),
            scene_mean_finite_scope_oracle=float(np.mean([r['finite_scope_oracle'] for r in rooms])),scenes=rooms)
    (root/'training_carrier_capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    report['training_carrier_capacity']=capacity
    for name in ['local','scene']:
        fit=read(root/name/'fit.json')
        report['fits'][name]={k:fit[k] for k in ['architecture','epochs','selected_epoch',
            'training_listener_correct_records','excluded_training_zero_scope_support','parameter_count','seconds']}
        for group in ['calibration','validation']:
            d=fit[group]
            report['fits'][name][group]=dict(iou=d['scene_macro_iou'],native_iou=d['scene_macro_native_iou'],
                finite_family_oracle=float(np.mean([s['finite_family_oracle'] for s in d['scenes']])),
                queries=sum(s['queries'] for s in d['scenes']),rooms=len(d['scenes']))
        report['fits'][name]['training_fit_diagnosis']=read(root/name/'training_fit_diagnosis.json')
        report['fits'][name]['checkpoint_bytes']=(root/name/'model.pt').stat().st_size
    for name,result in results.items():
        metrics={}
        for modality in ['text','image','point_2d','point_3d']:
            selection=result['explicit_selection']['modalities'][modality]
            old=results['native']['explicit_selection']['modalities'][modality]
            metrics[modality]=dict(ap=result['modalities'][modality]['scene_macro']['average_precision'],
                **selection['scene_macro'],iou_delta_pp=scene_clustered_bootstrap({s:100*(v['iou']-
                    old['per_scene'][s]['iou']) for s,v in selection['per_scene'].items()}))
        cases=[]
        for domain in dataset['scene_domains']:
            scene=domain['scene_id'];labels=np.load(package/domain['mesh_instance_ids_path'])
            membership=sparse.load_npz(maps[scene]/'membership.npz').tocsc()
            run=read(systems[name]/'uqis'/scene/'query/registered/runs'/f'{scene}.json')
            detail={q['query_id']:q for q in run['query_details']}
            for target in dataset['targets']:
                if target['scene_id']!=scene:continue
                qid=target['queries']['text'];seed=detail[qid]['instance_identity']['seed_region']
                vertices=membership[:,seed].indices;truth=labels[vertices]==target['instance_id']
                ids,count=np.unique(labels[vertices],return_counts=True);dominant=int(ids[count.argmax()])
                kind=('correct' if dominant==target['instance_id'] else
                    'same_class_distractor' if dominant in target['same_class_distractor_instance_ids'] else 'other_or_unlabelled')
                cases.append(dict(scene_id=scene,query_id=qid,seed=seed,seed_purity=float(truth.mean()),
                    seed_majority_correct=bool(truth.mean()>.5),identity_error_type=kind))
        unchanged={}
        for modality in ['image','point_2d','point_3d']:
            ids=[t['queries'][modality] for t in dataset['targets']]
            unchanged[modality]=dict(selection_changes=sum(not np.array_equal(
                np.load(systems[name]/'uqis/identity_scope/selections'/f'{q}.npy'),
                np.load(baseline/'uqis/identity_scope/selections'/f'{q}.npy')) for q in ids),
                maximum_rank_difference=max(float(np.max(np.abs(
                    np.load(systems[name]/'uqis/identity_scope/predictions'/f'{q}.npy')-
                    np.load(baseline/'uqis/identity_scope/predictions'/f'{q}.npy')))) for q in ids))
        text_ids=[t['queries']['text'] for t in dataset['targets']]
        native_difference=max(float(np.max(np.abs(
            np.load(systems[name]/'uqis/native_semantic/predictions'/f'{q}.npy')-
            np.load(baseline/'uqis/native_semantic/predictions'/f'{q}.npy')))) for q in text_ids)
        report['conditions'][name]=dict(metrics=metrics,correct_majority_seeds=sum(r['seed_majority_correct'] for r in cases),
            dominant_identity={k:sum(r['identity_error_type']==k for r in cases) for k in
                ['correct','same_class_distractor','other_or_unlabelled']},
            all_four_iou_ge_05=sum(r['all_four_iou_at_least_050'] for r in result['explicit_selection']['paired_query_robustness']['per_target']),
            unchanged_query_controls=unchanged,native_semantic_maximum_difference=native_difference,cases=cases)
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:dict(text=r['metrics']['text'],identity=r['dominant_identity'],
        all_four=r['all_four_iou_ge_05']) for name,r in report['conditions'].items()},indent=2))


if __name__=='__main__':main()
