"""Separate a label-assisted diagnosis from complete automatic method results."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def main():
    root=Path('output/identity_bottleneck_20261005')
    read=lambda path:json.loads(path.read_text())
    conditions=['baseline','unmasked','object_context']
    results={name:read(root/(name+'_system')/'uqis/identity_scope/result.json') for name in conditions}
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator');dataset=read(package/'targets.json')
    output=dict(diagnosis=read(root/'seed_counterfactuals.json')['summary'],conditions={},
        independent_final_test=False,source_images_read_at_query=False,parameters_fit_on_benchmark=False,
        candidate_rules_fixed_before_automatic_scoring=True,public_tasks_rerun=False)
    for condition,result in results.items():
        system=root/(condition+'_system');metrics={};cases=[]
        for modality in ['text','image','point_2d','point_3d']:
            selected=result['explicit_selection']['modalities'][modality]
            baseline=results['baseline']['explicit_selection']['modalities'][modality]
            metrics[modality]=dict(ap=result['modalities'][modality]['scene_macro']['average_precision'],
                **selected['scene_macro'],iou_delta_pp=scene_clustered_bootstrap({scene:100*(v['iou']-
                    baseline['per_scene'][scene]['iou']) for scene,v in selected['per_scene'].items()}))
        for domain in dataset['scene_domains']:
            scene=domain['scene_id'];labels=np.load(package/domain['mesh_instance_ids_path'])
            membership=sparse.load_npz(root/'maps'/scene/'membership.npz').tocsc()
            area=np.asarray(membership.sum(0)).ravel()
            run=read(system/'uqis'/scene/'query/registered/runs'/f'{scene}.json')
            details={q['query_id']:q for q in run['query_details']}
            for target in dataset['targets']:
                if target['scene_id']!=scene:continue
                qid=target['queries']['text'];seed=details[qid]['instance_identity']['seed_region']
                truth=labels==target['instance_id'];vertices=membership[:,seed].indices
                tp=int(truth[vertices].sum())
                ids,count=np.unique(labels[vertices],return_counts=True);dominant=int(ids[count.argmax()])
                kind=('correct' if dominant==target['instance_id'] else
                    'same_class_distractor' if dominant in target['same_class_distractor_instance_ids'] else 'other_or_unlabelled')
                cases.append(dict(scene_id=scene,query_id=qid,seed=seed,seed_purity=tp/max(area[seed],1),
                    seed_coverage=tp/max(int(truth.sum()),1),seed_majority_correct=bool(tp>.5*area[seed]),
                    dominant_instance=dominant,identity_error_type=kind))
        unchanged={}
        for modality in ['image','point_2d','point_3d']:
            ids=[t['queries'][modality] for t in dataset['targets']]
            unchanged[modality]=dict(selection_changes=sum(not np.array_equal(np.load(system/'uqis/identity_scope/selections'/f'{q}.npy'),
                np.load(root/'baseline_system/uqis/identity_scope/selections'/f'{q}.npy')) for q in ids),
                maximum_rank_difference=max(float(np.max(np.abs(np.load(system/'uqis/identity_scope/predictions'/f'{q}.npy')-
                np.load(root/'baseline_system/uqis/identity_scope/predictions'/f'{q}.npy')))) for q in ids))
        output['conditions'][condition]=dict(metrics=metrics,correct_text_majority_seeds=sum(r['seed_majority_correct'] for r in cases),
            dominant_seed_identity={kind:sum(r['identity_error_type']==kind for r in cases)
                for kind in ['correct','same_class_distractor','other_or_unlabelled']},
            all_four_iou_ge_05=sum(r['all_four_iou_at_least_050'] for r in result['explicit_selection']['paired_query_robustness']['per_target']),
            controls=unchanged,cases=cases)
    builds=[read(p) for p in sorted((root/'context_maps').glob('*/build.json'))]
    output['build']=dict(source_frames=sum(r['source_frames'] for r in builds),regions=sum(r['regions'] for r in builds),
        sum_scene_build_wall_seconds=sum(r['seconds'] for r in builds),
        timing_note='Scene wall times overlap across GPUs; additional parallel cache helper not separately timed',
        extra_descriptor_bytes=sum(p.stat().st_size for p in (root/'context_maps').glob('*/*descriptors.npy')
            if p.name in ['unmasked_descriptors.npy','context_descriptors.npy']))
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({c:{k:v for k,v in r.items() if k!='cases'} for c,r in output['conditions'].items()},indent=2))


if __name__=='__main__':main()
