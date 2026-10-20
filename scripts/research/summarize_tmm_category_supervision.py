"""Publish corpus completion separately from learned or benchmark results."""
import json
from pathlib import Path
import numpy as np
from radio_gs.query_field.bounded_experiment import tree_bytes
from radio_gs.querying.response_study_adapters import read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def main():
    native=STUDY/'category_native_responses_v1'; source=STUDY/'category_utility_supervision_v1'
    manifest=read(native/'manifest.json'); rooms=sum(manifest['splits'].values(),[])
    reports=[read(source/(s+'.json')) for s in rooms if (source/(s+'.json')).exists()]
    native_summary=read(native/'summary.json')
    trained=STUDY/'response_value_multitask_v1/summary.json'
    feedback=dict(native_complete=native_summary['complete'], native_rooms=native_summary['rooms'],
        native_views=native_summary['frames'], native_regions=native_summary['regions'],
        native_run_bytes=tree_bytes(native), source_rooms_complete=len(reports), source_rooms_required=len(rooms),
        source_room_counts={k:sum(r['split']==k for r in reports) for k in manifest['splits']},
        source_episodes=sum(r['queries'] for r in reports), source_run_bytes=tree_bytes(source) if source.exists() else 0,
        source_teacher_configuration=['SigLIP2 giant','Mask-Adapter ConvNeXt CLIP','literal-category SAM3'],
        source_complete=(source/'summary.json').exists(), model_fitted=trained.exists(),
        shared_decision='one intact answer, one model, one margin; no classwise or taskwise winner stitching',
        calibration_protected_outputs=['text IoU','image IoU','menu mIoU19','menu mIoU15','menu mIoU10'],
        calibration_cells=15, calibration_missing_classes=read(STUDY/'cross_task_targets_v1/summary.json')['missing_calibration_classes'],
        new_benchmark_metrics=False, default_changed=False, independent_final_test=False,
        benchmark_transfer={k:'pending; no new inference in this data-construction program' for k in
            ('lerf_semantic2d','lerf_selected_set_projection2d','nvos','scannet_ovs','uqis')},
        limitations=['registered mesh carrier differs from benchmark Gaussian renderer',
            'full-map episode reuses source evidence; three folds have disjoint views but shared full mesh and adjacent boundary poses',
            'six calibration classes remain absent', 'previous development and calibration data already opened'],
        report_hashes={str(source/(r['scene']+'.json')):sha(source/(r['scene']+'.json')) for r in reports},
        native_summary_sha256=sha(native/'summary.json'), study_budget_remaining=POLICY['study_bytes']-tree_bytes(STUDY))
    if trained.exists():
        learned=read(trained)
        feedback['learned_development']=learned['development']; feedback['development_gate']=learned['development_gate']
        feedback['training_summary_sha256']=sha(trained)
    # Construction diagnosis on fit rooms only. A better label oracle is NOT
    # a learned result; never use it to route benchmark outputs or trim rooms.
    fit=[r for r in reports if r['split']=='fit'];capacity={}
    for group in ('full_map','disjoint_view_folds'):
        values=[np.mean([row['menu_utilities'][k] for k in ('19','15','10')],axis=0)
            for r in fit for row in r['rows'] if (row['fold']=='full')==(group=='full_map')]
        if values:
            value=np.stack(values)
            capacity[group]=dict(episodes=len(value),fixed_provider_mean_utility=value.mean(0).tolist(),
                episodes_with_better_candidate=int((value[:,1:].max(1)>value[:,0]+1e-12).sum()),
                finite_pool_oracle_mean_gain=float((value.max(1)-value[:,0]).mean()))
    feedback['fit_pool_diagnosis']=dict(rooms=len(fit),required_rooms=len(manifest['splits']['fit']),
        provisional=len(fit)!=len(manifest['splits']['fit']),groups=capacity,
        diagnostic_only=True,benchmark_accuracy_result=False,calibration_and_development_utilities_not_inspected=True)
    paper=Path('paper/tmm_response'); (paper/'category_supervision_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
    print(json.dumps({k:v for k,v in feedback.items() if k not in ('report_hashes','learned_development')},indent=2))


if __name__=='__main__':
    main()
