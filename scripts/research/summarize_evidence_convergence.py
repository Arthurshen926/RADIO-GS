"""Report complete methods, keeping rankings and physical selections separate."""
import json
from pathlib import Path
import numpy as np
from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(path.read_text())


def main():
    root=Path('output/evidence_convergence_20261005')
    paths=dict(v8=Path('output/system_v8_membership_closure_20261005'),
        surface=Path('output/surface_membership_study/learned_system'),
        local=root/'local_system',partial=root/'partial_system')
    summaries={name:read(path/'summary.json') for name,path in paths.items()}
    uqis={name:read(path/'uqis/identity_scope/result.json') for name,path in paths.items()}
    output=dict(calibration=read(root/'element_domain_calibration.json'),conditions={},
        default_changed=False,benchmark_labels_used_for_parameters=False,independent_final_test=False,
        source_geometry_native_spaces_unchanged=True)
    for name,summary in summaries.items():
        d=uqis[name];row=dict(macro=summary['macro'],uqis={},differences={},
            all_four_successes=sum(r['all_four_iou_at_least_050'] for r in d['explicit_selection']['paired_query_robustness']['per_target']))
        for modality in ['text','image','point_2d','point_3d']:
            measured=d['explicit_selection']['modalities'][modality]
            row['uqis'][modality]=dict(ap=d['modalities'][modality]['scene_macro']['average_precision'],
                **measured['scene_macro'])
            row['differences'][modality]={}
            for reference in ['v8','surface','local']:
                other=uqis[reference]['explicit_selection']['modalities'][modality]['per_scene']
                delta={s:100*(v['iou']-other[s]['iou']) for s,v in measured['per_scene'].items()}
                row['differences'][modality][reference]=dict(per_scene_pp=delta,scene_macro_pp=scene_clustered_bootstrap(delta))
        if name in ['local','partial']:
            row['public_deltas']={}
            for benchmark in ['lerf','nvos','scannet']:
                current={r['job']['name']:r['metrics'] for r in summary['rows'] if r['job']['benchmark']==benchmark}
                old={r['job']['name']:r['metrics'] for r in summaries['v8']['rows'] if r['job']['benchmark']==benchmark}
                row['public_deltas'][benchmark]={key:dict(
                    macro_pp=100*(summary['macro'][benchmark][key]-summaries['v8']['macro'][benchmark][key]),
                    per_scene_pp={s:100*(v[key]-old[s][key]) for s,v in current.items()}) for key in next(iter(current.values()))}
            # The new API computes both outputs in one execution. Verify that
            # disentangling ranks did not secretly substitute historical sets.
            selection=root/(name+'_system')/'uqis/identity_scope/selections'
            row['selections_changed_vs_surface']=sum(not np.array_equal(np.load(p),
                np.load(paths['surface']/'uqis/identity_scope/selections'/p.name)) for p in selection.glob('*.npy'))
        output['conditions'][name]=row
    ownership=root/'ownership_system/summary.json'
    if ownership.exists():
        control=read(ownership)
        output['positive_ownership_control']=dict(completed=control['completed'],macro=control['macro'],
            public_only=True,uqis_implementation_identical_to='local',
            differences_vs_v8={b:{k:100*(v-summaries['v8']['macro'][b][k]) for k,v in metrics.items()}
                for b,metrics in control['macro'].items()})
    output['nvos_actual_sets']={p.stem:read(p) for p in (root/'nvos_sets').glob('*.json')}
    output['decision']=dict(promoted=False,retained_recipe='configs/pipeline/system_v8.json',
        correctness_repairs=['deployment-domain calibration','separate unary ranking and actual MAP selection'],
        source_result='positive ownership approximately preserves accuracy; minimum-complement rules regress public tasks',
        scope_result='only about0.21pp text/image IoU gain; all-four success remains7/67',
        next='joint native-space query identity and complete membership beyond one-hop support; no further scalar confidence search')
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({name:dict(public={b:r['macro'][b] for b in ['lerf','nvos','scannet']},
        uqis=r['uqis'],all_four=r['all_four_successes']) for name,r in output['conditions'].items()},indent=2))


if __name__=='__main__':main()
