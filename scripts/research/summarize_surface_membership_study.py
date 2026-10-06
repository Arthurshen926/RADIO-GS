"""Compare the same frozen unary with fixed and learned surface relations."""
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(path.read_text())


def main():
    root=Path('output/surface_membership_study')
    paths=dict(v8=Path('output/system_v8_membership_closure_20261005'),
        parent=Path('output/learned_instance_additions/element_query_system'),
        fixed=root/'fixed_system',learned=root/'learned_system')
    scored={key:read(path/'uqis/identity_scope/result.json') for key,path in paths.items()}
    output=dict(conditions={},calibration=read(root/'cut_readout.json'),
        laplacian_development=read(root/'calibration.json'),cut_development=read(root/'cut_calibration.json'),
        frozen_confirmation=read(root/'cut_confirmation.json'),
        unknown_completion=read(root/'cut_unknown_confirmation.json'),
        training=read(root/'fit.json'),default_changed=False,independent_final_test=False,
        additional_3d_supervision=True,benchmark_gt_used_for_parameters=False,
        public_tasks=read(paths['learned']/'summary.json')['macro'],
        execution='31 learned-condition scene jobs plus9 fixed-relation UQIS jobs; public readers are identical')
    for condition,data in scored.items():
        row=dict(modalities={},controls={},
            all_four_successes=sum(v['all_four_iou_at_least_050']
                for v in data['explicit_selection']['paired_query_robustness']['per_target']))
        for modality in ['text','image','point_2d','point_3d']:
            selected=data['explicit_selection']['modalities'][modality]
            result=dict(ap=data['modalities'][modality]['scene_macro']['average_precision'],
                **selected['scene_macro'],per_scene=selected['per_scene'],differences={})
            for reference in ['v8','parent','fixed']:
                other=scored[reference]['explicit_selection']['modalities'][modality]
                differences={s:100*(v['iou']-other['per_scene'][s]['iou']) for s,v in selected['per_scene'].items()}
                result['differences'][reference]=dict(scene_macro_pp=scene_clustered_bootstrap(differences),
                    per_scene_pp=differences,improved_scenes=sum(v>0 for v in differences.values()))
            row['modalities'][modality]=result
        if condition in ['fixed','learned']:
            for readout in ['native_semantic','anchor_comembership','identity_region']:
                for kind in ['predictions','selections']:
                    files=list((paths[condition]/'uqis'/readout/kind).glob('*.npy'))
                    if files:row['controls'][readout+'/'+kind]=dict(queries=len(files),changed=sum(
                        not np.array_equal(np.load(file),np.load(paths['v8']/'uqis'/readout/kind/file.name)) for file in files))
            receipts=[read(p) for p in (paths[condition]/'uqis').glob('scene*/query/registered/runs/*.json')]
            row['resources']=dict(map_arrays_mib=sum(r['map_bytes'] for r in receipts)/2**20,
                registered_query_seconds=sum(r['seconds'] for r in receipts),
                query_source_rgb_reads=sum(r['source_rgb_read'] for r in receipts),
                query_source_depth_reads=sum(r['source_depth_read'] for r in receipts),
                extra_graph_mib=sum(p.stat().st_size for p in (root/'maps'/condition).glob('*/surface_affinity.npz'))/2**20,
                shared_encoders_extra=True,shared_membership_checkpoint_extra=True,
                compact_student=False)
        output['conditions'][condition]=row
    output['selection_diagnosis']=read(root/'selection_diagnosis.json')
    output['decision']=dict(promoted=False,retained_recipe='configs/pipeline/system_v8.json',
        useful_result='Joint set inference improves actual point IoU over identical independent-element unaries.',
        learned_edge_contribution='No consistent advantage over fixed relations across both point modalities.',
        remaining_limits='Severe two-scene regressions versus v8; 3D-point interval crosses zero; image/text AP declines; all-four success unchanged.',
        unknown_completion='Rejected on eight independent-training-domain development rooms; no benchmark execution or parameter sweep.',
        next='Identity-aware support beyond one-hop candidates, with explicit membership/unknown evidence; do not expand graph-strength search.')
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:dict(actual_iou={m:r['iou'] for m,r in v['modalities'].items()},
        ap={m:r['ap'] for m,r in v['modalities'].items()},all_four=v['all_four_successes'])
        for k,v in output['conditions'].items()},indent=2))


if __name__=='__main__':main()
