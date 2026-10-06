"""Report frozen membership mechanisms, independent rooms and real selected sets."""
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(path.read_text())


def main():
    root=Path('output/learned_instance_additions')
    reference=Path('output/system_v8_membership_closure_20261005')
    base=read(reference/'uqis/identity_scope/result.json')
    result=dict(reference=str(reference),default_changed=False,
        conditions={},public_common_control={k:v for k,v in read(root/'element_joint_system/summary.json')['macro'].items() if k!='uqis'},
        execution='31 fresh full-system jobs plus9+9 fresh UQIS jobs; public methods identical across conditions',
        mainline_gate='actual-set improvements must be stable, not AP alone or positive mean with severe scene regressions',
        independent_final_test=False,benchmark_parameters_fitted=False)
    for name in ['element_joint','element_joint_oof','element_query']:
        system=root/(name+'_system');new=read(system/'uqis/identity_scope/result.json')
        row=dict(modalities={},unchanged_controls={},all_four_successes=int(sum(
            r['all_four_iou_at_least_050'] for r in new['explicit_selection']['paired_query_robustness']['per_target'])))
        for modality in ['text','image','point_2d','point_3d']:
            data=new['explicit_selection']['modalities'][modality]
            previous=base['explicit_selection']['modalities'][modality]
            delta={s:100*(v['iou']-previous['per_scene'][s]['iou']) for s,v in data['per_scene'].items()}
            row['modalities'][modality]=dict(ap=new['modalities'][modality]['scene_macro']['average_precision'],
                **data['scene_macro'],iou_delta_pp=scene_clustered_bootstrap(delta),per_scene_delta_pp=delta,
                improved_scenes=sum(v>0 for v in delta.values()),
                dominant_target_count=new['explicit_selection']['identity_diagnosis'][modality]['dominant_target_count'])
        for readout in ['native_semantic','anchor_comembership','identity_region']:
            for kind in ['predictions','selections']:
                paths=list((system/'uqis'/readout/kind).glob('*.npy'))
                if paths:row['unchanged_controls'][readout+'/'+kind]=dict(queries=len(paths),changed=sum(
                    not np.array_equal(np.load(p),np.load(reference/'uqis'/readout/kind/p.name)) for p in paths))
        semantic_ids=[r['query_id'] for r in new['explicit_selection']['per_query'] if r['modality'] in ['text','image']]
        for kind in ['predictions','selections']:
            row['unchanged_controls']['text_image_scope/'+kind]=dict(queries=len(semantic_ids),changed=sum(
                not np.array_equal(np.load(system/'uqis/identity_scope'/kind/f'{qid}.npy'),
                                   np.load(reference/'uqis/identity_scope'/kind/f'{qid}.npy')) for qid in semantic_ids))
        row['decision']='research only; actual-set mean gains insufficiently stable for default promotion'
        result['conditions'][name]=row
    result['additional_room_evaluations']={name:{k:v for k,v in read(root/(name+'.json')).items()
        if k in ['rooms','queries','scene_macro','per_scene','complete_carrier_scene_macro']}
        for name in ['joint_confirmation_anchor_control','joint_confirmation','joint_oof_confirmation',
                     'query_confirmation_oof_control','query_confirmation']}
    region=read(root/'region_cross_fit.json')
    result['upstream_training_prediction_error']={key:np.mean([r[key] for r in region['rows']],axis=0).tolist()
        for key in ['in_sample_mae','out_of_fold_mae']}
    seed_rows=[r for p in (root/'anchor_region_training').glob('*.json') for r in read(p)]
    result['point_conditioned_training']=dict(anchor_episodes=len(seed_rows),
        seed_target_fraction_below_08=sum(r['target_fraction_in_seed']<.8 for r in seed_rows),
        seed_target_fraction_below_05=sum(r['target_fraction_in_seed']<.5 for r in seed_rows),
        regional_pairs=sum(r['pairs'] for r in seed_rows),element_training_samples=887126)
    (root/'joint_evidence_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({n:r['modalities'] for n,r in result['conditions'].items()},indent=2))


if __name__=='__main__':main()
