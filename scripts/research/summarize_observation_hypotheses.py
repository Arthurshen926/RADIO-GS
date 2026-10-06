"""Compare complete, fixed-input candidates; separate identity from scope loss."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/observation_hypotheses_study/comparison.json'))
    p.add_argument('--include-tracks',action='store_true');p.add_argument('--include-appearance',action='store_true')
    p.add_argument('--include-dual',action='store_true');a=p.parse_args()
    roots={'v7':Path('output/system_v7'),'scope_hypotheses':Path('output/observation_hypotheses_study/system')}
    if a.include_tracks:roots['view_exclusive_tracks']=Path('output/view_exclusive_tracks_study/system')
    if a.include_appearance:roots['native_local_appearance']=Path('output/region_appearance_study/system')
    if a.include_dual:roots['dual_native_identity']=Path('output/dual_native_identity_study/system')
    results={name:read(root/'uqis/identity_scope/result.json') for name,root in roots.items()}
    baseline=results['v7'];rows={}
    for name,result in results.items():
        row=dict(modalities={},paired=result['explicit_selection']['paired_query_robustness'],
            text_language_groups={tag:value['scene_macro']['average_precision'] for tag,value in result['text_language_groups'].items()},
            image_reference_groups={tag:value['scene_macro']['average_precision'] for tag,value in result['image_reference_difficulty_groups'].items()})
        for m,value in result['modalities'].items():
            selected=result['explicit_selection']['modalities'][m]
            row['modalities'][m]=dict(ap=100*value['scene_macro']['average_precision'],
                **{k:100*v for k,v in selected['scene_macro'].items()},
                empty_selections=selected['empty_selections'],
                ap_gain_pp=scene_clustered_bootstrap({s:100*(v['average_precision']-baseline['modalities'][m]['per_scene'][s]['average_precision']) for s,v in value['per_scene'].items()}),
                iou_gain_pp=scene_clustered_bootstrap({s:100*(v['iou']-baseline['explicit_selection']['modalities'][m]['per_scene'][s]['iou']) for s,v in selected['per_scene'].items()}))
        rows[name]=row
    evaluation=read(Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator/targets.json'))
    cases=[]
    for domain in evaluation['scene_domains']:
        scene=domain['scene_id'];labels=np.load(Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')/domain['mesh_instance_ids_path'])
        membership=sparse.load_npz(Path('output/uqis_map_only_v1/maps')/scene/'membership.npz').tocsc()
        area=np.asarray(membership.sum(0)).reshape(-1)
        details={name:{q['query_id']:q for q in read(root/'uqis'/scene/'query/registered/runs'/f'{scene}.json')['query_details']} for name,root in roots.items()}
        for target in evaluation['targets']:
            if target['scene_id']!=scene:continue
            truth=labels==target['instance_id'];intersection=np.asarray(membership.T@truth.astype(np.float32)).reshape(-1)
            for modality in ['text','image']:
                qid=target['queries'][modality];row=dict(scene_id=scene,modality=modality,query_id=qid,instance_id=target['instance_id'],methods={})
                for name,detail in details.items():
                    scope=detail[qid]['instance_identity'];seed=scope['seed_region']
                    row['methods'][name]=dict(seed=seed,matched_views=scope['matched_views'],
                        seed_purity=float(intersection[seed]/max(area[seed],1)),seed_coverage=float(intersection[seed]/truth.sum()),
                        seed_iou=float(intersection[seed]/max(area[seed]+truth.sum()-intersection[seed],1)))
                cases.append(row)
    for name in rows:
        rows[name]['seed_diagnosis']={m:dict(majority_target=sum(c['methods'][name]['seed_purity']>.5 for c in cases if c['modality']==m),
            changed_from_v7=sum(c['methods'][name]['seed']!=c['methods']['v7']['seed'] for c in cases if c['modality']==m),
            **{key:float(np.mean([c['methods'][name][key] for c in cases if c['modality']==m])) for key in ['seed_purity','seed_coverage','seed_iou','matched_views']}) for m in ['text','image']}
        if name!='v7':
            maps=roots[name].parent/'maps';builds=[read(p) for p in maps.glob('*/build.json')]
            rows[name]['cost']=dict(map_mib=sum(read(p)['map_bytes'] for p in maps.glob('*/map.json'))/2**20,
                build_seconds_sum=sum(d['seconds'] for d in builds))
            if name in ['native_local_appearance','dual_native_identity']:
                reused=all(d.get('feature_build_reused',False) for d in builds)
                rows[name]['cost'].update(raw_regions=sum(d['regions'] for d in builds),
                    new_teacher_calls=0 if reused else sum(d['source_frames'] for d in builds),
                    required_teacher_calls_for_fresh_map=sum(d['source_frames'] for d in builds),
                    source_features_reused=reused,
                    build_time_scope='original source descriptor compilation; reused for dual identity; excludes hardlink materialization',
                    extra_shared_models=1,
                    shared_checkpoint_mib=builds[0]['shared_checkpoint_bytes']/2**20,
                    loaded_half_model_mib=builds[0]['shared_parameter_bytes']/2**20)
            else:
                rows[name]['cost'].update(raw_regions=sum(d['hypotheses'] for d in builds),
                    supported_regions=sum(d['supported_hypotheses'] for d in builds),multiview_hypotheses=sum(d['multiview_hypotheses'] for d in builds),
                    new_teacher_calls=0,extra_shared_models=0)
            rows[name]['control_parity']={}
            query_modes={q['query_id']:q['modality'] for q in results[name]['per_query']}
            for control in ['native_semantic','anchor_comembership']:
                files=sorted((roots[name]/'uqis'/control/'predictions').glob('*.npy'))
                differences={f.stem:float(np.max(np.abs(np.load(f)-np.load(roots['v7']/'uqis'/control/'predictions'/f.name)))) for f in files}
                rows[name]['control_parity'][control]=dict(queries=len(files),maximum_absolute_difference=max(differences.values()),
                    by_modality={m:max(v for q,v in differences.items() if query_modes[q]==m) for m in results[name]['modalities']})
    report=dict(scope='complete 9-scene, 67-target, 268-query opened-development comparison; same source views, masks and geometry; appearance arm explicitly adds a teacher',
        rules_chosen_before_scoring=True,new_source_images=0,additional_query_rgb_for_points=False,
        public_benchmarks='declared common recipe retained; unchanged public providers not rerun for this isolated mesh-scope study',
        results=rows,post_prediction_label_seed_diagnosis=cases)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{m:{k:v[k] for k in ['ap','iou','purity','coverage']} for m,v in row['modalities'].items()} for name,row in rows.items()},indent=2))


if __name__=='__main__':main()
