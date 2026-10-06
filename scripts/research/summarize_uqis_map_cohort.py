"""Post-prediction localization of map support, anchors and query retrieval."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/uqis_map_only_v1'));a=p.parse_args()
    complete=read(a.root/'predictions_complete.json')
    cfg=read(a.root/'configuration.json');package=Path(cfg['benchmark_package'])
    evaluation=read(package/'evaluator/targets.json')
    results={r:read(a.root/r/'result.json') for r in cfg['query_readouts']}
    by_query={r:{v['query_id']:v for v in result['per_query']} for r,result in results.items()}
    cases=[];maps=[]
    for domain in evaluation['scene_domains']:
        scene=domain['scene_id'];folder=a.root/'maps'/scene
        spec=read(folder/'map.json');run=read(a.root/'runs'/f'{scene}.json')
        anchors={r['query_id']:r for r in run['query_details']}
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        transfer=sparse.load_npz(folder/'transfer.npz');supported=transfer.getnnz(axis=1)>0
        visibility=sparse.load_npz(folder/'visibility.npz');membership=sparse.load_npz(folder/'membership.npz')
        region_frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
        maps.append({k:spec[k] for k in ['scene_id','source_count','elements','regions','supported_elements','map_bytes','seconds']})
        maps[-1]['query_seconds']=run['seconds']
        for target in evaluation['targets']:
            if target['scene_id']!=scene:continue
            truth=labels==target['instance_id']
            row=dict(scene_id=scene,instance_id=target['instance_id'],raw_label=target['raw_semantic_label'],
                     field_target_support=float(supported[truth].mean()),modalities={})
            for modality,qid in target['queries'].items():
                detail=dict(native_ap=by_query['native_semantic'][qid]['average_precision'],
                    scope_ap=by_query['anchor_comembership'][qid]['average_precision'])
                if modality.startswith('point_'):
                    anchor=anchors[qid];index=anchor['anchor_vertex']
                    frames=np.unique(region_frames[membership[index].indices])
                    known=np.asarray(visibility[:,frames].sum(1)).reshape(-1)>0
                    detail.update(anchor_is_target=bool(labels[index]==target['instance_id']),
                        anchor_has_feature=anchor['anchor_has_feature'],anchor_views=anchor['anchor_views'],
                        target_in_known_anchor_views=float(known[truth].mean()),
                        ray_surface_missing=bool(anchor.get('surface_intersection_missing',False)))
                row['modalities'][modality]=detail
            cases.append(row)
    modalities=['text','image','point_2d','point_3d']
    primary={name:{m:result['modalities'][m]['scene_macro'] for m in modalities} for name,result in results.items()}
    gain={m:100*(primary['anchor_comembership'][m]['average_precision']-primary['native_semantic'][m]['average_precision']) for m in modalities}
    paired_ci={m:scene_clustered_bootstrap({s:100*(v['average_precision']-
        results['native_semantic']['modalities'][m]['per_scene'][s]['average_precision'])
        for s,v in results['anchor_comembership']['modalities'][m]['per_scene'].items()}) for m in modalities}
    anchor_summary={m:{'correct':sum(c['modalities'][m]['anchor_is_target'] for c in cases),
        'with_features':sum(c['modalities'][m]['anchor_has_feature'] for c in cases),
        'missing_ray_surface':sum(c['modalities'][m]['ray_surface_missing'] for c in cases),
        'mean_target_in_known_views':float(np.mean([c['modalities'][m]['target_in_known_anchor_views'] for c in cases]))}
        for m in ['point_2d','point_3d']}
    result=dict(scope='opened development cohort; identical maps/evidence, not old-version or public-SOTA comparison',
        scene_count=len(maps),target_count=len(cases),queries=4*len(cases),all_predictions_before_scoring=complete['all_predictions_before_scoring'],
        modalities=primary,ap_gain_percentage_points=gain,paired_scene_bootstrap_gain=paired_ci,anchors=anchor_summary,
        mean_target_field_support=float(np.mean([c['field_target_support'] for c in cases])),
        deployment_map_bytes=sum(m['map_bytes'] for m in maps),cost_excludes_global_encoders=True,
        source_count=sum(m['source_count'] for m in maps),regions=sum(m['regions'] for m in maps),
        maps=maps,cases=cases)
    (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['maps','cases','modalities']},indent=2))


if __name__=='__main__':main()
