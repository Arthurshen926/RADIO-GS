"""Post-prediction identity/range diagnosis for the complete common-system run."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/system_uqis_joint_candidate'))
    p.add_argument('--package',type=Path,default=Path('output/scannet_uqis_9_v0_3_1_candidate'))
    p.add_argument('--maps',type=Path,default=Path('output/uqis_map_only_v1/maps'))
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    complete=read(a.root/'predictions_complete.json')
    names=['native_semantic','anchor_comembership','identity_region','identity_scope']
    results={n:read(a.root/'uqis'/n/'result.json') for n in names}
    primary={n:{m:v['scene_macro'] for m,v in d['modalities'].items()} for n,d in results.items()}
    gains={}
    for n in names[2:]:
        gains[n]={m:scene_clustered_bootstrap({s:100*(v['average_precision']-
            results['anchor_comembership']['modalities'][m]['per_scene'][s]['average_precision'])
            for s,v in results[n]['modalities'][m]['per_scene'].items()}) for m in primary[n]}
    evaluation=read(a.package/'evaluator/targets.json');cases=[]
    for domain in evaluation['scene_domains']:
        scene=domain['scene_id'];labels=np.load(a.package/'evaluator'/domain['mesh_instance_ids_path'])
        membership=sparse.load_npz(a.maps/scene/'membership.npz').tocsc()
        area=np.asarray(membership.sum(0)).reshape(-1)
        run=read(a.root/'uqis'/scene/'query/registered/runs'/f'{scene}.json')
        details={q['query_id']:q for q in run['query_details']}
        for target in evaluation['targets']:
            if target['scene_id']!=scene:continue
            truth=labels==target['instance_id'];intersection=np.asarray(membership.T@truth.astype(np.float32)).reshape(-1)
            iou=intersection/np.maximum(area+truth.sum()-intersection,1.)
            for modality in ['text','image']:
                qid=target['queries'][modality];identity=details[qid]['instance_identity'];seed=identity['seed_region']
                cases.append(dict(scene_id=scene,instance_id=target['instance_id'],modality=modality,query_id=qid,
                    seed_precision=float(intersection[seed]/max(area[seed],1)),seed_recall=float(intersection[seed]/truth.sum()),
                    seed_iou=float(iou[seed]),best_available_region_iou=float(iou.max()),
                    matched_views=identity['matched_views'],language_tags=target['language_tags']))
    diagnosis={m:dict(queries=sum(c['modality']==m for c in cases),
        seeds_majority_target=sum(c['modality']==m and c['seed_precision']>.5 for c in cases),
        mean_seed_precision=float(np.mean([c['seed_precision'] for c in cases if c['modality']==m])),
        mean_seed_recall=float(np.mean([c['seed_recall'] for c in cases if c['modality']==m])),
        mean_best_available_region_iou=float(np.mean([c['best_available_region_iou'] for c in cases if c['modality']==m])))
        for m in ['text','image']}
    parity={}
    previous=Path('output/uqis_map_only_v031')
    for n in names[:2]:
        files=sorted((a.root/'uqis'/n/'predictions').glob('*.npy'))
        parity[n]=dict(count=len(files),max_difference=max(float(np.max(np.abs(np.load(f)-np.load(previous/n/'predictions'/f.name)))) for f in files))
    summary=dict(scope='same maps, query inputs and native teachers; opened development; candidate chosen before this cohort scoring',
        all_predictions_before_scoring=complete['all_predictions_before_scoring'],modalities=primary,
        gain_over_anchor_readout_pp=gains,paired_robustness={n:{k:d['paired_query_robustness'][k] for k in ['minimum_ap','geometric_mean_ap']} for n,d in results.items()},
        control_parity=parity,identity_diagnosis=diagnosis,cases=cases,
        oracle_scope='best available projected region uses labels for diagnosis only, never automatic prediction')
    selected={n:d['explicit_selection'] for n,d in results.items() if 'explicit_selection' in d}
    if selected:
        summary['explicit_selection']={n:{m:v['scene_macro'] for m,v in d['modalities'].items()} for n,d in selected.items()}
        summary['selection_scope_gain_pp']={m:scene_clustered_bootstrap({s:100*(v['iou']-
            selected['identity_region']['modalities'][m]['per_scene'][s]['iou'])
            for s,v in selected['identity_scope']['modalities'][m]['per_scene'].items()})
            for m in ['text','image','point_2d','point_3d']}
    (a.root/'uqis/mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','modalities']},indent=2))


if __name__=='__main__':main()
