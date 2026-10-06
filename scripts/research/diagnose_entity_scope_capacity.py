"""Post-prediction comparison of entity scope capacity and source coverage."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from radio_gs.querying.observation_hypotheses import compile_hypotheses
from scripts.research.diagnose_uqis_hypothesis_capacity import candidate_capacity,selected_scopes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--compare-capacity',type=Path);args=p.parse_args();root=args.root
    complete=json.loads((root/'system/predictions_complete.json').read_text())
    assert complete['jobs']==9
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    data=json.loads((package/'targets.json').read_text());cases=[];start=time.monotonic()
    for domain in data['scene_domains']:
        scene=domain['scene_id'];folder=root/'maps'/scene
        membership=sparse.load_npz(folder/'entity_membership.npz').astype(np.float32)
        visibility=sparse.load_npz(folder/'visibility.npz')
        frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'entity_offsets.npy')))
        labels=np.load(package/domain['mesh_instance_ids_path'])
        targets=[t for t in data['targets'] if t['scene_id']==scene]
        truth=np.stack([labels==t['instance_id'] for t in targets],axis=1).astype(np.float32)
        covered=membership.getnnz(1)>0
        if membership.shape[1]:
            hypotheses=compile_hypotheses(membership,visibility,frames)
            selected=selected_scopes(membership,visibility,frames,hypotheses)
            scores={n:candidate_capacity(m,truth) for n,m in [('entity_region',membership),('entity_physical_scope',selected)]}
        else:
            scores={n:[dict(candidate=None,iou=0.,purity=0.,coverage=0.,selected_elements=0) for _ in targets]
                    for n in ['entity_region','entity_physical_scope']}
        for i,target in enumerate(targets):
            cases.append(dict(scene_id=scene,instance_id=target['instance_id'],
                all_entity_observation_coverage=float(covered[truth[:,i]>0].mean()),
                best_label_selected={n:v[i] for n,v in scores.items()}))
        print(scene,'diagnosed',len(targets),flush=True)
    scenes=sorted({r['scene_id'] for r in cases});summary={}
    for name in scores:
        per_scene={s:{k:float(np.mean([r['best_label_selected'][name][k] for r in cases if r['scene_id']==s]))
            for k in ['iou','purity','coverage']} for s in scenes}
        summary[name]=dict(scene_macro={k:float(np.mean([r[k] for r in per_scene.values()])) for k in ['iou','purity','coverage']},per_scene=per_scene)
    report=dict(role='post-prediction label-selected finite-candidate diagnosis only',
        automatic_method_result=False,query_process_can_access=False,summary=summary,cases=cases,
        source_support_scene_macro=float(np.mean([np.mean([r['all_entity_observation_coverage'] for r in cases if r['scene_id']==s]) for s in scenes])),
        seconds=time.monotonic()-start)
    (root/'label_capacity.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.compare_capacity:
        prior=json.loads(args.compare_capacity.read_text())
        lookup={(r['scene_id'],r['instance_id']):r['best_label_selected']['one_hop_scope'] for r in prior['cases']}
        combined=[]
        for row in cases:
            old=lookup[(row['scene_id'],row['instance_id'])]
            new=row['best_label_selected']['entity_physical_scope']
            best=new if new['iou']>old['iou'] else old
            combined.append(dict(scene_id=row['scene_id'],instance_id=row['instance_id'],old=old['iou'],entity=new['iou'],
                **{k:best[k] for k in ['iou','purity','coverage']}))
        union=dict(role='label-selected union of two finite candidate families; diagnosis only, never automatic method',
            summary={k:float(np.mean([np.mean([r[k] for r in combined if r['scene_id']==s]) for s in scenes]))
                for k in ['iou','purity','coverage']},
            targets_with_better_entity_scope=sum(r['entity']>r['old'] for r in combined),cases=combined)
        (root/'combined_label_capacity.json').write_text(json.dumps(union,indent=2)+'\n')
    print({k:v['scene_macro'] for k,v in summary.items()})


if __name__=='__main__':main()
