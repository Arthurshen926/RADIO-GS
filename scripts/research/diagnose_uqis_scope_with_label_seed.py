"""Label-using range diagnosis after the automatic full-system comparisons.

GT chooses the best available input region, NOT the best final output. These
numbers are not automatic predictions, a learned model, or a formal upper bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.querying.instance_observation_scope import InstanceObservationScope


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/member_identity_study/label_seed_diagnosis.json'))
    a=p.parse_args();package=Path('output/scannet_uqis_9_v0_3_1_candidate')
    evaluation=json.loads((package/'evaluator/targets.json').read_text());rows=[]
    for domain in evaluation['scene_domains']:
        scene=domain['scene_id'];root=Path('output/uqis_map_only_v1/maps')/scene
        labels=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        membership=sparse.load_npz(root/'membership.npz');visibility=sparse.load_npz(root/'visibility.npz')
        frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(root/'region_offsets.npy')))
        reader=InstanceObservationScope(membership,visibility,frames);area=np.asarray(membership.sum(0)).reshape(-1)
        for target in evaluation['targets']:
            if target['scene_id']!=scene:continue
            truth=labels==target['instance_id'];inter=np.asarray(membership.T@truth.astype(np.float32)).reshape(-1)
            iou=inter/np.maximum(area+truth.sum()-inter,1.)
            seed=int(iou.argmax());identity=np.zeros(len(area),np.float32);identity[seed]=1.
            _,_,selected,details=reader.infer(identity,np.zeros(len(labels),np.float32))
            masks={'best_input_region':selected['identity_region'],'current_scope':selected['identity_scope'],
                'matched_positive_union':np.asarray(membership[:,details['matched_regions']].sum(1)).reshape(-1)>0}
            record=dict(scene_id=scene,instance_id=target['instance_id'],label_selected_seed=seed,matched_views=details['matched_views'])
            for name,mask in masks.items():
                tp=int((mask&truth).sum());record[name]=dict(iou=tp/max(int((mask|truth).sum()),1),
                    purity=tp/max(int(mask.sum()),1),coverage=tp/max(int(truth.sum()),1))
            rows.append(record)
        print(scene,'label-using scope diagnosis complete',flush=True)
    scenes=sorted({r['scene_id'] for r in rows});names=['best_input_region','current_scope','matched_positive_union']
    per_scene={s:{n:{k:float(np.mean([r[n][k] for r in rows if r['scene_id']==s])) for k in ['iou','purity','coverage']} for n in names} for s in scenes}
    macro={n:{k:float(np.mean([per_scene[s][n][k] for s in scenes])) for k in ['iou','purity','coverage']} for n in names}
    result=dict(labels_used_to_choose_seed=True,automatic_method_result=False,formal_upper_bound=False,
        description='Best input-region IoU under labels, then fixed geometry correspondence; union is a diagnosis of partial-scope exclusion, not a promoted readout',
        targets=len(rows),scene_macro=macro,per_scene=per_scene,per_target=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(macro,indent=2))


if __name__=='__main__':main()
