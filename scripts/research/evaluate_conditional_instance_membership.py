"""Fixed positive-point episodes on the four held-out training-domain rooms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_conditional_instance_membership import read_scene
from radio_gs.querying.conditional_region_membership import ConditionalRegionMembership
from radio_gs.querying.conditional_element_membership import ConditionalElementMembership
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.registered_observation_query import anchor_membership


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--scenes',nargs='+');p.add_argument('--element-checkpoint',type=Path)
    p.add_argument('--output-name',default='heldout_instances.json');a=p.parse_args()
    design=json.loads((a.root/'design.json').read_text());rows=[]
    scenes=a.scenes or design['validation']
    for scene in scenes:
        f,labels=read_scene(a.root,scene);b=f.membership.tocsr()
        learned=ConditionalRegionMembership(b,f.visibility,f.frames,np.load(a.root/'maps'/scene/'mesh_xyz.npy'),
            f.descriptors,a.root/'conditional_membership.joblib')
        if a.element_checkpoint:
            learned=ConditionalElementMembership(b,f.visibility,f.frames,np.load(a.root/'maps'/scene/'mesh_xyz.npy'),
                f.descriptors,a.element_checkpoint,np.load(a.root/'maps'/scene/'element_appearance.npy',mmap_mode='r'),
                np.load(a.root/'maps'/scene/'element_appearance_observation.npy'))
        geometric=InstanceObservationScope(b,f.visibility,f.frames);known=labels>=0
        identity=np.zeros(b.shape[1]);native=np.zeros(b.shape[0]);scene_rows=[]
        for instance in np.unique(labels[known]):
            members=np.flatnonzero(labels==instance)
            seed=int.from_bytes(hashlib.sha256(f'{scene}/{instance}/positive-point'.encode()).digest()[:8],'little')
            anchor=int(members[np.random.default_rng(seed).integers(len(members))])
            base,_=anchor_membership(b,f.visibility,f.frames,anchor)
            finest=base>.5
            regions=b[anchor].indices
            if len(regions):
                seed_region=int(regions[np.argmin(f.area[regions])])
                _,_,selected,_=geometric.infer(identity,native,seed_region)
                finest=selected['identity_scope']
            _,selected,detail=learned.infer(identity,native,anchor)
            if not detail['applied']:selected=base>.5
            truth=labels==instance;record=dict(scene=scene,instance=int(instance),anchor=anchor,
                anchor_supported=bool(len(regions)),learned_applied=detail['applied'],arms={})
            for name,mask in [('anchor',base>.5),('finest_geometric',finest),('learned',selected)]:
                selected_known=mask&known;tp=int((mask&truth).sum());total=int(selected_known.sum());count=len(members)
                record['arms'][name]=dict(iou=tp/max(total+count-tp,1),purity=tp/max(total,1),coverage=tp/count,
                    false_positive=total-tp,false_negative=count-tp,unknown_selected=int((mask&~known).sum()),
                    complete_carrier_iou=tp/max(int(mask.sum())+count-tp,1),
                    complete_carrier_purity=tp/max(int(mask.sum()),1),
                    unlabelled_fraction_of_selection=int((mask&~known).sum())/max(int(mask.sum()),1))
            rows.append(record);scene_rows.append(record)
        print(scene,{name:float(np.mean([r['arms'][name]['iou'] for r in scene_rows]))
            for name in ['anchor','finest_geometric','learned']},flush=True)
    report=dict(condition=design['condition'],rooms=scenes,queries=len(rows),
        positive_point='one deterministic uniformly sampled labelled vertex per instance; all annotated instances retained',
        evaluation_domain='all labelled vertices; unlabelled vertices remain unknown and their selected count is reported',
        method_saw_instance_ids=False,checkpoint_fitted_on_validation=False,
        benchmark_labels_used=False,rows=rows,scene_macro={},per_scene={})
    for name in ['anchor','finest_geometric','learned']:
        per_scene={scene:{k:float(np.mean([r['arms'][name][k] for r in rows if r['scene']==scene]))
            for k in ['iou','purity','coverage']} for scene in scenes}
        report['per_scene'][name]=per_scene
        report['scene_macro'][name]={k:float(np.mean([r[k] for r in per_scene.values()])) for k in ['iou','purity','coverage']}
    report['complete_carrier_scene_macro']={name:{k:float(np.mean([
        np.mean([row['arms'][name][k] for row in rows if row['scene']==scene]) for scene in scenes]))
        for k in ['complete_carrier_iou','complete_carrier_purity','unlabelled_fraction_of_selection']}
        for name in ['anchor','finest_geometric','learned']}
    report['complete_carrier_contract']='complete annotated target mask; unlabelled selections outside that mask count as errors only in this additional output metric, not as new training negatives'
    (a.root/a.output_name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['scene_macro'],indent=2),flush=True)


if __name__=='__main__':main()
