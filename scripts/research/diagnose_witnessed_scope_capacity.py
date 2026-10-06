"""After automatic prediction: measure the finite scope family's capacity.

This label-selected diagnosis cannot be used by the method and never supplies
query answers. It asks whether third-view evidence can assemble a whole object
at all, separately from automatic scope and instance choice.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from scripts.research.diagnose_uqis_hypothesis_capacity import candidate_capacity


def main():
    root=Path('output/witnessed_scope_study')
    for mode in ['majority','union']:
        complete=json.loads((root/mode/'system/predictions_complete.json').read_text())
        assert complete['jobs']==9
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    data=json.loads((package/'targets.json').read_text())
    cases=[];start=time.monotonic()
    for domain in data['scene_domains']:
        scene=domain['scene_id'];folder=root/'maps'/scene
        membership=sparse.load_npz(folder/'membership.npz').astype(np.float32)
        graph=sparse.load_npz(folder/'scope_witnesses.npz')
        union=(membership@graph.T).tocsc();union.data[:]=1
        labels=np.load(package/domain['mesh_instance_ids_path'])
        targets=[t for t in data['targets'] if t['scene_id']==scene]
        truth=np.stack([labels==t['instance_id'] for t in targets],axis=1).astype(np.float32)
        scores=candidate_capacity(union,truth)
        for target,score in zip(targets,scores):
            cases.append(dict(scene_id=scene,instance_id=target['instance_id'],**score))
        print(scene,'diagnosed',len(targets),flush=True)
    scenes=sorted({r['scene_id'] for r in cases})
    per_scene={s:{k:float(np.mean([r[k] for r in cases if r['scene_id']==s])) for k in ['iou','purity','coverage']} for s in scenes}
    report=dict(role='post-prediction label-selected finite-candidate diagnosis; not an automatic score',
        scene_macro={k:float(np.mean([r[k] for r in per_scene.values()])) for k in ['iou','purity','coverage']},
        per_scene=per_scene,cases=cases,seconds=time.monotonic()-start)
    (root/'label_capacity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['scene_macro'])


if __name__=='__main__':main()
