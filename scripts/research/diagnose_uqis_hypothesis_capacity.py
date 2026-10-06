"""Label-selected candidate capacity AFTER automatic predictions complete.

This is a diagnosis over a finite candidate family, not an automatic method,
not a score available to the query process, and not a ceiling for arbitrary
future reconstruction methods. It never writes predictions into a method run.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse


def selected_scopes(membership, visibility, region_frames, hypotheses):
    positive=(membership@hypotheses.T).tocsc()
    rows=[];columns=[]
    for hypothesis in range(hypotheses.shape[0]):
        lo,hi=positive.indptr[hypothesis:hypothesis+2]
        vertices=positive.indices[lo:hi]
        if not len(vertices):continue
        row=hypotheses[hypothesis];views=region_frames[row.indices]
        known=np.asarray(visibility[vertices][:,views]@row.data).reshape(-1)
        selected=vertices[positive.data[lo:hi]>.5*known]
        rows.extend(selected);columns.extend([hypothesis]*len(selected))
    return sparse.csc_matrix((np.ones(len(rows),np.float32),(rows,columns)),shape=positive.shape)


def candidate_capacity(matrix, truths):
    area=np.asarray(matrix.sum(0)).reshape(-1)
    intersection=np.asarray(matrix.T@truths)
    truth_area=truths.sum(0)
    iou=intersection/np.maximum(area[:,None]+truth_area[None,:]-intersection,1.)
    best=iou.argmax(0)
    return [dict(candidate=int(h),iou=float(iou[h,t]),purity=float(intersection[h,t]/max(area[h],1)),
        coverage=float(intersection[h,t]/max(truth_area[t],1)),selected_elements=int(area[h])) for t,h in enumerate(best)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/observation_hypotheses_study/label_capacity.json'))
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    for root in ['observation_hypotheses_study','view_exclusive_tracks_study']:
        complete=read(Path('output')/root/'system/predictions_complete.json')
        if complete['jobs']!=9:raise ValueError('Complete automatic cohort must precede this label diagnosis')
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator');evaluation=read(package/'targets.json')
    cases=[];start=time.monotonic()
    for domain in evaluation['scene_domains']:
        scene=domain['scene_id'];root=Path('output/view_exclusive_tracks_study/maps')/scene
        membership=sparse.load_npz(root/'membership.npz').astype(np.float32).tocsr()
        visibility=sparse.load_npz(root/'visibility.npz').astype(np.float32).tocsr()
        frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(root/'region_offsets.npy')))
        labels=np.load(package/domain['mesh_instance_ids_path'])
        targets=[t for t in evaluation['targets'] if t['scene_id']==scene]
        truths=np.stack([labels==t['instance_id'] for t in targets],axis=1).astype(np.float32)
        scores={'raw_region':candidate_capacity(membership,truths)}
        for name,file in [('one_hop_scope','scope_hypotheses.npz'),('view_exclusive_track','instance_tracks.npz')]:
            hypotheses=sparse.load_npz(root/file)
            selected=selected_scopes(membership,visibility,frames,hypotheses)
            scores[name]=candidate_capacity(selected,truths)
            del selected
        for index,target in enumerate(targets):
            cases.append(dict(scene_id=scene,instance_id=target['instance_id'],
                best_label_selected={name:value[index] for name,value in scores.items()}))
        print(scene,'finished',len(targets),'targets',flush=True)
    scene_names=sorted({r['scene_id'] for r in cases});summary={}
    for method in scores:
        per_scene={s:{key:float(np.mean([r['best_label_selected'][method][key] for r in cases if r['scene_id']==s]))
                      for key in ['iou','purity','coverage']} for s in scene_names}
        summary[method]=dict(scene_macro={key:float(np.mean([v[key] for v in per_scene.values()])) for key in ['iou','purity','coverage']},per_scene=per_scene)
    a.output.write_text(json.dumps(dict(role='post-prediction label-selected finite-candidate diagnosis only',
        automatic_method_result=False,query_process_can_access=False,all_automatic_predictions_finished_before_label_read=True,
        automatic_prediction_cohorts_required=['observation_hypotheses_study','view_exclusive_tracks_study'],
        summary=summary,cases=cases,seconds=time.monotonic()-start),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
