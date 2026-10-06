"""Separate native retrieval from label-selected proposal capacity on two rooms."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.querying.observation_hypotheses import compile_hypotheses
from scripts.research.diagnose_uqis_hypothesis_capacity import selected_scopes


def measure(folder,scene,uniform_twelve=False):
    root=Path('output/referent_grounding_20261005')
    membership=sparse.load_npz(folder/'membership.npz').tocsr()
    visibility=sparse.load_npz(folder/'visibility.npz').tocsr()
    offsets=np.load(folder/'region_offsets.npy');frames=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
    descriptors=np.load(folder/'descriptors.npy').astype(np.float32)
    if uniform_twelve:
        views=np.linspace(0,visibility.shape[1]-1,12,dtype=int)
        ids=np.flatnonzero(np.isin(frames,views))
        membership=membership[:,ids];visibility=visibility[:,views];descriptors=descriptors[ids]
        frames=np.searchsorted(views,frames[ids])
    descriptors/=np.maximum(np.linalg.norm(descriptors,axis=1,keepdims=True),1e-8)
    text=np.load(root/'language'/(scene+'.npz'))['pooled'].astype(np.float32)
    text/=np.linalg.norm(text,axis=1,keepdims=True)
    area=np.asarray(membership.sum(0)).ravel();available=np.flatnonzero(area>0)
    identity=text@descriptors.T;seeds=available[identity[:,available].argmax(1)]
    hypotheses=compile_hypotheses(membership,visibility,frames)
    selections=selected_scopes(membership,visibility,frames,hypotheses)
    # Everything above is query/map-only. Instance annotations now score the
    # fixed native choices and, separately, the finite-family diagnostic.
    labels=np.load(root/'labels'/(scene+'.npy'))
    queries=json.loads((root/'language'/(scene+'.json')).read_text())
    ids=sorted({q['raw_object_id']+1 for q in queries});columns={x:i for i,x in enumerate(ids)}
    truth=np.stack([labels==i for i in ids],axis=1).astype(np.float32)
    intersection=np.asarray(selections.T@truth);scope_area=np.asarray(selections.sum(0)).ravel()
    iou=intersection/np.maximum(scope_area[:,None]+truth.sum(0)[None,:]-intersection,1)
    region_intersection=np.asarray(membership.T@truth)
    purity=region_intersection/np.maximum(area[:,None],1)
    covered=membership.getnnz(axis=1)>0
    metrics=[]
    for q,seed in zip(queries,seeds):
        column=columns[q['raw_object_id']+1]
        metrics.append(dict(record=q['record'],seed=int(seed),native_iou=float(iou[seed,column]),
            native_seed_purity=float(purity[seed,column]),finite_scope_oracle=float(iou[:,column].max())))
    return dict(views=visibility.shape[1],regions=membership.shape[1],queries=len(queries),targets=len(ids),
        target_mean_source_coverage=float(np.mean([(covered&(labels==i)).sum()/np.count_nonzero(labels==i) for i in ids])),
        targets_with_majority_candidate=int(np.count_nonzero(purity.max(0)>.5)),
        target_mean_finite_scope_oracle=float(iou.max(0).mean()),
        query_mean_native_iou=float(np.mean([x['native_iou'] for x in metrics])),
        native_majority_seed_queries=sum(x['native_seed_purity']>.5 for x in metrics),cases=metrics)


def main():
    root=Path('output/referent_grounding_20261005');control=root/'real_observation_control'
    report=dict(role='observation-quality diagnosis, not new benchmark result',model_fits=0,scenes={})
    for scene in ['scene0271_00','scene0540_00']:
        report['scenes'][scene]={
            'point_render_12':measure(root/'maps'/scene,scene),
            'real_rgbd_12':measure(control/'maps'/scene,scene,uniform_twelve=True),
            'real_rgbd_full':measure(control/'maps'/scene,scene)}
        print(scene,json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in report['scenes'][scene].items()}),flush=True)
    (control/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
