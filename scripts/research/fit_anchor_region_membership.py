"""Whole-instance regional targets refer to the exact positive point throughout."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from fit_conditional_instance_membership import read_scene
from radio_gs.querying.conditional_region_membership import FEATURE_NAMES, ANCHOR_REGION_FEATURE_NAMES


def compile_scene(root,scene):
    f,labels=read_scene(root,scene);known=labels>=0
    appearance=np.load(root/'maps'/scene/'element_appearance.npy',mmap_mode='r')
    anchors=f.membership.tocsr();known_area=np.asarray(f.membership.T@known.astype(np.float32)).ravel()
    rows=[];targets=[];weights=[];details=[]
    for instance in np.unique(labels[known]):
        truth=labels==instance;members=np.flatnonzero(truth)
        seed=int.from_bytes(hashlib.sha256(f'{scene}/{instance}/positive-point'.encode()).digest()[:8],'little')
        anchor=int(members[np.random.default_rng(seed).integers(len(members))])
        available=anchors[anchor].indices
        if not len(available):continue
        region=int(available[np.argmin(f.area[available])])
        ids,x,shared=f.anchor_pairs(region,anchor,appearance)
        area=np.asarray(f.membership[:,ids].T@truth.astype(np.float32)).ravel()
        visible=np.asarray(f.visibility.T@truth.astype(np.float32)).ravel()[f.frames[ids]]
        scope=f.membership[:,region]
        shared_known=(f.membership[:,ids].T@scope.multiply(known[:,None])).toarray().ravel()
        shared_target=(f.membership[:,ids].T@scope.multiply(truth[:,None])).toarray().ravel()
        added_known=np.maximum(known_area[ids]-shared_known,0)
        added_target=np.maximum(area-shared_target,0)
        purity=area/np.maximum(known_area[ids],1)
        recall=area/np.maximum(visible,1)
        added=np.divide(added_target,added_known,out=purity.copy(),where=added_known>0)
        valid=(known_area[ids]>=.8*f.area[ids])&(known_area[ids]>0)
        rows.append(x[valid]);targets.append(np.column_stack([purity,recall,added])[valid])
        weights.extend([1/max(int(valid.sum()),1)]*int(valid.sum()))
        details.append(dict(instance=int(instance),anchor=anchor,seed_region=region,
            target_fraction_in_seed=float(purity[-1]),pairs=int(valid.sum())))
    folder=root/'anchor_region_training';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{scene}.npz',x=np.concatenate(rows),y=np.concatenate(targets).clip(0,1),
                        weight=np.asarray(weights)/max(len(rows),1))
    (folder/f'{scene}.json').write_text(json.dumps(details,indent=2)+'\n')
    print('point-conditioned region labels',scene,len(np.concatenate(rows)),flush=True)


def main():
    root=Path('output/learned_instance_additions');design=json.loads((root/'design.json').read_text())
    with ThreadPoolExecutor(3) as pool:list(pool.map(lambda s:compile_scene(root,s),design['fit']))
    data=[np.load(root/'anchor_region_training'/f'{s}.npz') for s in design['fit']]
    model=clone(joblib.load(root/'conditional_membership.joblib')['estimator'])
    model.fit(np.concatenate([d['x'] for d in data]),np.concatenate([d['y'] for d in data]),
              sample_weight=np.concatenate([d['weight'] for d in data]))
    joblib.dump(dict(estimator=model,feature_names=FEATURE_NAMES+ANCHOR_REGION_FEATURE_NAMES,
        fit=design['fit'],condition='exact positive-point whole instance; impure seed masks included; no fabricated seed purity'),
        root/'conditional_anchor_membership.joblib',compress=3)


if __name__=='__main__':main()
