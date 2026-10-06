"""Learn added-member quality from real, room-disjoint 3D instance labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import joblib
import numpy as np
from plyfile import PlyData
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor

from radio_gs.querying.conditional_region_membership import RegionPairFeatures,FEATURE_NAMES


def read_scene(root, scene):
    folder=root/'maps'/scene
    b=sparse.load_npz(folder/'membership.npz');v=sparse.load_npz(folder/'visibility.npz')
    frames=np.repeat(np.arange(v.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
    f=RegionPairFeatures(b,v,frames,np.load(folder/'mesh_xyz.npy'),np.load(folder/'appearance_descriptors.npy'))
    design=json.loads((root/'design.json').read_text())
    labels=np.asarray(PlyData.read(Path(design['data_root'])/'scans'/f'{scene}.ply')['vertex']['label'],np.int64)
    return f,labels


def compile_scene(root, scene):
    f,labels=read_scene(root,scene);known=labels>=0
    ids=np.flatnonzero(known);classes,codes=np.unique(labels[known],return_inverse=True)
    truth=sparse.csr_matrix((np.ones(len(ids),np.float32),(ids,codes)),shape=(len(labels),len(classes)))
    hist=(f.membership.T@truth).toarray();visible=(f.visibility.T@truth).toarray()
    known_area=hist.sum(1);purity=hist.max(1)/np.maximum(known_area,1)
    seeds=np.flatnonzero((purity>=.8)&(known_area>=.8*f.area)&(known_area>0))
    x=[];y=[];weights=[];addition_counts=[];seed_ids=[];candidate_ids=[]
    for seed in seeds:
        candidates,features,intersection=f.pairs(seed)
        if not len(candidates):continue
        scope=f.membership[:,seed];target=int(hist[seed].argmax())
        shared_known=(f.membership.T@scope.multiply(known[:,None])).toarray().ravel()[candidates]
        shared_target=(f.membership.T@scope.multiply(truth[:,target])).toarray().ravel()[candidates]
        added_known=np.maximum(known_area[candidates]-shared_known,0)
        added_target=np.maximum(hist[candidates,target]-shared_target,0)
        p=hist[candidates,target]/np.maximum(known_area[candidates],1)
        r=hist[candidates,target]/np.maximum(visible[f.frames[candidates],target],1)
        added=np.divide(added_target,added_known,out=p.copy(),where=added_known>0)
        valid=(known_area[candidates]>=.8*f.area[candidates])&(known_area[candidates]>0)
        x.append(features[valid]);y.append(np.column_stack([p,r,added])[valid])
        addition_counts.append(np.column_stack([added_target,added_known])[valid])
        weights.append(np.full(valid.sum(),1/max(valid.sum(),1),np.float32))
        seed_ids.append(np.full(valid.sum(),seed,np.int32));candidate_ids.append(candidates[valid])
    output=root/'training';output.mkdir(exist_ok=True)
    values=dict(x=np.concatenate(x),y=np.concatenate(y).clip(0,1),weight=np.concatenate(weights),
        addition=np.concatenate(addition_counts),seed=np.concatenate(seed_ids),candidate=np.concatenate(candidate_ids))
    values['weight']/=max(values['weight'].sum(),1)
    np.savez_compressed(output/f'{scene}.npz',**values)
    print('compiled labels',scene,'seeds',len(seeds),'pairs',len(values['x']),flush=True)


def admission_metrics(y, addition, admitted):
    mass=addition[admitted,1].sum()
    return dict(pairs=int(admitted.sum()),added_labelled_elements=float(mass),
        added_purity=float(addition[admitted,0].sum()/mass) if mass else 0.,
        pair_macro_added_purity=float(y[admitted,2].mean()) if admitted.any() else 0.,
        candidate_purity=float(y[admitted,0].mean()) if admitted.any() else 0.,
        visible_recall=float(y[admitted,1].mean()) if admitted.any() else 0.)


def fit(root, design):
    rows=[np.load(root/'training'/f'{s}.npz') for s in design['fit']]
    x=np.concatenate([r['x'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);weight=np.concatenate([r['weight'] for r in rows])
    start=time.monotonic()
    model=ExtraTreesRegressor(n_estimators=64,max_leaf_nodes=256,min_samples_leaf=32,
        max_features=1.,n_jobs=4,random_state=20261005)
    model.fit(x,y,sample_weight=weight)
    calibration=[np.load(root/'training'/f'{s}.npz') for s in design['calibration']]
    cx=np.concatenate([r['x'] for r in calibration]);cy=np.concatenate([r['y'] for r in calibration])
    addition=np.concatenate([r['addition'] for r in calibration]);pred=model.predict(cx)
    eligible=np.flatnonzero(addition[:,1]>0);order=eligible[np.argsort(-pred[eligible,2],kind='stable')]
    point_precision=np.cumsum(addition[order,0])/np.maximum(np.cumsum(addition[order,1]),1)
    pair_precision=np.cumsum(cy[order,2])/np.arange(1,len(order)+1)
    # Calibrate one global admission value on the separate four rooms. Ties are
    # admitted together; no benchmark scores are consulted. This is a precision
    # target for pair proposals, not a guarantee for final mesh selections.
    ends=np.r_[np.diff(pred[order,2])!=0,True]
    acceptable=np.flatnonzero((point_precision>=.9)&(pair_precision>=.9)&ends)
    threshold=float(pred[order[acceptable[-1]],2]) if len(acceptable) else 1.000001
    condition='additional 3D whole-instance supervision;24 official-training rooms rendered to12 RGB views each'
    checkpoint=dict(estimator=model,addition_threshold=threshold,feature_names=FEATURE_NAMES,
        condition=condition,fit=design['fit'],calibration=design['calibration'],validation=design['validation'],
        target_order=['candidate_purity','candidate_visible_recall','added_element_purity'])
    joblib.dump(checkpoint,root/'conditional_membership.joblib',compress=3)
    report=dict(training_pairs=len(x),calibration_pairs=len(cx),seconds=time.monotonic()-start,
        addition_threshold=threshold,feature_importance=dict(zip(FEATURE_NAMES,model.feature_importances_.tolist())),
        calibration=admission_metrics(cy,addition,(pred[:,2]>=threshold)&(addition[:,1]>0)),validation={})
    for scene in design['validation']:
        row=np.load(root/'training'/f'{scene}.npz');p=model.predict(row['x']);adds=row['addition'][:,1]>0
        heuristic=(row['x'][:,2]>=.5)&(row['x'][:,3]>=.5)&(row['x'][:,4]>=.5)&(row['x'][:,6]==0)&adds
        report['validation'][scene]=dict(pairs=len(p),
            mean_absolute_error=np.abs(p-row['y']).mean(0).tolist(),
            geometric_admission=admission_metrics(row['y'],row['addition'],heuristic),
            learned_admission=admission_metrics(row['y'],row['addition'],(p[:,2]>=threshold)&adds))
    (root/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--stage',choices=['compile','fit'],required=True);a=p.parse_args()
    design=json.loads((a.root/'design.json').read_text())
    if a.stage=='fit':fit(a.root,design)
    else:
        scenes=design['fit']+design['calibration']+design['validation']
        with ThreadPoolExecutor(4) as pool:list(pool.map(lambda s:compile_scene(a.root,s),scenes))


if __name__=='__main__':main()
