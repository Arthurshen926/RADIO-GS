"""Second bounded mechanism: supervise the final elements on independent rooms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier

from fit_conditional_instance_membership import read_scene
from radio_gs.querying.conditional_element_membership import ElementMembershipEvidence,FEATURE_NAMES,LOCAL_FEATURE_NAMES,ANCHOR_FEATURE_NAMES,JOINT_FEATURE_NAMES


def episodes(root,scene,region_model,local=False,anchor_condition=False,joint_observations=False,query_region=False,exclude_unlabelled=True):
    f,labels=read_scene(root,scene)
    evidence=ElementMembershipEvidence(f.membership,f.visibility,f.frames,
        np.load(root/'maps'/scene/'mesh_xyz.npy'),f.descriptors,region_model,
        np.load(root/'maps'/scene/'element_appearance.npy',mmap_mode='r') if local else None,
        np.load(root/'maps'/scene/'element_appearance_observation.npy') if local else None,anchor_condition,joint_observations,query_region)
    for instance in np.unique(labels[labels>=0]):
        target=np.flatnonzero(labels==instance)
        seed=int.from_bytes(hashlib.sha256(f'{scene}/{instance}/positive-point'.encode()).digest()[:8],'little')
        anchor=int(target[np.random.default_rng(seed).integers(len(target))])
        region=evidence.seed(None,anchor)
        if region is None:
            yield instance,len(target),None,None,None,None
            continue
        vertices,x=evidence.encode(region,anchor)
        # Unknown labels have no training target. At evaluation, however, the
        # deployed decoder can select these vertices and they belong in the
        # complete-carrier false-positive denominator, as in UQIS.
        known=labels[vertices]>=0 if exclude_unlabelled else np.ones(len(vertices),bool)
        y=labels[vertices]==instance
        anchor_local=int(np.flatnonzero(vertices[known]==anchor)[0])
        yield instance,len(target),x[known],y[known],seed,anchor_local


def compile_scene(root,scene,model,local=False,anchor_condition=False,joint_observations=False,cross_fit=False,query_region=False):
    rows=[];targets=[];weights=[];counts=[]
    for instance,count,x,y,seed,anchor_local in episodes(root,scene,model,local,anchor_condition,joint_observations,query_region):
        if x is None:continue
        rng=np.random.default_rng(seed);sample=[];weight=[]
        # This declared SGD-style training budget never truncates mapping views,
        # proposals or inference vertices. Restore class mass through weights.
        for label in [False,True]:
            ids=np.flatnonzero(y==label)
            selected=rng.choice(ids,min(len(ids),1024),replace=False)
            sample.extend(selected.tolist())
            weight.extend([len(ids)/max(len(selected),1)/len(y)]*len(selected))
        rows.append(x[sample]);targets.append(y[sample]);weights.extend(weight);counts.append(count)
    folder=root/('element_query_training' if query_region else 'element_joint_oof_training' if cross_fit else 'element_joint_training' if joint_observations else 'element_anchor_training' if anchor_condition else 'element_local_training' if local else 'element_training');folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{scene}.npz',x=np.concatenate(rows),y=np.concatenate(targets),
                        weight=np.asarray(weights)/max(len(rows),1))
    print('element samples',scene,len(np.concatenate(rows)),'episodes',len(rows),flush=True)


def cross_fit_regions(root,scenes,region,query_region=False):
    """The downstream learner sees upstream predictions on unseen rooms.

    Labels used by the region head must not re-enter an element's training input
    through an in-sample prediction. All rescans share a room fold. At deployment
    the region model still uses all 24 fit rooms; calibration stays separate.
    """
    rooms=sorted({s.split('_')[0] for s in scenes},key=lambda r:hashlib.sha256(('membership-oof/'+r).encode()).hexdigest())
    folds={room:i%4 for i,room in enumerate(rooms)}
    data={s:np.load(root/('anchor_region_training' if query_region else 'training')/f'{s}.npz') for s in scenes}
    models={};diagnosis=[]
    for fold in range(4):
        train=[s for s in scenes if folds[s.split('_')[0]]!=fold]
        held=[s for s in scenes if folds[s.split('_')[0]]==fold]
        model=clone(region).set_params(n_jobs=4)
        model.fit(np.concatenate([data[s]['x'] for s in train]),
                  np.concatenate([data[s]['y'] for s in train]),
                  sample_weight=np.concatenate([data[s]['weight'] for s in train]))
        for scene in held:
            row=data[scene];truth=row['y'];inside=region.predict(row['x']);outside=model.predict(row['x'])
            diagnosis.append(dict(scene=scene,fold=fold,
                in_sample_mae=np.abs(inside-truth).mean(0).tolist(),
                out_of_fold_mae=np.abs(outside-truth).mean(0).tolist()))
            models[scene]=model
        model.n_jobs=1
        joblib.dump(dict(estimator=model,fit=train,held=held),root/(('anchor_' if query_region else '')+f'region_fold_{fold}.joblib'),compress=3)
        print('cross-fit regions',fold,'fit rooms',len(train),'held rooms',len(held),flush=True)
    (root/('anchor_region_cross_fit.json' if query_region else 'region_cross_fit.json')).write_text(json.dumps(dict(folds=folds,rows=diagnosis,
        condition='out-of-room upstream predictions for downstream training; no calibration or benchmark labels'),indent=2)+'\n')
    return models


def evaluate(root,scenes,region_model,model,thresholds,local=False,anchor_condition=False,joint_observations=False,query_region=False,exclude_unlabelled=False):
    results={}
    for scene in scenes:
        rows=[]
        for instance,count,x,y,seed,anchor_local in episodes(root,scene,region_model,local,anchor_condition,joint_observations,query_region,exclude_unlabelled):
            if x is None:
                values=np.zeros((len(thresholds),3))
            else:
                p=model.predict_proba(x)[:,1];p[anchor_local]=1.
                values=[]
                for threshold in thresholds:
                    selected=p>threshold;tp=int((selected&y).sum());total=int(selected.sum())
                    values.append([tp/max(total+count-tp,1),tp/max(total,1),tp/count])
                values=np.array(values)
            rows.append(dict(instance=int(instance),metrics=values.tolist()))
        results[scene]=dict(rows=rows,macro=np.mean([r['metrics'] for r in rows],axis=0).tolist())
        print('element evaluation',scene,results[scene]['macro'],flush=True)
    macro=np.mean([s['macro'] for s in results.values()],axis=0)
    return results,macro


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--local-appearance',action='store_true')
    p.add_argument('--anchor-condition',action='store_true')
    p.add_argument('--joint-observations',action='store_true')
    p.add_argument('--query-region',action='store_true',help='Exact anchor conditions both regional targets and element predictions')
    p.add_argument('--cross-fit',action='store_true',help='Room-disjoint upstream predictions in joint-evidence training')
    p.add_argument('--stage',choices=['compile','fit'],required=True);a=p.parse_args();root=a.root
    a.cross_fit=a.cross_fit or a.query_region
    a.joint_observations=a.joint_observations or a.cross_fit
    a.anchor_condition=a.anchor_condition or a.joint_observations
    a.local_appearance=a.local_appearance or a.anchor_condition
    design=json.loads((root/'design.json').read_text());region=joblib.load(root/('conditional_anchor_membership.joblib' if a.query_region else 'conditional_membership.joblib'))['estimator']
    if a.stage=='compile':
        region.n_jobs=1
        models=cross_fit_regions(root,design['fit'],region,a.query_region) if a.cross_fit else dict.fromkeys(design['fit'],region)
        with ThreadPoolExecutor(3) as pool:list(pool.map(lambda s:compile_scene(root,s,models[s],a.local_appearance,a.anchor_condition,a.joint_observations,a.cross_fit,a.query_region),design['fit']))
        return
    folder='element_query_training' if a.query_region else 'element_joint_oof_training' if a.cross_fit else 'element_joint_training' if a.joint_observations else 'element_anchor_training' if a.anchor_condition else 'element_local_training' if a.local_appearance else 'element_training'
    data=[np.load(root/folder/f'{s}.npz') for s in design['fit']]
    x=np.concatenate([r['x'] for r in data]);y=np.concatenate([r['y'] for r in data]);weight=np.concatenate([r['weight'] for r in data])
    model=ExtraTreesClassifier(n_estimators=64,max_leaf_nodes=512,min_samples_leaf=32,
                              max_features=1.,n_jobs=4,random_state=20261005)
    model.fit(x,y,sample_weight=weight)
    thresholds=np.arange(.05,1,.05)
    calibration,macro=evaluate(root,design['calibration'],region,model,thresholds,a.local_appearance,a.anchor_condition,a.joint_observations,a.query_region)
    threshold=float(thresholds[np.argmax(macro[:,0])])
    condition='independent3D whole-instance supervision; final element membership on24 rendered official-training rooms'
    names=FEATURE_NAMES+(LOCAL_FEATURE_NAMES if a.local_appearance else [])+(ANCHOR_FEATURE_NAMES if a.anchor_condition else [])+(JOINT_FEATURE_NAMES if a.joint_observations else [])
    checkpoint='conditional_query_elements.joblib' if a.query_region else 'conditional_joint_oof_elements.joblib' if a.cross_fit else 'conditional_joint_elements.joblib' if a.joint_observations else 'conditional_anchor_elements.joblib' if a.anchor_condition else 'conditional_local_elements.joblib' if a.local_appearance else 'conditional_elements.joblib'
    joblib.dump(dict(estimator=model,region_estimator=region,threshold=threshold,feature_names=names,
        local_appearance=a.local_appearance,condition_on_anchor=a.anchor_condition,joint_observations=a.joint_observations,region_condition_on_anchor=a.query_region,
        upstream_training='room-disjoint four-fold predictions' if a.cross_fit else 'in-sample predictions',
        condition=condition,fit=design['fit'],calibration=design['calibration']),root/checkpoint,compress=3)
    validation,metrics=evaluate(root,design['validation'],region,model,[threshold],a.local_appearance,a.anchor_condition,a.joint_observations,a.query_region)
    report=dict(training_samples=len(x),features=names,threshold=threshold,
        calibration_selection_domain='complete deployment candidate domain; unlabelled selections count as false positives',
        calibration_thresholds=thresholds.tolist(),calibration=calibration,
        validation=validation,scene_macro=dict(zip(['iou','purity','coverage'],metrics[0].tolist())),
        calibration_only_threshold_selection=True,benchmark_labels_used=False,upstream_cross_fitted=a.cross_fit,region_condition_on_anchor=a.query_region,
        feature_importance=dict(zip(names,model.feature_importances_.tolist())))
    (root/('element_query_fit.json' if a.query_region else 'element_joint_oof_fit.json' if a.cross_fit else 'element_joint_fit.json' if a.joint_observations else 'element_anchor_fit.json' if a.anchor_condition else 'element_local_fit.json' if a.local_appearance else 'element_fit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['scene_macro']),flush=True)


if __name__=='__main__':main()
