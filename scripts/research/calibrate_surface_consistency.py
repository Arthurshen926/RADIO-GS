"""Select graph strength on independent calibration rooms, not benchmark masks."""
import hashlib
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from fit_conditional_instance_membership import read_scene
from radio_gs.querying.conditional_element_membership import ConditionalElementMembership
from radio_gs.querying.surface_consistency import SurfaceConsistency

DATA=Path('output/learned_instance_additions');ROOT=Path('output/surface_membership_study')


def measure(probability,truth,known,threshold):
    selected=probability>threshold;tp=int((selected&truth).sum());size=int(selected.sum());target=int(truth.sum())
    return dict(iou=tp/max(size+target-tp,1),purity=tp/max(size,1),coverage=tp/target,
        labelled_domain_iou=tp/max(int((selected&known).sum())+target-tp,1))


def evaluate(scenes,settings,solver_kind='laplacian',unknown_neutral=False):
    output={}
    for scene in scenes:
        f,labels=read_scene(DATA,scene);folder=DATA/'maps'/scene;known=labels>=0
        reader=ConditionalElementMembership(f.membership,f.visibility,f.frames,np.load(folder/'mesh_xyz.npy'),f.descriptors,
            DATA/'conditional_query_elements.joblib',np.load(folder/'element_appearance.npy',mmap_mode='r'),
            np.load(folder/'element_appearance_observation.npy'))
        solvers={name:SurfaceConsistency(sparse.load_npz(ROOT/'graphs'/scene/f'{kind}.npz'),strength,solver_kind,unknown_neutral)
                 for name,(kind,strength) in settings.items()}
        rows=[]
        for instance in np.unique(labels[known]):
            truth=labels==instance;ids=np.flatnonzero(truth)
            seed=int.from_bytes(hashlib.sha256(f'{scene}/{instance}/positive-point'.encode()).digest()[:8],'little')
            anchor=int(ids[np.random.default_rng(seed).integers(len(ids))])
            region=reader.evidence.seed(None,anchor);probability=np.zeros(len(labels),np.float32)
            if region is not None:
                vertices,features=reader.evidence.encode(region,anchor)
                probability[vertices]=reader.model['estimator'].predict_proba(features)[:,1];probability[anchor]=1.
            row=dict(instance=int(instance),anchor=anchor,observed=region is not None,
                base=measure(probability,truth,known,reader.model['threshold']),arms={})
            for name,solver in solvers.items():
                refined,detail=solver.refine(probability,vertices,anchor,threshold=reader.model['threshold']) if region is not None else (probability,dict(no_anchor_observation=True))
                row['arms'][name]=dict(**measure(refined,truth,known,reader.model['threshold']),solver=detail)
            rows.append(row)
        macro={name:{k:float(np.mean([r['base'][k] if name=='base' else r['arms'][name][k] for r in rows]))
            for k in ['iou','purity','coverage','labelled_domain_iou']} for name in ['base',*settings]}
        output[scene]=dict(rows=rows,macro=macro);print(scene,macro,flush=True)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--solver',choices=['laplacian','cut'],default='laplacian')
    parser.add_argument('--evaluate-scenes',nargs='+',help='Evaluate frozen strengths on additional rooms, without calibration')
    parser.add_argument('--unknown-neutral',action='store_true',help='Zero unary outside evidence on the observed surface graph')
    args=parser.parse_args();prefix='cut_' if args.solver=='cut' else ''
    if args.evaluate_scenes:
        config=json.loads((ROOT/(prefix+'readout.json')).read_text())
        settings={kind:(kind,strength) for kind,strength in config['strengths'].items()}
        rows=evaluate(args.evaluate_scenes,settings,args.solver,args.unknown_neutral)
        report=dict(configuration=config,rooms=args.evaluate_scenes,per_scene=rows,
            macro={name:{k:float(np.mean([scene['macro'][name][k] for scene in rows.values()]))
                for k in ['iou','purity','coverage','labelled_domain_iou']} for name in ['base',*settings]},
            strengths_refitted=False,unknown_neutral=args.unknown_neutral,
            previously_opened_for_other_membership_models=True,independent_final_test=False)
        suffix='unknown_confirmation.json' if args.unknown_neutral else 'confirmation.json'
        (ROOT/(prefix+suffix)).write_text(json.dumps(report,indent=2)+'\n')
        print(report['macro'],flush=True);return
    design=json.loads((DATA/'design.json').read_text())
    strengths=[0.,1.,4.,16.]
    settings={f'{kind}_{strength:g}':(kind,strength) for kind in ['fixed','learned'] for strength in strengths}
    calibration=evaluate(design['calibration'],settings,args.solver)
    mean={name:float(np.mean([scene['macro'][name]['iou'] for scene in calibration.values()])) for name in settings}
    chosen={kind:max(strengths,key=lambda strength:mean[f'{kind}_{strength:g}']) for kind in ['fixed','learned']}
    # One strength per method, subsequently shared across registered instance
    # query modalities. No scene/target/query-specific selection or cutoff fit.
    config=dict(strengths=chosen,solver=args.solver,base_element_checkpoint=str(DATA/'conditional_query_elements.joblib'),
        base_element_threshold=float(joblib.load(DATA/'conditional_query_elements.joblib')['threshold']),
        calibration_rooms=design['calibration'],calibration_metric='complete-carrier actual-set scene-macro IoU',
        graph_training=str(ROOT/'surface_affinity.joblib'),benchmark_labels_used=False,
        class_sets_or_parts_not_assumed_equivalent_to_whole_instances=True)
    (ROOT/(prefix+'readout.json')).write_text(json.dumps(config,indent=2)+'\n')
    validation=evaluate(design['validation'],{kind:(kind,strength) for kind,strength in chosen.items()},args.solver)
    report=dict(configuration=config,strength_grid=strengths,calibration=calibration,calibration_means=mean,
        validation=validation,validation_macro={name:{k:float(np.mean([scene['macro'][name][k] for scene in validation.values()]))
            for k in ['iou','purity','coverage','labelled_domain_iou']} for name in ['base','fixed','learned']},
        validation_rooms_previously_opened=True,default_changed=False)
    (ROOT/(prefix+'calibration.json')).write_text(json.dumps(report,indent=2)+'\n');print(report['validation_macro'],flush=True)


if __name__=='__main__':main()
