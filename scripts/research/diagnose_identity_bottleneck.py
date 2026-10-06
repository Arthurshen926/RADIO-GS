"""Counterfactual seed interventions after an unchanged automatic cohort.

Evaluator labels are used here only. No counterfactual is exported as a method
prediction. Identity-only repair keeps the native candidate order and decoder;
it removes regions whose majority is not the requested instance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.querying.instance_observation_scope import InstanceObservationScope


def measure(selected, truth):
    tp=int(np.count_nonzero(selected & truth)); area=int(selected.sum())
    return dict(iou=tp/max(int(truth.sum())+area-tp,1),
                purity=tp/max(area,1),coverage=tp/max(int(truth.sum()),1),
                selected_elements=area)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/identity_bottleneck_20261005'))
    args=p.parse_args();root=args.root;system=root/'baseline_system'
    complete=json.loads((system/'predictions_complete.json').read_text())
    if complete['jobs']!=9:raise ValueError('Complete automatic cohort required before label intervention')
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    dataset=json.loads((package/'targets.json').read_text())
    capacity=json.loads(Path('output/observation_hypotheses_study/label_capacity.json').read_text())
    best_scope={(r['scene_id'],r['instance_id']):r['best_label_selected']['one_hop_scope']['candidate'] for r in capacity['cases']}
    cases=[]
    for domain in dataset['scene_domains']:
        scene=domain['scene_id'];folder=root/'maps'/scene
        membership=sparse.load_npz(folder/'membership.npz').tocsc()
        visibility=sparse.load_npz(folder/'visibility.npz')
        frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
        reader=InstanceObservationScope(membership,visibility,frames)
        run=system/'uqis'/scene/'query/registered'
        evidence=np.load(run/'identity_evidence.npz')
        columns={q:i for i,q in enumerate(evidence['query_ids'])}
        labels=np.load(package/domain['mesh_instance_ids_path'])
        zeros=np.zeros(len(labels),np.float32)
        for target in dataset['targets']:
            if target['scene_id']!=scene:continue
            truth=labels==target['instance_id']
            intersection=np.asarray(membership.T@truth.astype(np.float32)).ravel()
            purity=intersection/np.maximum(reader.area,1)
            region_iou=intersection/np.maximum(reader.area+truth.sum()-intersection,1)
            for modality in ['text','image']:
                qid=target['queries'][modality]
                identity=evidence['identity_response'][:,columns[qid]]
                available=np.flatnonzero(reader.area>0)
                order=available[np.argsort(-identity[available],kind='stable')]
                actual_seed=int(order[0]);correct=order[purity[order]>.5]
                ids,counts=np.unique(labels[membership[:,actual_seed].indices],return_counts=True)
                dominant=int(ids[counts.argmax()])
                identity_error=('correct' if dominant==target['instance_id'] else
                    'same_class_distractor' if dominant in target['same_class_distractor_instance_ids'] else 'other_or_unlabelled')
                seeds=dict(original=actual_seed,
                    identity_only=int(correct[0]) if len(correct) else None,
                    best_identity_compatible_region=int(correct[np.argmax(region_iou[correct])]) if len(correct) else None,
                    best_single_region=int(region_iou.argmax()),
                    best_decoded_scope=best_scope[(scene,target['instance_id'])])
                row=dict(scene_id=scene,instance_id=target['instance_id'],modality=modality,query_id=qid,
                    supported_regions=len(order),majority_correct_seed_rank=int(np.flatnonzero(purity[order]>.5)[0]+1) if len(correct) else None,
                    original_seed_purity=float(purity[actual_seed]),
                    original_seed_coverage=float(intersection[actual_seed]/truth.sum()),interventions={})
                row.update(dominant_seed_instance=dominant,identity_error_type=identity_error,
                    language_tags=target['language_tags'],raw_semantic_label=target['raw_semantic_label'])
                for name,seed in seeds.items():
                    if seed is None:
                        row['interventions'][name]=dict(seed=None,**measure(np.zeros(len(truth),bool),truth));continue
                    _,_,selection,detail=reader.infer(identity,zeros,seed=seed)
                    selected=selection['identity_scope']
                    members=detail['matched_regions']
                    support=np.asarray(membership[:,members].sum(1)).ravel()>0
                    row['interventions'][name]=dict(seed=int(seed),seed_purity=float(purity[seed]),
                        **measure(selected,truth),positive_candidate_coverage=float(np.mean(support[truth])),
                        matched_views=detail['matched_views'])
                    if name=='original':
                        actual=np.load(run/'identity_scope/selections'/f'{qid}.npy')
                        row['original_set_changed']=int(np.count_nonzero(actual!=selected))
                cases.append(row)
        print(scene,'complete',flush=True)
    summary={}
    for modality in ['text','image']:
        rows=[r for r in cases if r['modality']==modality];scenes=sorted({r['scene_id'] for r in rows})
        interventions={name:{key:float(np.mean([np.mean([r['interventions'][name][key] for r in rows if r['scene_id']==scene]) for scene in scenes]))
            for key in ['iou','purity','coverage']} for name in seeds}
        ranks=[r['majority_correct_seed_rank'] for r in rows]
        summary[modality]=dict(interventions=interventions,
            correct_majority_in_top={str(k):sum(rank is not None and rank<=k for rank in ranks) for k in [1,2,5,10,20,50,100]},
            no_majority_region=sum(rank is None for rank in ranks),
            dominant_seed_identity={kind:sum(r['identity_error_type']==kind for r in rows)
                for kind in ['correct','same_class_distractor','other_or_unlabelled']},
            original_sets_changed=sum(r['original_set_changed'] for r in rows))
    report=dict(automatic_method_result=False,labels_used_only_after_predictions=True,
        identity_only_intervention='highest original identity score among target-majority regions; unchanged decoder',
        best_scope_intervention='label-selected existing finite one-hop family; not a general method ceiling',
        summary=summary,cases=cases)
    (root/'seed_counterfactuals.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
