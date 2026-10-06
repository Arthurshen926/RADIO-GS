"""Score all paired queries on the same official mesh; no Core filtering."""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from . import MODALITIES
from .construct import write_json
from .metrics import evaluate_query_probabilities, scene_clustered_bootstrap


def aggregate(rows):
    keys=['average_precision','oracle_iou','fixed_iou_0.5','acc_at_iou_0.25','acc_at_iou_0.50','selected_purity','positive_coverage','same_class_distractor_iou','centroid_error_m']
    scenes=sorted({r['scene_id'] for r in rows})
    values={}
    for key in keys:
        values[key]={s:float(np.mean([r[key] for r in rows if r['scene_id']==s and r[key] is not None])) for s in scenes if any(r['scene_id']==s and r[key] is not None for r in rows)}
    return {'query_count':len(rows),'scene_count':len(scenes),
        'scene_macro':{k:float(np.mean(list(v.values()))) if v else None for k,v in values.items()},
        'scene_clustered_ci':{k:scene_clustered_bootstrap(v) if v else None for k,v in values.items()},
        'per_scene':{s:{k:v.get(s) for k,v in values.items()} for s in scenes}}


def score_predictions(targets, predictions, xyz, ids, modalities=MODALITIES):
    modalities=tuple(modalities)
    if not modalities or len(set(modalities))!=len(modalities) or not set(modalities)<=set(MODALITIES):
        raise ValueError('Declare distinct supported query modalities')
    expected={t['queries'][m] for t in targets for m in modalities}
    if set(predictions)!=expected:
        raise ValueError(f'Prediction set mismatch: {len(expected-set(predictions))} missing, {len(set(predictions)-expected)} extra')
    rows=[]
    for t in targets:
        for modality in modalities:
            qid=t['queries'][modality]
            scene=t['scene_id'];scores=predictions[qid]
            metrics=evaluate_query_probabilities(scores,target_instance_id=t['instance_id'],
                same_class_distractor_instance_ids=t['same_class_distractor_instance_ids'],
                mesh_instance_ids=ids[scene],mesh_xyz=xyz[scene])
            rows.append({'query_id':qid,'scene_id':scene,'instance_id':t['instance_id'],
                         'modality':modality,'language_tags':t.get('language_tags',[]),
                         'reference_difficulty_flags':t.get('reference_difficulty_flags',[]),**metrics})
    by_modality={m:aggregate([r for r in rows if r['modality']==m]) for m in modalities}
    scene_ids=sorted(xyz)
    complete=set(modalities)==set(MODALITIES)
    means={s:float(np.mean([by_modality[m]['per_scene'][s]['average_precision'] for m in MODALITIES])) for s in scene_ids} if complete else None
    paired=None
    if complete:
        index={r['query_id']:r for r in rows}
        target_rows=[]
        for target in targets:
            ap={m:index[target['queries'][m]]['average_precision'] for m in MODALITIES}
            target_rows.append(dict(scene_id=target['scene_id'],instance_id=target['instance_id'],
                modality_ap=ap,minimum_ap=min(ap.values()),
                geometric_mean_ap=float(np.prod(list(ap.values()))**.25),
                weakest_modalities=[m for m,v in ap.items() if v==min(ap.values())]))
        paired={key:scene_clustered_bootstrap({s:float(np.mean([r[key] for r in target_rows if r['scene_id']==s]))
                    for s in scene_ids}) for key in ['minimum_ap','geometric_mean_ap']}
        paired.update(per_target=target_rows,interpretation='same-target four-query robustness; secondary AP summaries, not calibrated mask accuracy',
                      primary_metrics_unchanged=True)
    tags=sorted({tag for t in targets for tag in t.get('language_tags',[])}) if 'text' in modalities else []
    language={tag:aggregate([r for r in rows if r['modality']=='text' and tag in r['language_tags']]) for tag in tags}
    scope_groups={}
    for group,marked in [('review_marked_annotation_group',True),('not_marked_as_group',False)]:
        cohort=[r for r in rows if ('annotation_group' in r['language_tags'])==marked]
        if cohort:
            scope_groups[group]={m:aggregate([r for r in cohort if r['modality']==m]) for m in modalities}
    image_difficulty={}
    if any(r['reference_difficulty_flags'] for r in rows) and 'image' in modalities:
        for group,marked in [('review_flagged',True),('not_flagged',False)]:
            cohort=[r for r in rows if r['modality']=='image' and bool(r['reference_difficulty_flags'])==marked]
            if cohort:image_difficulty[group]=aggregate(cohort)
    return {'target_count':len(targets),'query_count':len(rows),'scene_count':len(scene_ids),
        'row_scope':'universal_complete' if complete else 'modality_comparator',
        'modalities':by_modality,'uq_rank':scene_clustered_bootstrap(means) if complete else None,
        'paired_query_robustness':paired,
        'text_language_groups':language,'language_groups_overlap':True,
        'target_scope_groups':scope_groups,'image_reference_difficulty_groups':image_difficulty,
        'diagnostic_groups_change_primary_cohort':False,
        'fixed_iou_status':'uncalibrated_diagnostic','oracle_iou_status':'diagnostic_only',
        'per_query':rows}


def score_selections(targets, selections, xyz, ids, modalities):
    """A submitted boolean set is evaluated directly, without fitting a cutoff."""
    expected={t['queries'][m] for t in targets for m in modalities}
    if set(selections)!=expected:
        raise ValueError('Explicit selections must cover every declared query exactly')
    rows=[]
    for target in targets:
        scene=target['scene_id']
        for modality in modalities:
            qid=target['queries'][modality];mask=selections[qid]
            if mask.dtype!=np.bool_:
                raise ValueError('Explicit element selection must be a boolean vector, not scores requiring a fitted threshold')
            v=evaluate_query_probabilities(mask.astype(np.float32),target_instance_id=target['instance_id'],
                same_class_distractor_instance_ids=target['same_class_distractor_instance_ids'],
                mesh_instance_ids=ids[scene],mesh_xyz=xyz[scene])
            instance_ids,counts=np.unique(ids[scene][mask],return_counts=True)
            dominant=instance_ids[counts==counts.max()].tolist() if len(counts) else []
            correct_dominant=dominant==[target['instance_id']]
            rows.append(dict(query_id=qid,scene_id=scene,instance_id=target['instance_id'],modality=modality,
                iou=v['fixed_iou_0.5'],purity=v['selected_purity'],coverage=v['positive_coverage'],
                same_class_distractor_iou=v['same_class_distractor_iou'],selected_elements=int(mask.sum()),
                dominant_selected_instance_ids=dominant,target_is_unique_dominant_instance=correct_dominant))
    keys=['iou','purity','coverage']
    by_modality={}
    for m in modalities:
        group=[r for r in rows if r['modality']==m];scenes=sorted({r['scene_id'] for r in group})
        scene_values={s:{k:float(np.mean([r[k] for r in group if r['scene_id']==s])) for k in keys} for s in scenes}
        by_modality[m]=dict(query_count=len(group),scene_count=len(scenes),per_scene=scene_values,
            empty_selections=sum(r['selected_elements']==0 for r in group),
            scene_macro={k:float(np.mean([v[k] for v in scene_values.values()])) for k in keys},
            iou_ci=scene_clustered_bootstrap({s:v['iou'] for s,v in scene_values.items()}))
    paired=None
    if set(modalities)==set(MODALITIES):
        index={r['query_id']:r for r in rows};target_rows=[]
        agreement_rows=[]
        for target in targets:
            minimum=min(index[target['queries'][m]]['iou'] for m in MODALITIES)
            target_rows.append(dict(scene_id=target['scene_id'],instance_id=target['instance_id'],
                minimum_iou=minimum,all_four_iou_at_least_025=float(minimum>=.25),
                all_four_iou_at_least_050=float(minimum>=.5)))
            masks={m:selections[target['queries'][m]] for m in MODALITIES}
            truth=ids[target['scene_id']]==target['instance_id']
            common=np.logical_and.reduce(list(masks.values()))
            all_selected=np.logical_or.reduce(list(masks.values()))
            values={f'{a}__{b}_iou':float(np.count_nonzero(masks[a]&masks[b])/max(np.count_nonzero(masks[a]|masks[b]),1))
                for a,b in combinations(MODALITIES,2)}
            values.update(common_true_coverage=float(np.count_nonzero(common&truth)/max(np.count_nonzero(truth),1)),
                common_false_elements=int(np.count_nonzero(common&~truth)),
                union_false_fraction=float(np.count_nonzero(all_selected&~truth)/max(np.count_nonzero(all_selected),1)))
            agreement_rows.append(dict(scene_id=target['scene_id'],instance_id=target['instance_id'],**values))
        scenes=sorted({r['scene_id'] for r in target_rows})
        paired={key:scene_clustered_bootstrap({s:float(np.mean([r[key] for r in target_rows if r['scene_id']==s]))
                    for s in scenes}) for key in ['minimum_iou','all_four_iou_at_least_025','all_four_iou_at_least_050']}
        paired['per_target']=target_rows
        agreement_keys=[k for k in agreement_rows[0] if k not in ['scene_id','instance_id','common_false_elements']]
        paired['selection_agreement']={key:scene_clustered_bootstrap({s:float(np.mean([r[key] for r in agreement_rows if r['scene_id']==s]))
            for s in scenes}) for key in agreement_keys}
        paired['selection_agreement'].update(per_target=agreement_rows,
            interpretation='secondary consistency diagnosis; agreement can be consistently wrong; empty/empty agreement is zero; accuracy metrics unchanged')
    identity={}
    for m in modalities:
        group=[r for r in rows if r['modality']==m];scenes=sorted({r['scene_id'] for r in group})
        per_scene={s:dict(query_count=sum(r['scene_id']==s for r in group),
            dominant_target_count=sum(r['scene_id']==s and r['target_is_unique_dominant_instance'] for r in group),
            dominant_target_rate=float(np.mean([r['target_is_unique_dominant_instance'] for r in group if r['scene_id']==s]))) for s in scenes}
        correct=[r for r in group if r['target_is_unique_dominant_instance']]
        conditional={s:{k:float(np.mean([r[k] for r in correct if r['scene_id']==s])) for k in keys}
                     for s in scenes if any(r['scene_id']==s for r in correct)}
        identity[m]=dict(query_count=len(group),dominant_target_count=len(correct),
            scene_macro_dominant_target_rate=float(np.mean([v['dominant_target_rate'] for v in per_scene.values()])),
            per_scene=per_scene,conditional_scene_count=len(conditional),
            conditional_scene_macro={k:float(np.mean([v[k] for v in conditional.values()])) if conditional else None for k in keys},
            interpretation='secondary actual-set identity/extent diagnosis; empty sets and dominant ties fail; background/void included; conditional results never replace full-cohort accuracy')
    return dict(query_count=len(rows),modalities=by_modality,per_query=rows,paired_query_robustness=paired,
        identity_diagnosis=identity,
        evaluator_threshold_fitted=False,ranking_scores_unchanged=True,
        contract='explicit method-selected elements; secondary output distinct from ranking AP and fixed-score diagnostics')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selections',type=Path,help='Optional complete boolean element sets, separate from ranking scores')
    p.add_argument('--modalities',nargs='+',choices=MODALITIES,default=MODALITIES)
    a=p.parse_args()
    release=json.loads((a.package/'release.json').read_text())
    targets=json.loads((a.package/'evaluator/targets.json').read_text())
    review_path=a.package/'evaluator/reference_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        flags={(r['scene_id'],r['instance_id']):r['difficulty_flags'] for r in review['targets']}
        for target in targets['targets']:
            target['reference_difficulty_flags']=flags[(target['scene_id'],target['instance_id'])]
    xyz={d['scene_id']:np.load(a.package/'method'/d['mesh_xyz_path']) for d in targets['scene_domains']}
    ids={d['scene_id']:np.load(a.package/'evaluator'/d['mesh_instance_ids_path']) for d in targets['scene_domains']}
    predictions={p.stem:np.load(p) for p in a.predictions.glob('*.npy')}
    if any(v.dtype!=np.float32 for v in predictions.values()):
        raise ValueError('Submit float32 mesh-domain scores in [0,1]')
    report=score_predictions(targets['targets'],predictions,xyz,ids,modalities=a.modalities)
    if a.selections:
        selections={p.stem:np.load(p) for p in a.selections.glob('*.npy')}
        report['explicit_selection']=score_selections(targets['targets'],selections,xyz,ids,a.modalities)
    write_json(a.output,{'benchmark_version':release['benchmark_version'],'role':release['role'],**report})
    print(json.dumps({'query_count':report['query_count'],'uq_rank':report['uq_rank']},indent=2))


if __name__=='__main__':
    main()
