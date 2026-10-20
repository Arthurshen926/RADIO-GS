"""Post-scoring UQIS diagnosis. Evaluator labels never enter refinement.

Stream existing predictions, retaining only small aggregate/query reports. This
is an explanatory development analysis, not a new inference arm or oracle gate.
"""
import json
from pathlib import Path
import numpy as np

from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.querying.response_study_adapters import SYSTEM, read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def aggregate(rows):
    if not rows:
        return dict(queries=0)
    total=sum(r['positive_delta_sum'] for r in rows)
    return dict(queries=len(rows),
        ap_improved=sum(r['ap_delta_pp']>1e-8 for r in rows),
        ap_degraded=sum(r['ap_delta_pp'] < -1e-8 for r in rows),
        query_mean_ap_delta_pp=float(np.mean([r['ap_delta_pp'] for r in rows])),
        updated_query_vertices=sum(r['updated_vertices'] for r in rows),
        target_updated_query_vertices=sum(r['target_updated_vertices'] for r in rows),
        positive_delta_sum=total,
        non_target_fraction_of_positive_delta=(sum(r['non_target_delta_sum'] for r in rows)/total if total else None),
        same_class_distractor_fraction_of_positive_delta=(sum(r['distractor_delta_sum'] for r in rows)/total if total else None))


def main():
    root=STUDY/'post_response_v2'
    if not read(root/'summary.json')['complete']:
        raise ValueError('diagnosis only after all-five scoring')
    jobs=[j for j in read(root/'manifest.json')['jobs'] if j['benchmark']=='uqis']
    package=Path(jobs[0]['package']);targets=read(package/'evaluator/targets.json')
    truth_by_query={qid:t for t in targets['targets'] for qid in t['queries'].values()}
    domains={d['scene_id']:d for d in targets['scene_domains']}
    original={r['query_id']:r for r in read(root/'uqis_baseline.json')['per_query']}
    changed={r['query_id']:r for r in read(root/'uqis_observable.json')['per_query']}
    rows=[]
    for job in jobs:
        scene=job['name'];domain=domains[scene]
        id_path=package/'evaluator'/domain['mesh_instance_ids_path']
        if sha(id_path)!=domain['mesh_instance_ids_sha256']:
            raise ValueError('evaluator instance labels changed')
        ids=np.load(id_path)
        backend=SYSTEM/'uqis'/scene/'query/registered'
        details={r['query_id']:r for r in read(backend/'runs'/f'{scene}.json')['query_details']}
        prediction_root=STUDY/('post_response_v2_uqis_'+scene)
        report=read(prediction_root/'prediction.json')
        for index,entry in enumerate(report['queries']):
            qid=entry['query_id'];detail=details[qid];target=truth_by_query[qid]
            if target['scene_id']!=scene:raise ValueError('query scene differs')
            old_path=backend/'identity_scope/predictions'/f'{qid}.npy'
            new_path=prediction_root/f'observable_{index:03d}.npy'
            if sha(old_path)!=entry['source_response_sha256'] or sha(new_path)!=report['prediction_files'][new_path.name]:
                raise ValueError('sealed responses changed')
            delta=np.load(new_path).astype(np.float64)-np.load(old_path)
            if bool((delta < 0).any()):
                raise ValueError('positive-only UQIS update lowered a response')
            positive=ids==target['instance_id']
            distractor=np.isin(ids,target['same_class_distractor_instance_ids'])
            modality=entry['modality'];anchor=None;supported=None
            if modality.startswith('point'):
                supported=bool(detail['anchor_has_feature'])
                anchor=int(detail['anchor_vertex']) if supported else None
            elif modality=='image':
                anchor=int(detail['inferred_anchor_membership']['anchor_vertex'])
            rows.append(dict(query_id=qid,scene_id=scene,modality=modality,
                anchor_supported=supported,anchor_correct=(bool(ids[anchor]==target['instance_id']) if anchor is not None else None),
                ap_delta_pp=100*(changed[qid]['average_precision']-original[qid]['average_precision']),
                updated_vertices=int((delta>0).sum()),target_updated_vertices=int(((delta>0)&positive).sum()),
                positive_delta_sum=float(delta.sum()),target_delta_sum=float(delta[positive].sum()),
                non_target_delta_sum=float(delta[~positive].sum()),distractor_delta_sum=float(delta[distractor].sum())))
    if len(rows)!=268 or {r['query_id'] for r in rows}!=set(truth_by_query):
        raise ValueError('incomplete diagnosis cohort')
    summary={}
    for modality in ['text','image','point_2d','point_3d']:
        group=[r for r in rows if r['modality']==modality]
        summary[modality]=dict(all=aggregate(group),
            correct_anchor=aggregate([r for r in group if r['anchor_correct'] is True]),
            incorrect_anchor=aggregate([r for r in group if r['anchor_correct'] is False]),
            no_anchor_diagnosis=aggregate([r for r in group if r['anchor_correct'] is None]))
    run=BoundedExperiment(STUDY,'response_update_diagnosis',POLICY)
    result=dict(schema='tmm.response_update.diagnosis.v1',summary=summary,per_query=rows,
        evaluator_labels_used=True,post_scoring_only=True,not_an_inference_gate=True,
        query_mean_is_secondary_not_primary_scene_macro=True,
        delta_mass_is_score_sum_not_probability_or_error_rate=True,
        unsupported_points_unchanged=all(r['updated_vertices']==0 for r in rows if r['anchor_supported'] is False),
        source_sha256={str(p):sha(p) for p in [root/'summary.json',root/'uqis_baseline.json',root/'uqis_observable.json',
            package/'evaluator/targets.json',Path(__file__)]})
    run.json('summary.json',result)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
