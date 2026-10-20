"""Attribute frozen transfer results to pool capacity versus decision errors.

Runs only after all decisions and fresh metrics are sealed. Cached UQIS pool
utilities are reused only after candidate-byte/order and target-hash checks.
"""
import argparse
from pathlib import Path
import numpy as np
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.querying.response_study_adapters import read
from radio_gs.response_study.transfer_diagnosis import menu_capacity,instance_capacity
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transfer-run',default='response_value_multitask_transfer_v1')
    p.add_argument('--run',default='response_value_multitask_diagnosis_v1');args=p.parse_args()
    root=STUDY/args.transfer_run; summary=read(root/'summary.json'); manifest=read(root/'manifest.json')
    if not summary['complete']:raise ValueError('complete scored transfer required')
    for name,digest in read(root/'predictions_complete.json')['report_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('transfer decision report changed')
    old=STUDY/'response_replacement_capacity_v1'
    historical=read(old/'summary.json'); old_seal=read(old/'predictions_complete.json')
    if any(summary['uqis_evaluator_hashes'].get(p)!=h for p,h in historical['evaluator_hashes'].items()):
        raise ValueError('cached utilities and current target inputs differ')
    history={r['query_id']:r for r in historical['rows']}
    actual={r['query_id']:r['iou'] for r in read(root/'uqis_full_sets.json')['per_query']}
    category=[]; instances=[]; paths=[root/'summary.json',root/'manifest.json',root/'predictions_complete.json',
        root/'uqis_full_sets.json',old/'summary.json',old/'predictions_complete.json']
    for row in summary['category_rows']:
        path=root/('scannet_'+row['job']['name']+'.json'); report=read(path);paths.append(path)
        keys=('miou_19','miou_15','miou_10')
        values=[[row['metrics'][a][k] for k in keys] for a in ('baseline','fixed_dual_native','fixed_source_readout')]
        winner=report['arms']['full']['candidate']
        if any(abs(v-row['metrics']['full'][k])>1e-12 for k,v in zip(keys,values[winner])):
            raise ValueError('learned category result differs from its chosen intact candidate')
        category.append(dict(scene=row['job']['name'],candidate=winner,
            capacity=menu_capacity(values,winner),protected_capacity=menu_capacity(values,winner,protect_all=True),
            same_choice_as_blind=winner==report['arms']['query_blind']['candidate'],
            same_choice_as_permuted=winner==report['arms']['permuted']['candidate']))
    for job in manifest['jobs']:
        if job['benchmark']!='uqis':continue
        path=root/('uqis_'+job['name']+'.json');paths.append(path)
        # A07 sealed target/label files; its exact-pool predecessor also sealed
        # map coordinates. The new scorer additionally hashes package copies
        # of these coordinates. Bind those copies by bytes, not path equality.
        old_path=STUDY/'response_replacement_transfer_v1'/(job['name']+'.json');paths.append(old_path)
        original=read(old_path)
        geometry=[h for p,h in summary['uqis_evaluator_hashes'].items()
                  if p.endswith('/'+job['name']+'/mesh_xyz.npy')]
        old_geometry=[h for p,h in original['input_hashes'].items() if p.endswith('/mesh_xyz.npy')]
        if len(geometry)!=1 or geometry!=old_geometry:
            raise ValueError('cached utilities used different mesh coordinate bytes')
        for row in read(path)['rows']:
            qid=row['query_id']
            instances.append(instance_capacity(row,history[qid],old_seal['candidate_sha256'][qid],actual[qid]))
    if len(category)!=10 or len(instances)!=134:raise ValueError('joint diagnosis cohort incomplete')
    category_macro={key:{name:float(np.mean([r[key][name] for r in category]))
        for name in ('available_mean_gain','decision_mean_regret','selected_mean_gain')}
        for key in ('capacity','protected_capacity')}
    instance_macro={}
    for m in ('text','image'):
        selected=[r for r in instances if r['modality']==m]; rooms={r['scene'] for r in selected}
        instance_macro[m]={k:float(np.mean([np.mean([r[k] for r in selected if r['scene']==s]) for s in rooms]))
            for k in ('baseline_iou','selected_iou','oracle_iou','fixed_provider_iou','available_gain','decision_regret')}
        instance_macro[m]['same_choice_as_blind']=sum(r['same_choice_as_blind'] for r in selected)
        instance_macro[m]['same_choice_as_permuted']=sum(r['same_choice_as_permuted'] for r in selected)
    source=[__file__,'radio_gs/response_study/transfer_diagnosis.py']
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(1024**2)
    run.json('manifest.json',dict(transfer_run=args.transfer_run,input_hashes={str(p):sha(p) for p in paths},
        source_hashes={p:sha(p) for p in source},diagnostic_only=True,benchmark_parameters_changed=False))
    run.json('source_snapshot.json',{p:Path(p).read_text() for p in source})
    run.json('summary.json',dict(complete=True,diagnostic_only=True,category_rows=category,instance_rows=instances,
        category_macro=category_macro,instance_macro=instance_macro,
        category_same_choice_as_blind=sum(r['same_choice_as_blind'] for r in category),
        category_same_choice_as_permuted=sum(r['same_choice_as_permuted'] for r in category),
        all_oracles_are_whole_answers=True,category_outputs_share_one_oracle_choice=True,
        uqis_cached_utilities_verified_against_exact_candidates_and_fresh_selected_scores=True,
        independent_final_test=False,default_changed=False,durable_dense_prediction_bytes=0))
    print('category attribution',category_macro,flush=True);print('instance attribution',instance_macro,flush=True)


if __name__=='__main__':main()
