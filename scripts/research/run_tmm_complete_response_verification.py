"""Q02: verify whole frozen responses from three existing identity proposals.

All 67 image predictions seal before scoring all 268 UQIS queries. Other tasks
are explicitly unchanged historical references. No new teachers, training,
source-image access, dense disk outputs, or benchmark-specific winner selection.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from radio_gs.querying.complete_response_verification import CompleteResponseVerifier, recover_second_teacher_ranks
from radio_gs.querying.registered_observation_query import public_queries
from radio_gs.querying.response_study_adapters import SYSTEM, read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_support_agreement import PredictionMemory
from scripts.research.tmm_ram_predictions import public_query_files

ARMS = ('verified', 'permuted')
PROPOSALS = ('joint', 'siglip', 'dino')


def predict(job, run, memory):
    started = time.monotonic()
    folder = Path(job['scene']).parent; spec = read(job['scene'])
    source = SYSTEM/'uqis'/job['name']/'query/registered'
    receipt = source/'runs'/f"{job['name']}.json"
    details = read(receipt)['query_details']
    queries, _ = public_queries(Path(job['queries']), job['name'])
    if {d['query_id']:d['modality'] for d in details} != {q['query_id']:q['modality'] for q in queries}:
        raise ValueError('public query cohort changed')
    if spec.get('query_conditioned_membership') or spec['inferred_anchor_membership'] != ['image']:
        raise ValueError('effective frozen reader changed')
    identity_path = source/'identity_evidence.npz'
    identity = np.load(identity_path, allow_pickle=False)
    ids = identity['query_ids'].tolist()
    if ids != [d['query_id'] for d in details]:
        raise ValueError('cached legal query identity order changed')
    membership = sparse.load_npz(folder/'membership.npz').tocsr()
    visibility = sparse.load_npz(folder/'visibility.npz').tocsr()
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder/'region_offsets.npy')))
    xyz = np.load(folder/'mesh_xyz.npy')
    descriptors = np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    checkpoint = spec['direct_anchor_membership']['checkpoint']
    reader = DirectAnchorMembership(membership, visibility, frames, xyz, descriptors,
        sparse.load_npz(folder/'transfer.npz'), checkpoint)
    verifier = CompleteResponseVerifier(membership, visibility, frames)
    paths = [Path(job['scene']), receipt, identity_path, Path(checkpoint), *public_query_files(job['queries'])]
    paths += [folder/n for n in ['membership.npz','visibility.npz','region_offsets.npy',
                                'mesh_xyz.npy','descriptors.npy','transfer.npz']]
    columns = {a:[] for a in ARMS+PROPOSALS}; rows = []; image_ids = []
    for i, detail in enumerate(details):
        qid = detail['query_id']; base = source/'identity_scope'
        p, s = base/'predictions'/f'{qid}.npy', base/'selections'/f'{qid}.npy'
        paths.extend([p,s])
        if detail['modality'] != 'image': continue
        available=verifier.available
        seed = int(available[np.argmax(identity['identity_response'][available,i])])
        old_detail = detail['inferred_anchor_membership']
        if seed != old_detail['seed_region']:
            raise ValueError('retained full-teacher identity seed does not replay')
        old_score, old_selection, old = reader.infer_region(seed)
        error = float(np.abs(old_score-np.load(p)).max())
        if (old['anchor_vertex'] != old_detail['anchor_vertex'] or error > 1e-7
                or not np.array_equal(old_selection,np.load(s))):
            raise ValueError('frozen complete decoder does not replay')
        native=identity['native_response'][:,i]
        second,recovery=recover_second_teacher_ranks(native,identity['identity_response'][:,i],available)
        seeds=[seed,int(available[np.argmax(native[available])]),int(available[np.argmax(second[available])])]
        decoded={seed:(old_score,old_selection,old)}
        for proposed in seeds:
            if proposed not in decoded:decoded[proposed]=reader.infer_region(proposed)
        candidate_scores=[decoded[s][0] for s in seeds]
        candidate_sets=[decoded[s][1] for s in seeds]
        row = dict(query_id=qid, seed_region=seed, original_anchor=old['anchor_vertex'],
            original_replay_max_error=error, arms={},rank_recovery=recovery,
            proposals={a:dict(seed_region=seeds[k],anchor_vertex=decoded[seeds[k]][2]['anchor_vertex'])
                       for k,a in enumerate(PROPOSALS)})
        for k,label in enumerate(PROPOSALS):columns[label].append(candidate_scores[k])
        for arm in ARMS:
            winner,metadata=verifier.verify(candidate_sets,seeds,native,second,permuted=arm=='permuted')
            value,selected=candidate_scores[winner],candidate_sets[winner]
            anchor=decoded[seeds[winner]][2]['anchor_vertex']
            columns[arm].append(value)
            row['arms'][arm] = dict(metadata,anchor_vertex=anchor,chosen_proposal=PROPOSALS[winner],
                changed_anchor=anchor!=old['anchor_vertex'],
                changed_selected_vertices=int((selected!=old_selection).sum()))
        if row['arms'][ARMS[0]]['common_views']!=row['arms'][ARMS[1]]['common_views']:
            raise AssertionError('verification evidence domains differ')
        rows.append(row);image_ids.append(qid)
        print('predicted',job['name'],qid,flush=True)
    return dict(job=job,columns=image_ids,queries=rows,ground_truth_read=False,
        image_queries=len(rows),all_other_queries_unchanged=True,
        source_rgb_read=False,source_depth_read=False,source_masks_read=False,
        query_encodings='retained legal native responses; hashed intermediates, not freshly re-encoded',
        full_teacher_baseline_candidate_preserved=True,threshold=float(reader.saved['threshold']),
        input_hashes={str(p):sha(p) for p in sorted(set(paths))},
        prediction_arrays={a:memory.add(run,job['name']+'_'+a,
            torch.from_numpy(np.stack(columns[a],1))) for a in ARMS+PROPOSALS},seconds=time.monotonic()-started)


def score(jobs, reports, run, memory):
    from radio_gs.benchmarks.uqis import MODALITIES
    from radio_gs.benchmarks.uqis.evaluate import score_predictions, score_selections
    from scripts.research.summarize_tmm_response_study import paired
    package = Path(jobs[0]['package']); target_path = package/'evaluator/targets.json'
    targets = read(target_path)
    xyz = {d['scene_id']:np.load(package/'method'/d['mesh_xyz_path']) for d in targets['scene_domains']}
    ids = {d['scene_id']:np.load(package/'evaluator'/d['mesh_instance_ids_path']) for d in targets['scene_domains']}
    evaluator_inputs = [target_path]
    evaluator_inputs += [package/'method'/d['mesh_xyz_path'] for d in targets['scene_domains']]
    evaluator_inputs += [package/'evaluator'/d['mesh_instance_ids_path'] for d in targets['scene_domains']]
    evaluator_hashes = {str(p):sha(p) for p in evaluator_inputs}
    reference, baseline_sets = {}, {}
    for job in jobs:
        source = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        public,_ = public_queries(Path(job['queries']),job['name'])
        for q in public:
            qid = q['query_id']
            reference[qid] = np.load(source/'predictions'/f'{qid}.npy')
            baseline_sets[qid] = np.load(source/'selections'/f'{qid}.npy')
    scores, selections = {}, {}
    for arm in ('baseline',)+ARMS:
        prediction, selected = dict(reference),dict(baseline_sets)
        if arm != 'baseline':
            for job,report in zip(jobs,reports):
                array = memory.read(run,job['name']+'_'+arm,len(report['columns'])).numpy()
                for i,qid in enumerate(report['columns']):
                    prediction[qid] = array[:,i]; selected[qid] = array[:,i]>report['threshold']
        scores[arm] = score_predictions(targets['targets'],prediction,xyz,ids)
        selections[arm] = score_selections(targets['targets'],selected,xyz,ids,MODALITIES)
        run.json('uqis_'+arm+'.json',scores[arm])
        run.json('uqis_'+arm+'_selections.json',selections[arm])
        print('scored all 268 queries:',arm,flush=True)
    reference_summary = read(SYSTEM/'summary.json')
    macro = {}
    for arm in scores:
        macro[arm] = {m+'_ap':scores[arm]['modalities'][m]['scene_macro']['average_precision'] for m in MODALITIES}
        macro[arm].update({m+'_selected_iou':selections[arm]['modalities'][m]['scene_macro']['iou'] for m in MODALITIES})
    if any(abs(v-reference_summary['macro']['uqis'][k])>1e-8 for k,v in macro['baseline'].items()):
        raise AssertionError('retained baseline metrics do not replay')
    effects = {}
    for arm,base in [('verified','baseline'),('permuted','baseline'),('verified','permuted')]:
        comparison = arm+'_minus_'+base;effects[comparison] = {}
        for m in MODALITIES:
            for name,records,key in [('ap',scores,'average_precision'),('selected_iou',selections,'iou')]:
                left=records[arm]['modalities'][m]['per_scene'];right=records[base]['modalities'][m]['per_scene']
                effects[comparison][m+'_'+name]=paired([left[s][key]-right[s][key] for s in right])
    # Post-seal attribution only: no oracle candidate or thresholds enter inference.
    diagnostics=[]
    query_metrics={a:{r['query_id']:r for r in selections[a]['per_query']} for a in selections}
    target_by_query={t['queries']['image']:t for t in targets['targets']}
    for job,report in zip(jobs,reports):
        for row in report['queries']:
            qid=row['query_id'];truth=ids[job['name']]==target_by_query[qid]['instance_id']
            diagnostics.append(dict(query_id=qid,scene=job['name'],
                baseline_anchor_correct=bool(truth[row['original_anchor']]),
                baseline_iou=query_metrics['baseline'][qid]['iou'],
                arms={a:dict(anchor_correct=bool(truth[row['arms'][a]['anchor_vertex']]),
                    iou=query_metrics[a][qid]['iou']) for a in ARMS}))
    run.json('anchor_diagnosis.json',dict(rows=diagnostics,uses_gt_after_prediction_seal=True,
        conditional_statistics_are_not_causal_effects=True))
    # This finite-pool ceiling is labelled explicitly; it never selects inference.
    capacity=[]
    for job,report in zip(jobs,reports):
        arrays={a:memory.read(run,job['name']+'_'+a,len(report['columns'])).numpy() for a in PROPOSALS}
        for i,qid in enumerate(report['columns']):
            truth=ids[job['name']]==target_by_query[qid]['instance_id']
            values={}
            for a in PROPOSALS:
                selected=arrays[a][:,i]>report['threshold']
                intersection=int(np.count_nonzero(selected&truth))
                values[a]=intersection/max(int(np.count_nonzero(selected|truth)),1)
            capacity.append(dict(query_id=qid,scene=job['name'],proposal_iou=values,
                oracle_iou=max(values.values()),
                verified_iou=query_metrics['verified'][qid]['iou']))
    ceiling=float(np.mean([np.mean([r['oracle_iou'] for r in capacity if r['scene']==j['name']]) for j in jobs]))
    run.json('proposal_capacity.json',dict(rows=capacity,finite_pool_oracle_iou=ceiling,
        gt_selected_after_all_predictions=True,not_an_automatic_method=True,
        not_an_upper_bound_on_other_proposals=True))
    if any(sha(p)!=h for p,h in evaluator_hashes.items()):
        raise ValueError('evaluator asset changed during scoring')
    return dict(complete=True,kind='targeted mechanism study; not a full shared-method trial',
        uqis_macro=macro,effects=effects,image_queries=67,scored_uqis_queries=268,
        anchor_correct_counts=dict(baseline=sum(r['baseline_anchor_correct'] for r in diagnostics),
            **{a:sum(r['arms'][a]['anchor_correct'] for r in diagnostics) for a in ARMS}),
        changed_anchor_counts={a:sum(r['arms'][a]['changed_anchor'] for p in reports for r in p['queries']) for a in ARMS},
        finite_pool_oracle_iou=ceiling,oracle_is_not_a_method=True,
        no_common_evidence_queries=sum(r['arms']['verified']['no_common_evidence'] for p in reports for r in p['queries']),
        source_benchmark_compatibility={f:dict(metrics=v,changed=False,fresh_inference=False,
            provenance=str(SYSTEM/'summary.json'),sha256=sha(SYSTEM/'summary.json'))
            for f,v in reference_summary['macro'].items() if f!='uqis'},
        all_five_feedback_present=True,all31_fresh_inference=False,actual_point_and_text_outputs_exact=True,
        independent_test=False,default_changed=False,no_new_supervision=True,
        evaluator_input_hashes=evaluator_hashes,transient_prediction_bytes=memory.bytes,
        durable_dense_prediction_bytes=0,study_logical_bytes=tree_bytes(STUDY))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='complete_response_verification_v1')
    args=parser.parse_args();torch.set_num_threads(4)
    run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(8*2**20)
    recipe=read('configs/pipeline/system_v10.json')
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    sources=['scripts/research/run_tmm_complete_response_verification.py','radio_gs/querying/complete_response_verification.py',
        'radio_gs/querying/observation_readout.py','radio_gs/querying/direct_anchor_membership.py',
        'radio_gs/querying/anchor_observation_evidence.py','scripts/research/run_tmm_support_agreement.py',
        'scripts/research/tmm_ram_predictions.py','radio_gs/querying/registered_observation_query.py',
        'radio_gs/benchmarks/uqis/evaluate.py','radio_gs/benchmarks/uqis/metrics.py',
        'scripts/research/summarize_tmm_response_study.py','configs/pipeline/system_v10.json']
    run.json('manifest.json',dict(jobs=recipe['jobs'],intervention_jobs=jobs,arms=ARMS,
        changed_variable='query identity verification chooses one unmodified complete response',
        proposals=PROPOSALS,proposal_rules='joint native identity winner plus each of two existing teacher winners',
        verification='mean common-view difference of IoU-weighted and complement-weighted query percentile',
        control='fixed within-view query-score permutation; identical candidates, geometry and common views',
        baseline_complete_response_always_a_candidate=True,all_query_teachers_retained=True,
        second_teacher_rank_recovery='invert equal-rank fusion and validate the half-integer rank lattice; no cosine recovery',
        union_of_all_proposal_views_excluded=True,complete_decoder_and_threshold_frozen=True,
        single_fixed_control_seed=20261016,parameter_sweep=False,new_models=False,
        other_benchmarks='unchanged historical reference, not fresh inference',
        input_access='map-only with retained legal query identity intermediates',
        all_predictions_before_target_scoring=True,independent_test=False,default_changed=False,
        source_hashes={p:sha(p) for p in sources}))
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in sources})
    memory=PredictionMemory();reports=[]
    for job in jobs:
        report=predict(job,run,memory);run.json(job['name']+'_prediction.json',report);reports.append(report);gc.collect()
    if len(reports)!=9 or sum(p['image_queries'] for p in reports)!=67:
        raise ValueError('complete intervention cohort changed')
    run.json('predictions_complete.json',dict(image_queries=67,scored_queries=268,ground_truth_read=False,
        report_hashes={str(run.root/(j['name']+'_prediction.json')):sha(run.root/(j['name']+'_prediction.json')) for j in jobs},
        transient_prediction_bytes=memory.bytes))
    for report in reports:
        for path,digest in report['input_hashes'].items():
            if sha(path)!=digest:raise ValueError('input changed after prediction seal')
        for arm in ARMS+PROPOSALS:memory.read(run,report['job']['name']+'_'+arm,len(report['columns']))
    run.json('evaluation_manifest.json',dict(prediction_seal_sha256=sha(run.root/'predictions_complete.json'),
        all_inputs_and_predictions_rechecked=True))
    summary=score(jobs,reports,run,memory);run.json('summary.json',summary)
    memory.values.clear();gc.collect()
    run.json('ram_released.json',dict(complete=True,no_dense_predictions_on_persistent_disk=True))
    print(json.dumps({k:summary[k] for k in ['uqis_macro','anchor_correct_counts','changed_anchor_counts']},indent=2))


if __name__=='__main__':main()
