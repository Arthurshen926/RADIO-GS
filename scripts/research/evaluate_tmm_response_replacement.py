"""Frozen learned replacement policy transferred to text and image UQIS inputs.

One checkpoint, one full-policy margin, both existing complete-response
providers, all existing applicable query teachers. Actual point outputs remain
exact. Source-assisted families are explicitly pending transfer, not successes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import sparse
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.complete_response_verification import recover_second_teacher_ranks
from radio_gs.querying.response_replacement import (
    FEATURE_NAMES, RegionResponseEvidence, propose_regions, relative_utilities, choose_response,
)
from radio_gs.querying.registered_observation_query import public_queries
from radio_gs.querying.response_study_adapters import SYSTEM, read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.run_tmm_support_agreement import PredictionMemory
from scripts.research.tmm_ram_predictions import public_query_files

ARMS = ('full', 'query_blind', 'permuted', 'unrestricted')


def predict(job, run, memory, bundle, *, arms=ARMS):
    start = time.monotonic(); folder = Path(job['scene']).parent; spec = read(job['scene'])
    utility = relative_utilities
    if bundle.get('decision_rule', 'ordered_pair') == 'set_value':
        from radio_gs.querying.response_value import relative_utilities as utility
    elif bundle.get('decision_rule', 'ordered_pair') != 'ordered_pair':
        raise ValueError('unknown complete-response utility rule')
    root = SYSTEM/'uqis'/job['name']/'query/registered'
    receipt = root/'runs'/f"{job['name']}.json"; details = read(receipt)['query_details']
    public, _ = public_queries(Path(job['queries']), job['name'])
    if [(q['query_id'], q['modality']) for q in public] != [(d['query_id'], d['modality']) for d in details]:
        raise ValueError('public and cached query cohorts differ')
    if spec.get('positive_membership') or spec.get('surface_consistency') or spec.get('query_conditioned_membership'):
        raise ValueError('unexpected provider variant requires an explicit adapter')
    paths = [folder/n for n in ('membership.npz', 'visibility.npz', 'region_offsets.npy',
                              'mesh_xyz.npy', 'descriptors.npy', 'transfer.npz')]
    paths += [Path(job['scene']), receipt, root/'identity_evidence.npz', *public_query_files(job['queries'])]
    membership = sparse.load_npz(folder/'membership.npz'); visibility = sparse.load_npz(folder/'visibility.npz')
    transfer = sparse.load_npz(folder/'transfer.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder/'region_offsets.npy')))
    checkpoint = Path(spec['direct_anchor_membership']['checkpoint']); paths.append(checkpoint)
    if sha(checkpoint) != bundle['provider_sha256']:
        raise ValueError('training and transfer complete-member provider differ')
    descriptors = np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    reader = DirectAnchorMembership(membership, visibility, frames, np.load(folder/'mesh_xyz.npy'),
        descriptors, transfer, checkpoint)
    reader.saved['estimator'].n_jobs = 4
    scope = InstanceObservationScope(membership, visibility, frames,
        exclusion_policy=spec.get('scope_exclusion_policy', 'visible_complement'))
    evidence = RegionResponseEvidence(membership, visibility, frames)
    identity = np.load(root/'identity_evidence.npz', allow_pickle=False)
    if identity['query_ids'].tolist() != [d['query_id'] for d in details]:
        raise ValueError('cached query evidence order changed')
    columns = {a: [] for a in arms}; sets = {a: [] for a in arms}; rows = []; ids = []
    for i, detail in enumerate(details):
        qid = detail['query_id']; modality = detail['modality']
        p = root/'identity_scope/predictions'/f'{qid}.npy'; s = root/'identity_scope/selections'/f'{qid}.npy'
        paths += [p, s]
        if modality not in ('text', 'image'):
            continue
        native = identity['native_response'][:, i]; joint = identity['identity_response'][:, i]
        old_score, old_set = np.load(p), np.load(s)
        if modality == 'image':
            seed = detail['inferred_anchor_membership']['seed_region']; baseline_provider = 'direct'
            second, _ = recover_second_teacher_ranks(native, joint, evidence.available)
            teacher_scores = np.stack([native, second], 1)
        else:
            seed = detail['instance_identity']['seed_region']; baseline_provider = 'scope'
            teacher_scores = native
        seeds = propose_regions(teacher_scores, evidence.available, frames,
            baseline_seed=seed, budget=bundle['config']['candidate_budget'])
        semantic = np.clip((np.asarray(transfer @ native)+1)*.5, 0, 1).astype(np.float32)
        semantic[transfer.getnnz(axis=1) == 0] = 0
        decoded = {}
        for region in seeds:
            probability, selected, _ = reader.infer_region(region)
            decoded[('direct', region)] = probability, selected
            _, probability, selected, _ = scope.infer(joint, semantic, seed=region)
            decoded[('scope', region)] = probability.astype(np.float32), selected['identity_scope']
        key = (baseline_provider, seed)
        replay_error = float(np.max(np.abs(decoded[key][0]-old_score)))
        if replay_error > 1e-7 or not np.array_equal(decoded[key][1], old_set):
            raise ValueError(f'{modality} incumbent provider replay failed: {replay_error}')
        order = [key] + [k for k in decoded if k != key]
        # Preserve candidate zero byte-for-byte from the reference after replay.
        decoded[key] = old_score, old_set
        probabilities = np.stack([decoded[k][0] for k in order]); selections = np.stack([decoded[k][1] for k in order])
        excluded = sorted(set(frames[seeds].tolist()))
        x, info = evidence.encode(probabilities, selections, teacher_scores, exclude_views=excluded)
        shuffled, _ = evidence.encode(probabilities, selections, teacher_scores, exclude_views=excluded, permuted=True)
        row = dict(query_id=qid, modality=modality, baseline_provider=baseline_provider,
            candidates=[dict(provider=k[0], seed=k[1]) for k in order], evidence=info,
            baseline_replay_max_error=replay_error, arms={},
            candidate_probability_sha256=hashlib.sha256(probabilities.tobytes()).hexdigest(),
            candidate_selection_sha256=hashlib.sha256(selections.tobytes()).hexdigest())
        for arm in arms:
            condition = arm if arm in ('query_blind', 'unregularized') else 'full'
            gains = utility(bundle['models'][condition], shuffled if arm == 'permuted' else x,
                                      query_blind=condition == 'query_blind')
            margin = 0. if arm == 'unrestricted' else bundle['margins'][condition]
            winner = choose_response(gains, margin, evidence_available=info['verification_views'] > 0)
            columns[arm].append(probabilities[winner]); sets[arm].append(selections[winner])
            row['arms'][arm] = dict(candidate=winner, gains=gains.tolist(), margin=margin,
                changed_vertices=int(np.count_nonzero(selections[winner] != old_set)),
                provider=order[winner][0], seed=order[winner][1])
        rows.append(row); ids.append(qid)
    report = dict(scene=job['name'], columns=ids, rows=rows, seconds=time.monotonic()-start,
        ground_truth_read=False, source_images_read=False, all_applicable_teachers_retained=True,
        input_hashes={str(p): sha(p) for p in sorted(set(paths))}, prediction_arrays={})
    for arm in arms:
        for kind, arrays in [('score', columns), ('set', sets)]:
            name = job['name']+'_'+arm+'_'+kind
            report['prediction_arrays'][name] = memory.add(run, name, torch.from_numpy(np.stack(arrays[arm], 1)))
    print('transfer predicted', job['name'], len(rows), 'queries', round(report['seconds'], 1), 's', flush=True)
    return report


def score(jobs, reports, run, memory, *, arms=ARMS):
    from radio_gs.benchmarks.uqis import MODALITIES
    from radio_gs.benchmarks.uqis.evaluate import score_predictions, score_selections
    from scripts.research.summarize_tmm_response_study import paired
    package = Path(jobs[0]['package']); target_path = package/'evaluator/targets.json'; target = read(target_path)
    xyz = {d['scene_id']: np.load(package/'method'/d['mesh_xyz_path']) for d in target['scene_domains']}
    labels = {d['scene_id']: np.load(package/'evaluator'/d['mesh_instance_ids_path']) for d in target['scene_domains']}
    evaluation_paths = [target_path] + [package/'evaluator'/d['mesh_instance_ids_path'] for d in target['scene_domains']]
    evaluation_paths += [package/'method'/d['mesh_xyz_path'] for d in target['scene_domains']]
    evaluator_hashes = {str(p): sha(p) for p in evaluation_paths}
    base, base_sets = {}, {}
    for job in jobs:
        root = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        public, _ = public_queries(Path(job['queries']), job['name'])
        for q in public:
            qid = q['query_id']; base[qid] = np.load(root/'predictions'/f'{qid}.npy')
            base_sets[qid] = np.load(root/'selections'/f'{qid}.npy')
    metrics = {}; selections = {}; macro = {}
    for arm in ('baseline',)+tuple(arms):
        probabilities, selected = dict(base), dict(base_sets)
        if arm != 'baseline':
            for report in reports:
                p = memory.read(run, report['scene']+'_'+arm+'_score', len(report['columns'])).numpy()
                s = memory.read(run, report['scene']+'_'+arm+'_set', len(report['columns'])).numpy().astype(bool)
                for i, qid in enumerate(report['columns']):
                    probabilities[qid] = p[:, i]; selected[qid] = s[:, i]
        metrics[arm] = score_predictions(target['targets'], probabilities, xyz, labels)
        selections[arm] = score_selections(target['targets'], selected, xyz, labels, MODALITIES)
        run.json('uqis_'+arm+'.json', metrics[arm]); run.json('uqis_'+arm+'_sets.json', selections[arm])
        macro[arm] = {m+'_ap': metrics[arm]['modalities'][m]['scene_macro']['average_precision'] for m in MODALITIES}
        macro[arm].update({m+'_selected_iou': selections[arm]['modalities'][m]['scene_macro']['iou'] for m in MODALITIES})
    reference = read(SYSTEM/'summary.json')
    if any(abs(v-reference['macro']['uqis'][k]) > 1e-8 for k, v in macro['baseline'].items()):
        raise ValueError('full UQIS baseline metric replay failed')
    effects = {}
    for arm in arms:
        effects[arm] = {}
        for modality in MODALITIES:
            for name, records, key in [('ap', metrics, 'average_precision'), ('selected_iou', selections, 'iou')]:
                left = records[arm]['modalities'][modality]['per_scene']
                right = records['baseline']['modalities'][modality]['per_scene']
                effects[arm][modality+'_'+name] = paired([left[s][key]-right[s][key] for s in right])
    if any(sha(p) != h for p, h in evaluator_hashes.items()):
        raise ValueError('evaluator inputs changed')
    return dict(complete=True, uqis_macro=macro, effects=effects, evaluator_hashes=evaluator_hashes,
        changed_queries={a: sum(r['arms'][a]['candidate'] != 0 for p in reports for r in p['rows']) for a in arms},
        intervened_modalities=['text', 'image'], freshly_scored_queries=len(base),
        point_outputs_exact=True, one_shared_checkpoint=True, no_benchmark_fitted_margin=True,
        other_benchmarks={family: dict(status='transfer_pending', fresh_inference=False, metrics=value,
                                      reference=str(SYSTEM/'summary.json'))
                          for family, value in reference['macro'].items() if family != 'uqis'},
        all31_fresh_inference=False, independent_final_test=False, default_changed=False,
        durable_dense_prediction_bytes=0, transient_prediction_bytes=memory.bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', default='response_replacement_train_v2')
    parser.add_argument('--run', default='response_replacement_transfer_v1')
    args = parser.parse_args(); torch.set_num_threads(4)
    training = STUDY/args.training_run
    if not read(training/'summary.json')['complete']:
        raise ValueError('independent training/development must finish first')
    bundle = joblib.load(training/'model.joblib'); seal = read(training/'model_seal.json')
    if sha(training/'model.joblib') != seal['model_sha256'] or bundle['feature_names'] != FEATURE_NAMES:
        raise ValueError('sealed model/schema changed')
    manifest = read(training/'manifest.json')
    # Refuse to use a silently modified implementation after model training.
    for path, digest in manifest['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('trained source implementation changed: '+path)
    bundle['provider_sha256'] = manifest['input_hashes'][bundle['config']['provider_checkpoint']]
    run = BoundedExperiment(STUDY, args.run, POLICY); run.preflight(12*2**20)
    jobs = [j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark'] == 'uqis']
    sources = [__file__, 'radio_gs/querying/response_replacement.py',
        'radio_gs/querying/complete_response_verification.py', 'scripts/research/run_tmm_support_agreement.py',
        'radio_gs/querying/registered_observation_query.py', 'radio_gs/benchmarks/uqis/evaluate.py',
        'radio_gs/benchmarks/uqis/metrics.py', 'scripts/research/summarize_tmm_response_study.py',
        'configs/pipeline/system_v10.json']
    if bundle.get('decision_rule') == 'set_value':
        sources.append('radio_gs/querying/response_value.py')
    run.json('manifest.json', dict(training_run=args.training_run, model_sha256=seal['model_sha256'],
        training_summary_sha256=sha(training/'summary.json'), jobs=jobs, arms=ARMS, margins=bundle['margins'],
        source_hashes={p: sha(p) for p in sources}, benchmark_threshold_search=False,
        decision_rule=bundle.get('decision_rule', 'ordered_pair'),
        learned_query_types=bundle.get('learned_query_types', 'independent natural text; image transfer without finetuning'),
        calibration_incumbent=bundle.get('calibration_incumbent', 'direct provider; text scope incumbent is a declared transfer'),
        other_benchmarks='pending source-response signature adapter and independent task supervision'))
    run.json('source_snapshot.json', {p: Path(p).read_text() for p in sources})
    memory = PredictionMemory(); reports = []
    for job in jobs:
        report = predict(job, run, memory, bundle); reports.append(report)
        run.json(job['name']+'.json', report); gc.collect()
    if sum(len(r['rows']) for r in reports) != 134:
        raise ValueError('text/image intervention cohort changed')
    run.json('predictions_complete.json', dict(intervention_queries=134, all_uqis_queries=268,
        report_hashes={r['scene']: sha(run.root/(r['scene']+'.json')) for r in reports}, ground_truth_read=False))
    for report in reports:
        if any(sha(p) != h for p, h in report['input_hashes'].items()):
            raise ValueError('inputs changed after prediction seal')
    summary = score(jobs, reports, run, memory); run.json('summary.json', summary)
    memory.values.clear(); gc.collect()
    run.json('ram_released.json', dict(complete=True, study_bytes=tree_bytes(STUDY)))
    print(json.dumps(summary['uqis_macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
