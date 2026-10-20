"""Post-seal attribution: replay the exact candidate pool, then inspect utility.

One label-free control swaps the complete provider at the ORIGINAL identity.
The finite-pool label oracle is diagnostic only. Neither changes the learner,
its margin, its reported predictions or the retained production default.
"""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.response_study_adapters import SYSTEM, read
from radio_gs.querying.registered_observation_query import public_queries
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

TRANSFER = STUDY/'response_replacement_transfer_v1'


def replay(job):
    report = read(TRANSFER/(job['name']+'.json'))
    if any(sha(p) != h for p, h in report['input_hashes'].items()):
        raise ValueError('candidate replay inputs changed')
    folder = Path(job['scene']).parent; spec = read(job['scene'])
    membership = sparse.load_npz(folder/'membership.npz'); visibility = sparse.load_npz(folder/'visibility.npz')
    transfer = sparse.load_npz(folder/'transfer.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder/'region_offsets.npy')))
    descriptors = np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    reader = DirectAnchorMembership(membership, visibility, frames, np.load(folder/'mesh_xyz.npy'), descriptors,
        transfer, spec['direct_anchor_membership']['checkpoint']); reader.saved['estimator'].n_jobs = 1
    scope = InstanceObservationScope(membership, visibility, frames,
        exclusion_policy=spec.get('scope_exclusion_policy', 'visible_complement'))
    identity = np.load(SYSTEM/'uqis'/job['name']/'query/registered/identity_evidence.npz')
    query_index = {q: i for i, q in enumerate(identity['query_ids'].tolist())}
    output = []
    for row in report['rows']:
        i = query_index[row['query_id']]; native = identity['native_response'][:, i]
        joint = identity['identity_response'][:, i]
        semantic = np.clip((np.asarray(transfer @ native)+1)*.5, 0, 1).astype(np.float32)
        semantic[transfer.getnnz(axis=1) == 0] = 0
        probabilities, sets = [], []
        for candidate in row['candidates']:
            if candidate['provider'] == 'direct':
                p, s, _ = reader.infer_region(candidate['seed'])
            else:
                _, p, selections, _ = scope.infer(joint, semantic, seed=candidate['seed'])
                p = p.astype(np.float32); s = selections['identity_scope']
            probabilities.append(p); sets.append(s)
        probabilities, sets = np.stack(probabilities), np.stack(sets)
        # The original run replaced the replayed incumbent with reference bytes.
        root = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        probabilities[0] = np.load(root/'predictions'/f"{row['query_id']}.npy")
        sets[0] = np.load(root/'selections'/f"{row['query_id']}.npy")
        if hashlib.sha256(probabilities.tobytes()).hexdigest() != row['candidate_probability_sha256']:
            raise ValueError('complete candidate probabilities did not replay exactly')
        if hashlib.sha256(sets.tobytes()).hexdigest() != row['candidate_selection_sha256']:
            raise ValueError('complete candidate sets did not replay exactly')
        assert row['candidates'][1]['seed'] == row['candidates'][0]['seed']
        assert row['candidates'][1]['provider'] != row['candidates'][0]['provider']
        output.append(dict(query_id=row['query_id'], scene=job['name'], modality=row['modality'],
            candidates=row['candidates'], selected_candidate=row['arms']['full']['candidate'],
            vertex_count=sets.shape[1], packed_sets=np.packbits(sets, axis=1),
            fixed_provider_score=probabilities[1].copy(),
            candidate_sha256=row['candidate_selection_sha256']))
    print('candidate pool replayed exactly', job['name'], len(output), flush=True)
    return output


def main():
    from radio_gs.benchmarks.uqis import MODALITIES
    from radio_gs.benchmarks.uqis.evaluate import score_predictions, score_selections
    run = BoundedExperiment(STUDY, 'response_replacement_capacity_v1', POLICY); run.preflight(5*2**20)
    manifest = read(TRANSFER/'manifest.json'); jobs = manifest['jobs']
    sources = [__file__, 'radio_gs/querying/direct_anchor_membership.py', 'radio_gs/querying/instance_observation_scope.py',
        'radio_gs/querying/anchor_observation_evidence.py', 'radio_gs/querying/observation_readout.py',
        'radio_gs/benchmarks/uqis/evaluate.py', 'radio_gs/benchmarks/uqis/metrics.py']
    run.json('manifest.json', dict(kind='post-seal diagnostic, no new learned policy',
        transfer_summary_sha256=sha(TRANSFER/'summary.json'), source_hashes={p: sha(p) for p in sources},
        default_changed=False, fixed_control='other complete provider at same original identity',
        oracle_is_not_an_automatic_method=True))
    run.json('source_snapshot.json', {p: Path(p).read_text() for p in sources})
    with ThreadPoolExecutor(4) as pool:
        rows = [row for result in pool.map(replay, jobs) for row in result]
    transient = sum(r['packed_sets'].nbytes+r['fixed_provider_score'].nbytes for r in rows)
    if transient > 2**31:
        raise ValueError('transient prediction budget exceeded')
    run.json('predictions_complete.json', dict(queries=len(rows), ground_truth_read=False,
        all_134_original_candidate_pools_exact=True, transient_bytes=transient,
        candidate_sha256={r['query_id']: r['candidate_sha256'] for r in rows}))
    package = Path(jobs[0]['package']); target_path = package/'evaluator/targets.json'; target = read(target_path)
    xyz = {d['scene_id']: np.load(package/'method'/d['mesh_xyz_path']) for d in target['scene_domains']}
    labels = {d['scene_id']: np.load(package/'evaluator'/d['mesh_instance_ids_path']) for d in target['scene_domains']}
    target_by_id = {q: t for t in target['targets'] for q in t['queries'].values()}
    probabilities, selected = {}, {}
    for job in jobs:
        root = SYSTEM/'uqis'/job['name']/'query/registered/identity_scope'
        public, _ = public_queries(Path(job['queries']), job['name'])
        for query in public:
            qid = query['query_id']; probabilities[qid] = np.load(root/'predictions'/f'{qid}.npy')
            selected[qid] = np.load(root/'selections'/f'{qid}.npy')
    details = []
    full_iou = {r['query_id']: r['iou'] for r in read(TRANSFER/'uqis_full_sets.json')['per_query']}
    base_iou = {r['query_id']: r['iou'] for r in read(TRANSFER/'uqis_baseline_sets.json')['per_query']}
    for row in rows:
        qid = row['query_id']; sets = np.unpackbits(row['packed_sets'], axis=1, count=row['vertex_count']).astype(bool)
        truth = labels[row['scene']] == target_by_id[qid]['instance_id']
        ious = (sets & truth).sum(1)/np.maximum((sets | truth).sum(1), 1)
        if abs(ious[0]-base_iou[qid]) > 1e-12 or abs(ious[row['selected_candidate']]-full_iou[qid]) > 1e-12:
            raise ValueError('replayed candidate actual IoU differs from sealed evaluation')
        probabilities[qid] = row['fixed_provider_score']; selected[qid] = sets[1]
        details.append(dict(query_id=qid, scene=row['scene'], modality=row['modality'], candidates=row['candidates'],
            candidate_iou=ious.tolist(), baseline_iou=float(ious[0]), fixed_provider_iou=float(ious[1]),
            selected_iou=full_iou[qid], oracle_iou=float(ious.max())))
    scores = score_predictions(target['targets'], probabilities, xyz, labels)
    sets = score_selections(target['targets'], selected, xyz, labels, MODALITIES)
    run.json('fixed_provider_scores.json', scores); run.json('fixed_provider_sets.json', sets)
    macro = {}
    for modality in ('text', 'image'):
        local = [r for r in details if r['modality'] == modality]
        scenes = sorted({r['scene'] for r in local})
        result = {key: float(np.mean([np.mean([r[key] for r in local if r['scene'] == s]) for s in scenes]))
                  for key in ('baseline_iou', 'fixed_provider_iou', 'selected_iou', 'oracle_iou')}
        result.update(available_gain=result['oracle_iou']-result['baseline_iou'],
                      decision_regret=result['oracle_iou']-result['selected_iou'],
                      learned_minus_fixed_provider=result['selected_iou']-result['fixed_provider_iou'])
        macro[modality] = result
    run.json('summary.json', dict(complete=True, exact_candidate_replay=True, macro=macro, rows=details,
        fixed_provider_metrics={m: dict(ap=scores['modalities'][m]['scene_macro']['average_precision'],
                                      iou=sets['modalities'][m]['scene_macro']['iou']) for m in MODALITIES},
        all_predictions_before_gt=True, diagnostic_only=True, default_changed=False,
        durable_dense_prediction_bytes=0, transient_bytes=transient,
        evaluator_hashes={str(target_path): sha(target_path), **{
            str(package/'evaluator'/d['mesh_instance_ids_path']): sha(package/'evaluator'/d['mesh_instance_ids_path'])
            for d in target['scene_domains']}}))
    del rows, probabilities, selected; gc.collect()
    run.json('ram_released.json', dict(complete=True, study_bytes=tree_bytes(STUDY)))
    print(json.dumps(macro, indent=2), flush=True)


if __name__ == '__main__':
    main()
