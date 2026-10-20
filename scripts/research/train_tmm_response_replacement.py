"""Train a response-level decision once on independent NR3D/instance supervision.

Frozen teacher descriptors and the complete v10 membership provider are reused.
Only small signatures, decisions, model and metrics are retained. Dense complete
responses have a bounded per-worker LRU cache and are discarded after each room.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.response_replacement import (
    FEATURE_NAMES, RegionResponseEvidence, propose_regions, pair_features,
    relative_utilities, choose_response, calibrate_margin,
)
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/response_replacement.json')


def read(path):
    return json.loads(Path(path).read_text())


def compile_room(scene, split, config):
    start = time.monotonic()
    source = Path(config['source']); folder = source/'maps'/scene
    paths = [folder/n for n in ('membership.npz', 'visibility.npz', 'region_offsets.npy',
                              'mesh_xyz.npy', 'descriptors.npy', 'transfer.npz')]
    paths += [source/'labels'/f'{scene}.npy', source/'language'/f'{scene}.json',
              source/'language'/f'{scene}.npz']
    inputs = {str(p): sha(p) for p in paths}
    membership = sparse.load_npz(folder/'membership.npz')
    visibility = sparse.load_npz(folder/'visibility.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder/'region_offsets.npy')))
    descriptors = np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    reader = DirectAnchorMembership(membership, visibility, frames, np.load(folder/'mesh_xyz.npy'),
        descriptors, sparse.load_npz(folder/'transfer.npz'), config['provider_checkpoint'])
    reader.saved['estimator'].n_jobs = 1
    evidence = RegionResponseEvidence(membership, visibility, frames)
    scope = InstanceObservationScope(membership, visibility, frames)
    transfer = reader.evidence.transfer
    supported = transfer.getnnz(axis=1) > 0
    queries = read(source/'language'/f'{scene}.json')
    vectors = np.load(source/'language'/f'{scene}.npz')['pooled'].astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    indices = list(range(len(queries)))
    if split == 'fit':
        indices = [i for i in indices if queries[i]['listener_correct'] and queries[i]['mentions_target_class']]
        indices.sort(key=lambda i: hashlib.sha256(f"{scene}/{queries[i]['record']}/response-replacement".encode()).hexdigest())
        indices = indices[:config['fit_queries_per_scene']]
    # This is labelled independent training/development, not benchmark inference.
    labels = np.load(source/'labels'/f'{scene}.npy')
    cache = OrderedDict(); rows = []; decode_calls = 0
    for i in indices:
        query = queries[i]
        scores = descriptors @ vectors[i]
        seeds = propose_regions(scores, evidence.available, frames, budget=config['candidate_budget'])
        decoded = []
        proposal_seeds = []; providers = []
        semantic = np.clip((np.asarray(transfer @ scores)+1)*.5, 0, 1).astype(np.float32)
        semantic[~supported] = 0
        for seed in seeds:
            if seed not in cache:
                cache[seed] = reader.infer_region(seed); decode_calls += 1
                # The cache never exceeds 64 complete responses per worker.
                if len(cache) > 64:
                    cache.popitem(last=False)
            cache.move_to_end(seed)
            decoded.append(cache[seed])
            proposal_seeds.append(seed); providers.append('direct')
            _, scoped_rank, scoped_sets, _ = scope.infer(scores, semantic, seed=seed)
            decoded.append((scoped_rank.astype(np.float32), scoped_sets['identity_scope'], {'anchor_vertex': None}))
            proposal_seeds.append(seed); providers.append('scope')
        probability = np.stack([r[0] for r in decoded])
        selected = np.stack([r[1] for r in decoded])
        excluded = sorted(set(frames[seeds].tolist()))
        x, info = evidence.encode(probability, selected, scores, exclude_views=excluded)
        control, _ = evidence.encode(probability, selected, scores, exclude_views=excluded, permuted=True)
        truth = labels == query['raw_object_id']+1
        ious = ((selected & truth).sum(1)/np.maximum((selected | truth).sum(1), 1)).tolist()
        rows.append(dict(scene=scene, query_index=i, record=query['record'], seeds=proposal_seeds, providers=providers,
            anchors=[r[2]['anchor_vertex'] for r in decoded], features=x.tolist(),
            permuted_features=control.tolist(), ious=ious, evidence=info,
            truth_elements=int(truth.sum()), prediction_sha256=hashlib.sha256(probability.tobytes()).hexdigest()))
        if len(rows) % 16 == 0:
            print('compiled', split, scene, len(rows), '/', len(indices), 'decodes', decode_calls, flush=True)
    report = dict(scene=scene, split=split, rows=rows, input_hashes=inputs,
                  queries=len(rows), decode_calls=decode_calls, seconds=time.monotonic()-start,
                  dense_predictions_retained=False, benchmark_labels_used=False)
    print('room complete', split, scene, len(rows), 'seconds', round(report['seconds'], 1), flush=True)
    return report


def training_pairs(reports, blind):
    x, y, weight = [], [], []
    for report in reports:
        for row in report['rows']:
            features = np.asarray(row['features']); ious = row['ious']; k = len(ious)
            if k < 2:
                continue
            for a in range(k):
                for b in range(k):
                    if a != b:
                        x.append(pair_features(features[a], features[b], query_blind=blind))
                        y.append(ious[b]-ious[a])
                        weight.append(1/(len(report['rows'])*k*(k-1)))
    if not x:
        raise ValueError('no complete-response pairs')
    return np.asarray(x, np.float32), np.asarray(y), np.asarray(weight)


def infer_rows(reports, model, *, blind=False, permuted=False):
    rows = []
    for report in reports:
        for row in report['rows']:
            features = row['permuted_features' if permuted else 'features']
            rows.append(dict(scene=report['scene'], query_index=row['query_index'], ious=row['ious'],
                gains=relative_utilities(model, features, query_blind=blind).tolist(),
                evidence_available=row['evidence']['verification_views'] > 0))
    return rows


def summarize(rows, margin):
    records = []
    for row in rows:
        k = choose_response(row['gains'], margin, evidence_available=row['evidence_available'])
        records.append(dict(scene=row['scene'], query_index=row['query_index'], candidate=k,
            gains=row['gains'], baseline_iou=row['ious'][0], selected_iou=row['ious'][k],
            oracle_iou=max(row['ious'])))
    per_scene = {}
    for scene in sorted({r['scene'] for r in records}):
        local = [r for r in records if r['scene'] == scene]
        per_scene[scene] = {key: float(np.mean([r[key] for r in local]))
                           for key in ('baseline_iou', 'selected_iou', 'oracle_iou')}
        per_scene[scene]['delta'] = per_scene[scene]['selected_iou']-per_scene[scene]['baseline_iou']
    macro = {key: float(np.mean([r[key] for r in per_scene.values()])) for key in next(iter(per_scene.values()))}
    return dict(margin=margin, queries=len(records), scene_macro=macro, per_scene=per_scene, records=records,
                changed=sum(r['candidate'] != 0 for r in records),
                improved=sum(r['selected_iou'] > r['baseline_iou']+1e-12 for r in records),
                harmed=sum(r['selected_iou'] < r['baseline_iou']-1e-12 for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='response_replacement_train_v2')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read(CONFIG); design_path = Path(config['source'])/'design.json'; design = read(design_path)
    splits = {k: design[k] for k in ('fit', 'calibration', 'validation')}
    flat = sum(splits.values(), [])
    benchmarks = read('configs/pipeline/system_v10.json')['jobs']
    if len(set(flat)) != len(flat) or set(flat) & {j['name'] for j in benchmarks}:
        raise ValueError('independent training/calibration/development split overlaps a benchmark')
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume)
    run.preflight(24*2**20)
    sources = [str(CONFIG), 'scripts/research/train_tmm_response_replacement.py',
        'radio_gs/querying/response_replacement.py', 'radio_gs/querying/direct_anchor_membership.py',
        'radio_gs/querying/anchor_observation_evidence.py', 'radio_gs/querying/instance_observation_scope.py',
        'radio_gs/querying/observation_readout.py', 'radio_gs/query_field/bounded_experiment.py']
    manifest = dict(config=config, splits=splits, feature_names=FEATURE_NAMES,
        source_hashes={p: sha(p) for p in sources},
        input_hashes={str(design_path): sha(design_path), config['provider_checkpoint']: sha(config['provider_checkpoint'])})
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('resume requires the same sources/config/provider')
    else:
        run.json('manifest.json', manifest)
        run.json('source_snapshot.json', {p: Path(p).read_text() for p in sources})
    all_reports = {}
    # Fit/calibrate first. The decision checkpoint is sealed before development
    # responses are compiled; development cannot choose models or thresholds.
    for split in ('fit', 'calibration'):
        reports = []
        pending = []
        for scene in splits[split]:
            path = run.root/f'{scene}.json'
            if path.exists():
                report = read(path)
                if any(sha(p) != h for p, h in report['input_hashes'].items()):
                    raise ValueError('resumed room inputs changed')
                reports.append(report)
            else:
                pending.append(scene)
        with ThreadPoolExecutor(args.workers) as pool:
            for report in pool.map(lambda s: compile_room(s, split, config), pending):
                run.json(report['scene']+'.json', report); reports.append(report)
        all_reports[split] = sorted(reports, key=lambda r: r['scene'])
    models = {}; calibration = {}; bundle = dict(feature_names=FEATURE_NAMES, models={}, margins={},
        config=config, splits=splits, training_manifest_sha256=sha(run.root/'manifest.json'))
    for condition in config['conditions']:
        blind = condition == 'query_blind'
        x, y, weights = training_pairs(all_reports['fit'], blind)
        model = ExtraTreesRegressor(**config['estimator'], n_jobs=4).fit(x, y, sample_weight=weights)
        model.n_jobs = 1; models[condition] = model
        rows = infer_rows(all_reports['calibration'], model, blind=blind)
        selected, detail = calibrate_margin(rows)
        calibration[condition] = dict(selection=selected, rule=detail, evaluation=summarize(rows, selected['margin']),
            training_pairs=len(y), training_queries=sum(r['queries'] for r in all_reports['fit']))
        bundle['models'][condition] = model; bundle['margins'][condition] = selected['margin']
        print('fit/calibrated', condition, selected, flush=True)
    buffer = io.BytesIO(); joblib.dump(bundle, buffer, compress=3); payload = buffer.getvalue()
    if not (run.root/'model.joblib').exists():
        run._write('model.joblib', lambda f: f.write(payload), limit=len(payload))
        run.json('calibration.json', calibration)
        run.json('model_seal.json', dict(model_sha256=sha(run.root/'model.joblib'),
            calibration_sha256=sha(run.root/'calibration.json'), benchmark_feedback_used=False,
            development_feedback_used=False, default_changed=False))
    elif hashlib.sha256(payload).hexdigest() != sha(run.root/'model.joblib'):
        raise ValueError('resumed model differs from sealed model')
    development = []
    with ThreadPoolExecutor(args.workers) as pool:
        for report in pool.map(lambda s: compile_room(s, 'validation', config), splits['validation']):
            if not (run.root/(report['scene']+'.json')).exists():
                run.json(report['scene']+'.json', report)
            development.append(report)
    result = {}
    for condition, model in models.items():
        result[condition] = summarize(infer_rows(development, model, blind=condition == 'query_blind'), bundle['margins'][condition])
    result['permuted'] = summarize(infer_rows(development, models['full'], permuted=True), bundle['margins']['full'])
    result['unrestricted'] = summarize(infer_rows(development, models['full']), 0.)
    main_result = result['full']; gain = main_result['scene_macro']['delta']
    gate = (gain > 0 and all(main_result['scene_macro']['selected_iou'] > result[a]['scene_macro']['selected_iou']
                            for a in ('query_blind', 'permuted'))
            and sum(r['delta'] >= 0 for r in main_result['per_scene'].values()) >= 3)
    summary = dict(complete=True, development=result, development_gate=gate,
        training_queries=sum(r['queries'] for r in all_reports['fit']),
        calibration_queries=sum(r['queries'] for r in all_reports['calibration']),
        development_queries=sum(r['queries'] for r in development),
        checkpoint_sha256=sha(run.root/'model.joblib'), default_changed=False,
        independent_final_test=False, all_five_transfer_pending=True, durable_dense_prediction_bytes=0,
        study_bytes=tree_bytes(STUDY))
    run.json('summary.json', summary)
    print(json.dumps({k: {'macro': v['scene_macro'], 'changed': v['changed'], 'improved': v['improved'], 'harmed': v['harmed']}
                      for k, v in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
