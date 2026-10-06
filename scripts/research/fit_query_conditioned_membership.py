"""Matched query-ground-truth supervision with/without retained query evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier

from radio_gs.querying.query_conditioned_membership import QueryMembershipEvidence, FEATURE_NAMES, ANCHOR_FEATURES

ROOT = Path('output/query_conditioned_membership_20261006')
SOURCE = Path('output/native_phrase_binding_20261006')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_scene(scene):
    folder = SOURCE / 'maps' / scene
    membership = sparse.load_npz(folder / 'membership.npz');visibility = sparse.load_npz(folder / 'visibility.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(folder / 'region_offsets.npy')))
    evidence = QueryMembershipEvidence(membership, visibility, frames, np.load(folder / 'mesh_xyz.npy'),
        np.load(folder / 'descriptors.npy'), sparse.load_npz(folder / 'transfer.npz'))
    labels = np.load(SOURCE / 'labels' / (scene + '.npy'))
    queries = json.loads((SOURCE / 'language' / (scene + '.json')).read_text())
    vectors = np.load(SOURCE / 'language' / (scene + '.npz'))['pooled'].astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    identity = evidence.descriptors @ vectors.T
    return evidence, labels, queries, identity


def compile_scene(scene):
    evidence, labels, queries, identity = load_scene(scene)
    rows = [];targets = [];weights = [];records = []
    eligible = [(i, q) for i, q in enumerate(queries) if q['listener_correct'] and q['mentions_target_class']]
    for i, query in eligible:
        target = query['raw_object_id'] + 1
        seed = int.from_bytes(hashlib.sha256(f"{scene}/{query['record']}/query-membership".encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed);sample = [];weight = [];known = int((labels >= 0).sum())
        for positive in [False, True]:
            pool = np.flatnonzero((labels >= 0) & ((labels == target) == positive))
            chosen = rng.choice(pool, min(len(pool), 1024), replace=False)
            sample.extend(chosen.tolist());weight.extend([len(pool) / max(len(chosen), 1) / known] * len(chosen))
        vertices = np.asarray(sample)
        features, region, anchor = evidence.encode_query(identity[:, i], vertices)
        rows.append(features);targets.append(labels[vertices] == target);weights.extend(weight)
        records.append(dict(query=i, target=int(target), estimated_region=region, estimated_anchor=anchor,
            anchor_correct=bool(labels[anchor] == target), samples=len(vertices)))
    folder = ROOT / 'training';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / (scene + '.npz'), x=np.concatenate(rows), y=np.concatenate(targets),
        weight=np.asarray(weights) / len(eligible))
    write(folder / (scene + '.json'), records)
    print('compiled', scene, len(eligible), 'queries', 'anchor_correct', sum(r['anchor_correct'] for r in records), flush=True)


def evaluate(scene, models, thresholds, interventions=False):
    evidence, labels, queries, identity = load_scene(scene);vertices = np.arange(len(labels));rows = []
    old = joblib.load('output/real_direct_membership_20261006/real/model.joblib')
    family = sparse.load_npz(SOURCE / 'maps' / scene / 'scope_selections.npz').tocsc()
    for i, query in enumerate(queries):
        truth = labels == query['raw_object_id'] + 1;size = int(truth.sum())
        x, seed, anchor = evidence.encode_query(identity[:, i], vertices)
        def metrics(selected):
            tp = int((selected & truth).sum());total = int(selected.sum())
            return [tp / max(total + size - tp, 1), tp / max(total, 1), tp / max(size, 1)]
        predictions = {name: model.predict_proba(x if name == 'query_evidence' else x[:, :len(ANCHOR_FEATURES)])[:, 1]
            for name, model in models.items()}
        if interventions:
            corrupted = np.column_stack([x[:, :len(ANCHOR_FEATURES)], evidence.query_features(identity.mean(1), seed, vertices)])
            predictions['query_mean'] = models['query_evidence'].predict_proba(corrupted)[:, 1]
        baseline = old['estimator'].predict_proba(x[:, :len(ANCHOR_FEATURES)])[:, 1];baseline[anchor] = 1.
        native = np.zeros(len(labels), bool);native[family[:, seed].indices] = True
        rows.append(dict(query=i, target=int(query['raw_object_id'] + 1), seed=seed, anchor=anchor,
            anchor_correct=bool(truth[anchor]), native=metrics(native), old_anchor=metrics(baseline > old['threshold']),
            conditions={name: [metrics(p > t) for t in thresholds[name]] for name, p in predictions.items()}))
    result = dict(scene=scene, queries=len(rows), rows=rows,
        native=np.mean([r['native'] for r in rows], axis=0).tolist(),
        old_anchor=np.mean([r['old_anchor'] for r in rows], axis=0).tolist(),
        conditions={name: np.mean([r['conditions'][name] for r in rows], axis=0).tolist() for name in rows[0]['conditions']})
    print('evaluated', scene, 'old', result['old_anchor'][0], 'new', result['conditions'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage', choices=['compile', 'fit'], required=True)
    args = parser.parse_args();ROOT.mkdir(exist_ok=True)
    source = json.loads((SOURCE / 'design.json').read_text());groups = {key: source[key] for key in ['fit', 'calibration', 'validation']}
    design = dict(**groups, source=str(SOURCE), conditions=['anchor_only', 'query_evidence'],
        supervision='independent NR3D full expressions and official3D target instances, real registered observations',
        label_identity='actual referred instance, not inferred-anchor instance',
        unknown_labels='excluded from training; all elements scored at evaluation',
        native_identity='same full-expression descriptor; no phrase parsing or encoder update',
        anchor='same selected source region and geometric median as previous bridge; never forced positive',
        candidates=2, training_vertex_budget_per_class_query=1024, candidate_or_mapping_cap=None,
        estimator=dict(n_estimators=64, max_leaf_nodes=512, min_samples_leaf=32, max_features=1., random_state=20261005),
        calibration_rule='one threshold per condition on same3 calibration rooms; no benchmark threshold choice',
        development_gate='query_evidence beats anchor_only and old bridge each by2pp; at least3 of4 scenes beat old bridge; query-mean evidence inferior',
        source_images_at_query=False, independent_final_test=False, default_changed=False)
    write(ROOT / 'design.json', design)
    if args.stage == 'compile':
        with ThreadPoolExecutor(4) as pool:list(pool.map(compile_scene, groups['fit']))
        return
    start = time.monotonic();data = [np.load(ROOT / 'training' / (scene + '.npz')) for scene in groups['fit']]
    x = np.concatenate([r['x'] for r in data]);y = np.concatenate([r['y'] for r in data]);w = np.concatenate([r['weight'] for r in data])
    models = {}
    for condition in design['conditions']:
        model = ExtraTreesClassifier(**design['estimator'], n_jobs=6)
        model.fit(x if condition == 'query_evidence' else x[:, :len(ANCHOR_FEATURES)], y, sample_weight=w)
        model.n_jobs = 1;models[condition] = model
        print('fitted', condition, 'samples', len(x), flush=True)
    grid = np.arange(.05, 1., .05);thresholds = {name: grid for name in models}
    with ThreadPoolExecutor(3) as pool:calibration = list(pool.map(lambda s: evaluate(s, models, thresholds), groups['calibration']))
    selected = {name: float(grid[np.mean([r['conditions'][name] for r in calibration], axis=0)[:, 0].argmax()]) for name in models}
    for name, model in models.items():
        joblib.dump(dict(estimator=model, threshold=selected[name], condition=name, supervision=design['supervision'],
            feature_names=FEATURE_NAMES if name == 'query_evidence' else ANCHOR_FEATURES,
            fit=groups['fit'], calibration=groups['calibration']), ROOT / (name + '.joblib'), compress=3)
    thresholds = {name: [value] for name, value in selected.items()};thresholds['query_mean'] = thresholds['query_evidence']
    with ThreadPoolExecutor(4) as pool:development = list(pool.map(lambda s: evaluate(s, models, thresholds, True), groups['validation']))
    macro = {name: np.mean([r['conditions'][name][0] for r in development], axis=0).tolist() for name in thresholds}
    old_macro = np.mean([r['old_anchor'] for r in development], axis=0).tolist()
    delta = np.array([r['conditions']['query_evidence'][0][0] - r['old_anchor'][0] for r in development])
    gate = bool(macro['query_evidence'][0] >= max(macro['anchor_only'][0], old_macro[0]) + .02 and
        (delta >= 0).sum() >= 3 and macro['query_evidence'][0] > macro['query_mean'][0])
    result = dict(design=design, training_samples=len(x), training_queries=sum(len(json.loads((ROOT / 'training' / (s + '.json')).read_text())) for s in groups['fit']),
        calibration=calibration, thresholds=selected, development=development, scene_macro=macro, old_anchor_macro=old_macro,
        development_gate=gate, seconds=time.monotonic() - start,
        feature_importance={name: dict(zip(FEATURE_NAMES if name == 'query_evidence' else ANCHOR_FEATURES, model.feature_importances_.tolist())) for name, model in models.items()})
    write(ROOT / 'fit.json', result);print('COMPLETE', macro, 'old', old_macro, 'gate', gate, flush=True)


if __name__ == '__main__':main()
