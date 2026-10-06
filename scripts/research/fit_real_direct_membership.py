"""Matched real/rendered supervision for direct positive-anchor membership."""
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

from radio_gs.querying.direct_anchor_membership import DirectAnchorEvidence, view_balanced_transfer, FEATURE_NAMES
from radio_gs.querying.anchor_observation_evidence import anchor_membership

ROOT = Path('output/real_direct_membership_20261006')
REAL = Path('output/real_referent_grounding_20261005')
RENDERED = Path('output/referent_grounding_20261005')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_scene(scene, domain):
    folder = (REAL if domain == 'real' else RENDERED) / 'maps' / scene
    b = sparse.load_npz(folder / 'membership.npz');v = sparse.load_npz(folder / 'visibility.npz')
    offsets = np.load(folder / 'region_offsets.npy')
    frames = np.repeat(np.arange(v.shape[1]), np.diff(offsets))
    transfer = sparse.load_npz(folder / 'transfer.npz') if domain == 'real' else view_balanced_transfer(b, offsets)
    evidence = DirectAnchorEvidence(b, v, frames, np.load(folder / 'mesh_xyz.npy'),
                                   np.load(folder / 'descriptors.npy'), transfer)
    return evidence, np.load(REAL / 'labels' / (scene + '.npy'))


def episodes(scene, labels):
    for instance in np.unique(labels[labels >= 0]):
        target = np.flatnonzero(labels == instance)
        seed = int.from_bytes(hashlib.sha256(f'{scene}/{instance}/positive-point'.encode()).digest()[:8], 'little')
        anchor = int(target[np.random.default_rng(seed).integers(len(target))])
        yield int(instance), target, anchor, seed


def compile_scene(scene, domain):
    evidence, labels = load_scene(scene, domain)
    x = [];y = [];weights = [];records = []
    for instance, target, anchor, seed in episodes(scene, labels):
        rng = np.random.default_rng(seed);sample = [];weight = [];known_count = int((labels >= 0).sum())
        for positive in [False, True]:
            pool = np.flatnonzero((labels >= 0) & ((labels == instance) == positive))
            chosen = rng.choice(pool, min(len(pool), 1024), replace=False)
            sample.extend(chosen.tolist());weight.extend([len(pool) / max(len(chosen), 1) / known_count] * len(chosen))
        sample = np.asarray(sample)
        x.append(evidence.encode(anchor, sample));y.append(labels[sample] == instance);weights.extend(weight)
        records.append(dict(instance=instance, anchor=anchor, sampled=len(sample), target_size=len(target),
            has_anchor_observation=bool(evidence.membership[anchor].nnz)))
    folder = ROOT / domain / 'training';folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (scene + '.npz'), x=np.concatenate(x), y=np.concatenate(y),
                        weight=np.array(weights) / len(records))
    write(folder / (scene + '.json'), records)
    print('compiled', domain, scene, len(records), sum(len(a) for a in x), flush=True)


def evaluate(scene, model, thresholds):
    evidence, labels = load_scene(scene, 'real');vertices = np.arange(len(labels));rows = []
    for instance, target, anchor, seed in episodes(scene, labels):
        prediction = model.predict_proba(evidence.encode(anchor, vertices))[:, 1];prediction[anchor] = 1.
        native, _ = anchor_membership(evidence.membership, evidence.visibility, evidence.frames, anchor)
        truth = labels == instance
        def metrics(selected):
            tp = int((selected & truth).sum());size = int(selected.sum())
            return [tp / max(size + len(target) - tp, 1), tp / max(size, 1), tp / len(target)]
        rows.append(dict(instance=instance, anchor=anchor, native=metrics(native > .5),
            metrics=[metrics(prediction > t) for t in thresholds],
            has_anchor_observation=bool(evidence.membership[anchor].nnz)))
    record = dict(scene=scene, targets=len(rows), rows=rows,
        macro=np.mean([r['metrics'] for r in rows], axis=0).tolist(),
        native=np.mean([r['native'] for r in rows], axis=0).tolist())
    print('evaluated', scene, 'targets', len(rows), 'native', record['native'][0], flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['rendered', 'real'], required=True)
    p.add_argument('--stage', choices=['compile', 'fit'], required=True)
    a = p.parse_args();ROOT.mkdir(exist_ok=True);out = ROOT / a.domain;out.mkdir(exist_ok=True)
    design = json.loads((REAL / 'design.json').read_text())
    if a.stage == 'compile':
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(lambda s: compile_scene(s, a.domain), design['fit']))
        return
    start = time.monotonic();data = [np.load(out / 'training' / (s + '.npz')) for s in design['fit']]
    x = np.concatenate([r['x'] for r in data]);y = np.concatenate([r['y'] for r in data]);w = np.concatenate([r['weight'] for r in data])
    model = ExtraTreesClassifier(n_estimators=64, max_leaf_nodes=512, min_samples_leaf=32,
        max_features=1., n_jobs=4, random_state=20261005).fit(x, y, sample_weight=w)
    thresholds = np.arange(.05, 1., .05)
    # Each classifier uses the same deployed real-observation calibration domain.
    model.n_jobs = 1
    with ThreadPoolExecutor(3) as pool:
        calibration = list(pool.map(lambda s: evaluate(s, model, thresholds), design['calibration']))
    macro = np.mean([r['macro'] for r in calibration], axis=0)
    threshold = float(thresholds[macro[:, 0].argmax()])
    joblib.dump(dict(estimator=model, threshold=threshold, feature_names=FEATURE_NAMES,
        condition=f'independent3D instance supervision;22 {a.domain} observation rooms;actual geometric anchor',
        fit=design['fit'], calibration=design['calibration']), out / 'model.joblib', compress=3)
    with ThreadPoolExecutor(3) as pool:
        development = list(pool.map(lambda s: evaluate(s, model, [threshold]), design['validation']))
    report = dict(condition=a.domain, training_samples=len(x), training_positives=int(y.sum()),
        threshold=threshold, calibration_thresholds=thresholds.tolist(), calibration=calibration,
        validation=development, validation_macro=np.mean([r['macro'][0] for r in development], axis=0).tolist(),
        native_validation_macro=np.mean([r['native'] for r in development], axis=0).tolist(),
        feature_importance=dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
        seconds=time.monotonic() - start, benchmark_labels_used=False, complete_carrier_evaluation=True)
    write(out / 'fit.json', report)
    print('complete', a.domain, report['validation_macro'], 'native', report['native_validation_macro'], flush=True)


if __name__ == '__main__':
    main()
