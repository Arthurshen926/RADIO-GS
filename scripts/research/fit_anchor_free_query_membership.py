"""Structural ablation: no guessed positive point or guessed seed membership."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from scripts.research.fit_query_conditioned_membership import ROOT, load_scene, write, ANCHOR_FEATURES
from radio_gs.querying.query_conditioned_membership import QUERY_FEATURES


def evaluate(scene, model, thresholds, intervention=False):
    evidence, labels, queries, identity = load_scene(scene);vertices = np.arange(len(labels));rows = []
    for i, query in enumerate(queries):
        seed, anchor = evidence.context(identity[:, i])
        # The seed column is discarded. No point position, point appearance,
        # point co-membership or selected-region inclusion reaches the model.
        features = evidence.query_features(identity[:, i], seed, vertices)[:, 1:]
        prediction = model.predict_proba(features)[:, 1]
        scores = {'original': prediction}
        if intervention:
            scores['query_mean'] = model.predict_proba(evidence.query_features(identity.mean(1), seed, vertices)[:, 1:])[:, 1]
        truth = labels == query['raw_object_id'] + 1;size = int(truth.sum())
        def metrics(selected):
            tp = int((selected & truth).sum());total = int(selected.sum())
            return [tp / max(total + size - tp, 1), tp / max(total, 1), tp / max(size, 1)]
        rows.append(dict(query=i, anchor_correct=bool(truth[anchor]),
            conditions={name: [metrics(p > t) for t in thresholds] for name, p in scores.items()}))
    value = dict(scene=scene, rows=rows, conditions={name: np.mean([r['conditions'][name] for r in rows], axis=0).tolist()
        for name in rows[0]['conditions']})
    print('evaluated', scene, {name: max(row[0] for row in v) for name, v in value['conditions'].items()}, flush=True)
    return value


def main():
    start = time.monotonic();design = json.loads((ROOT / 'design.json').read_text())
    declaration = dict(condition='query_only', additional_fits=1, same_data_and_capacity=True,
        reason='query information is used, but two of first three external development rooms still regress; isolate dependence on estimated identity',
        features=QUERY_FEATURES[1:], guessed_point_used=False, guessed_region_membership_used=False,
        further_parameter_scan=False,
        gate='exceeds old bridge and supervision-only control each by2pp; at least3 of4 rooms beat old bridge; query-mean intervention inferior',
        benchmark_queries_seen_for_this_ablation=False, independent_final_test=False)
    write(ROOT / 'anchor_free_design.json', declaration)
    data = [np.load(ROOT / 'training' / (s + '.npz')) for s in design['fit']]
    x = np.concatenate([r['x'][:, len(ANCHOR_FEATURES) + 1:] for r in data])
    y = np.concatenate([r['y'] for r in data]);w = np.concatenate([r['weight'] for r in data])
    model = ExtraTreesClassifier(**design['estimator'], n_jobs=8).fit(x, y, sample_weight=w);model.n_jobs = 1
    print('fitted anchor-free', len(x), flush=True)
    grid = np.arange(.05, 1., .05)
    with ThreadPoolExecutor(3) as pool:calibration = list(pool.map(lambda s: evaluate(s, model, grid), design['calibration']))
    macro = np.mean([r['conditions']['original'] for r in calibration], axis=0)
    threshold = float(grid[macro[:, 0].argmax()])
    joblib.dump(dict(estimator=model, threshold=threshold, condition='query_only', supervision=design['supervision'],
        feature_names=QUERY_FEATURES[1:], fit=design['fit'], calibration=design['calibration']), ROOT / 'query_only.joblib', compress=3)
    with ThreadPoolExecutor(4) as pool:development = list(pool.map(lambda s: evaluate(s, model, [threshold], True), design['validation']))
    baseline = json.loads((ROOT / 'fit.json').read_text());old = {r['scene']: r for r in baseline['development']}
    values = {name: np.mean([r['conditions'][name][0] for r in development], axis=0).tolist() for name in ['original', 'query_mean']}
    delta = np.array([r['conditions']['original'][0][0] - old[r['scene']]['old_anchor'][0] for r in development])
    gate = bool(values['original'][0] >= max(baseline['old_anchor_macro'][0], baseline['scene_macro']['anchor_only'][0]) + .02 and
        (delta >= 0).sum() >= 3 and values['original'][0] > values['query_mean'][0])
    report = dict(declaration=declaration, threshold=threshold, calibration=calibration, development=development,
        scene_macro=values, delta_to_old_bridge=float(delta.mean()), per_scene_delta=delta.tolist(), development_gate=gate,
        feature_importance=dict(zip(QUERY_FEATURES[1:], model.feature_importances_.tolist())), seconds=time.monotonic() - start,
        training_samples=len(x), no_estimated_identity_feature=True)
    write(ROOT / 'anchor_free_fit.json', report);print('COMPLETE anchor-free', values, 'gate', gate, flush=True)


if __name__ == '__main__':main()
