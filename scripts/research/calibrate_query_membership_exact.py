"""Choose the exact scene-macro calibration cut, without a probability floor."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import joblib
import numpy as np

from scripts.research.fit_query_conditioned_membership import ROOT, load_scene, write, ANCHOR_FEATURES

NAMES = ['anchor_only', 'query_evidence', 'query_only']


def features_by_model(x):
    return dict(anchor_only=x[:, :len(ANCHOR_FEATURES)], query_evidence=x,
                query_only=x[:, len(ANCHOR_FEATURES) + 1:])


def curve(probability, truth, weight):
    order = np.argsort(-probability, kind='stable');values = probability[order]
    end = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    tp = np.cumsum(truth[order], dtype=np.int64)[end]
    iou = tp / (end + 1 + truth.sum() - tp)
    delta = np.diff(np.r_[0., iou]) * weight
    active = delta != 0
    return values[end][active], delta[active]


def best_threshold(curves):
    values = np.concatenate([r[0] for r in curves]);delta = np.concatenate([r[1] for r in curves])
    order = np.argsort(-values, kind='stable');values = values[order];delta = delta[order]
    end = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    macro = np.cumsum(delta)[end];index = int(macro.argmax())
    # Runtime uses float32 ranks and strict >. Include all tied elements at the
    # chosen breakpoint with exactly the same float32 decision in deployment.
    threshold = float(np.nextafter(values[end[index]], np.float32(-np.inf)))
    return dict(threshold=threshold, calibration_scene_macro_iou=float(macro[index]),
                unique_breakpoints=len(end), probability_floor=None, threshold_grid=None)


def main():
    start = time.monotonic();design = json.loads((ROOT / 'design.json').read_text())
    saved = {n: joblib.load(ROOT / (n + '.joblib')) for n in NAMES}
    write(ROOT / 'exact_calibration_design.json', dict(
        motivation='anchor-free best grid threshold equals historical0.05 lower boundary; do not infer method failure from an artificial cutoff',
        method='exact weighted IoU breakpoint merge across every calibration element and query; one threshold per fixed model',
        threshold_grid=None, probability_floor=None, models_refit=False,
        labels_used='only3 existing calibration rooms; no development or benchmark labels select thresholds',
        development_gates_changed=False, independent_final_test=False))
    scenes = {s: load_scene(s) for s in design['calibration']}
    tasks = [(s, i) for s, (_, _, q, _) in scenes.items() for i in range(len(q))]
    def calibration(task):
        scene, i = task;evidence, labels, queries, identity = scenes[scene]
        x, _, _ = evidence.encode_query(identity[:, i], np.arange(len(labels)))
        truth = labels == queries[i]['raw_object_id'] + 1;weight = 1 / len(scenes) / len(queries)
        result = {name: curve(saved[name]['estimator'].predict_proba(x_)[:, 1].astype(np.float32), truth, weight)
                  for name, x_ in features_by_model(x).items()}
        return result
    with ThreadPoolExecutor(6) as pool:
        rows = []
        for i, row in enumerate(pool.map(calibration, tasks)):
            rows.append(row)
            if (i + 1) % 40 == 0:print('exact calibration queries', i + 1, '/', len(tasks), flush=True)
    selected = {name: best_threshold([r[name] for r in rows]) for name in NAMES}
    for name in NAMES:
        saved[name]['previous_grid_threshold'] = saved[name]['threshold']
        saved[name]['threshold'] = selected[name]['threshold']
        saved[name]['calibration_rule'] = 'exact scene-macro IoU breakpoint; all calibration scores; no probability floor'
        joblib.dump(saved[name], ROOT / (name + '_exact.joblib'), compress=3)
    write(ROOT / 'exact_calibration.json', selected);print('exact cuts', selected, flush=True)
    del rows, scenes
    scenes = {s: load_scene(s) for s in design['validation']}
    tasks = [(s, i) for s, (_, _, q, _) in scenes.items() for i in range(len(q))]
    def development(task):
        scene, i = task;evidence, labels, queries, identity = scenes[scene]
        x, seed, anchor = evidence.encode_query(identity[:, i], np.arange(len(labels)))
        truth = labels == queries[i]['raw_object_id'] + 1;size = int(truth.sum())
        mean_features = evidence.query_features(identity.mean(1), seed, np.arange(len(labels)))
        variants = features_by_model(x)
        variants.update(query_evidence_mean=np.column_stack([x[:, :len(ANCHOR_FEATURES)], mean_features]),
                        query_only_mean=mean_features[:, 1:])
        metrics = {}
        for variant, features in variants.items():
            name = variant.removesuffix('_mean');model = saved[name]
            p = model['estimator'].predict_proba(features)[:, 1].astype(np.float32)
            selection = p > model['threshold'];tp = int((selection & truth).sum());total = int(selection.sum())
            metrics[variant] = [tp / max(total + size - tp, 1), tp / max(total, 1), tp / max(size, 1)]
        return dict(scene=scene, query=i, anchor_correct=bool(truth[anchor]), metrics=metrics)
    with ThreadPoolExecutor(6) as pool:
        rows = []
        for i, row in enumerate(pool.map(development, tasks)):
            rows.append(row)
            if (i + 1) % 40 == 0:print('exact-cut development queries', i + 1, '/', len(tasks), flush=True)
    grouped = {s: [r for r in rows if r['scene'] == s] for s in scenes}
    per_scene = {s: {name: np.mean([r['metrics'][name] for r in rr], axis=0).tolist()
                    for name in rr[0]['metrics']} for s, rr in grouped.items()}
    macro = {name: np.mean([rr[name] for rr in per_scene.values()], axis=0).tolist()
             for name in rows[0]['metrics']}
    previous = json.loads((ROOT / 'fit.json').read_text());old = {r['scene']: r['old_anchor'][0] for r in previous['development']}
    gates = {name: bool(macro[name][0] >= max(previous['old_anchor_macro'][0], macro['anchor_only'][0]) + .02 and
        sum(r[name][0] >= old[s] for s, r in per_scene.items()) >= 3 and macro[name][0] > macro[name + '_mean'][0])
        for name in ['query_evidence', 'query_only']}
    report = dict(calibration=selected, per_scene=per_scene, rows=rows, scene_macro=macro, development_gates=gates,
        model_refitted=False, benchmark_data_used=False, seconds=time.monotonic() - start)
    write(ROOT / 'exact_development.json', report);print('COMPLETE exact', macro, 'gates', gates, flush=True)


if __name__ == '__main__':main()
