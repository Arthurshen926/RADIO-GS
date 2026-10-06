"""Close typed identity, fixed-identity scope transfer and the common execution."""
import json
from pathlib import Path

import numpy as np

from scripts.research.fit_typed_referent_binding import ROOT, write

BASE = Path('output/real_direct_membership_20261006/real/isolated_system')


def compare_arrays(root, evaluation):
    result = dict(native_arrays=0, point_rank_arrays=0, point_selection_arrays=0,
        native_exact=True, point_rank_exact=True, point_selection_exact=True, maximum_difference=0.)
    for row in evaluation['explicit_selection']['per_query']:
        scene, qid = row['scene_id'], row['query_id']
        parent = BASE / 'uqis' / scene / 'query/registered'
        current = root / 'uqis' / scene / 'query/registered'
        pairs = [('native_semantic/predictions', 'native_arrays', 'native_exact')]
        if row['modality'] in ['point_2d', 'point_3d']:
            pairs.extend([('identity_scope/predictions', 'point_rank_arrays', 'point_rank_exact'),
                          ('identity_scope/selections', 'point_selection_arrays', 'point_selection_exact')])
        for folder, count, exact in pairs:
            old = np.load(parent / folder / (qid + '.npy'));new = np.load(current / folder / (qid + '.npy'))
            result[count] += 1;result[exact] &= np.array_equal(old, new)
            result['maximum_difference'] = max(result['maximum_difference'], float(np.abs(old.astype(float) - new.astype(float)).max()))
    return result


def effects(current, previous):
    result = {}
    for modality in ['text', 'image', 'point_2d', 'point_3d']:
        a = current['explicit_selection']['modalities'][modality]['per_scene']
        b = previous['explicit_selection']['modalities'][modality]['per_scene']
        names = sorted(a);deltas = np.array([a[name]['iou'] - b[name]['iou'] for name in names])
        rng = np.random.default_rng(20261006)
        interval = np.quantile(deltas[rng.integers(0, len(deltas), (20000, len(deltas)))].mean(1), [.025, .975])
        result[modality] = dict(mean=float(deltas.mean()), bootstrap95=interval.tolist(),
            improved=int((deltas > 0).sum()), per_scene=dict(zip(names, deltas.tolist())))
    return result


def compact(evaluation):
    target_rows = evaluation['explicit_selection']['paired_query_robustness']['per_target']
    return dict(modalities={key: dict(evaluation['explicit_selection']['modalities'][key]['scene_macro'],
        ap=evaluation['modalities'][key]['scene_macro']['average_precision']) for key in ['text', 'image', 'point_2d', 'point_3d']},
        all_four_success_count=sum(int(r['all_four_iou_at_least_050']) for r in target_rows), targets=len(target_rows))


def main():
    fits = {name: json.loads((ROOT / name / 'fit.json').read_text()) for name in ['semantic', 'frozen', 'joint']}
    frozen = fits['frozen']['validation']['conditions'];joint = fits['joint']['validation']['conditions']
    semantic = fits['semantic']['validation']['conditions']['original']
    relation_gate = dict(passed=frozen['original']['iou'] >= max(.11109819856937975, semantic['iou']) + .02 and
        frozen['original']['iou'] > joint['original']['iou'] and
        frozen['original']['class_accuracy'] >= semantic['class_accuracy'],
        frozen=frozen['original'], joint=joint['original'], semantic=semantic,
        frozen_semantic_parameters_exact=fits['frozen']['semantic_parameters_unchanged'],
        explanation='Frozen parameters do not certify unchanged final identity decisions.')
    names = ['semantic_bridge', 'semantic_family', 'native_bridge']
    systems = {name: json.loads((ROOT / name / 'system/summary.json').read_text()) for name in names}
    evaluations = {name: json.loads((ROOT / name / 'system/uqis/identity_scope/result.json').read_text()) for name in names}
    baseline = json.loads((BASE / 'uqis/identity_scope/result.json').read_text());evaluations['v9'] = baseline
    exact = {name: compare_arrays(ROOT / name / 'system', evaluations[name]) for name in names}
    deltas = effects(evaluations['semantic_bridge'], baseline)
    metrics = {name: compact(value) for name, value in evaluations.items()}
    attribution = {pair: effects(evaluations[a], evaluations[b]) for pair, a, b in [
        ('lexical_at_family', 'semantic_family', 'v9'), ('lexical_at_bridge', 'semantic_bridge', 'native_bridge'),
        ('scope_at_native', 'native_bridge', 'v9'), ('scope_at_lexical', 'semantic_bridge', 'semantic_family')]}
    old_public = json.loads(Path('output/real_direct_membership_20261006/final_validation.json').read_text())['macro']
    public = systems['semantic_bridge']['macro']
    public_difference = {benchmark: {key: public[benchmark][key] - value for key, value in old_public[benchmark].items()}
        for benchmark in ['lerf', 'nvos', 'scannet']}
    successes = lambda data: {(r['scene_id'], r['instance_id']) for r in data['explicit_selection']['paired_query_robustness']['per_target']
                             if r['all_four_iou_at_least_050']}
    old, new = successes(baseline), successes(evaluations['semantic_bridge'])
    point_and_native_exact = all(exact['semantic_bridge'][key] for key in ['native_exact', 'point_rank_exact', 'point_selection_exact'])
    retained_public = all(abs(value) < 1e-5 for metrics_ in public_difference.values() for value in metrics_.values())
    promoted = (all(deltas[m]['mean'] >= .02 and deltas[m]['bootstrap95'][0] > 0 for m in ['text', 'image']) and
        len(new) >= len(old) and point_and_native_exact and retained_public)
    gate = dict(passed=promoted, text=deltas['text'], image=deltas['image'], joint_success_non_decreasing=len(new) >= len(old),
        native_and_points_exact=point_and_native_exact, public_retained=retained_public,
        protocol='existing frozen development benchmarks; not independent final test or official SOTA equivalence')
    report = dict(status='complete', fits={name: {key: value[key] for key in ['selected_epoch', 'seconds', 'calibration', 'validation',
        'total_parameters', 'trainable_parameters']} for name, value in fits.items()}, relation_gate=relation_gate,
        membership_development=json.loads((ROOT / 'membership_bridge/summary.json').read_text()),
        benchmark_jobs=sum(value['completed'] for value in systems.values()), uqis_queries=sum(value['query_count'] for key, value in evaluations.items() if key != 'v9'),
        benchmark_metrics=metrics, primary_public_metrics={b: public[b] for b in ['lerf', 'nvos', 'scannet']},
        public_difference=public_difference, deltas=deltas, attribution=attribution, unchanged_outputs=exact,
        all_four_transitions=dict(old=len(old), new=len(new), gained=sorted(new - old), lost=sorted(old - new)),
        promotion_gate=gate, independent_final_test=False)
    write(ROOT / 'comparison.json', report)
    print(json.dumps(dict(metrics=metrics, relation_gate=relation_gate, promotion_gate=gate,
        attribution={key: {m: v['mean'] for m, v in values.items()} for key, values in attribution.items()}), indent=2), flush=True)


if __name__ == '__main__':main()
