"""Posthoc pixel oracle bounds; never produce method predictions or training data."""
from pathlib import Path
import numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_relations import read, write, receipt, check

ROOT = Path('output/query_field_20261011/native_region_appearance_v1')
ARMS = ['semantic_initial', 'frozen_visibility', 'seed_addition']


def bounds(prediction, target):
    p = np.asarray(prediction, dtype=bool); g = np.asarray(target, dtype=bool)
    if p.shape != g.shape: raise ValueError('unaligned masks')
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    # Among all P' containing P, best is P union G: TP'=|G|, FP'=FP.
    # Among all P' contained in P, best is P intersection G: FP'=0, TP'=TP.
    return dict(tp=tp, fp=fp, fn=fn, actual_iou=tp / max(tp + fp + fn, 1),
        addition_only_pixel_ceiling=(tp + fn) / max(tp + fp + fn, 1),
        deletion_only_pixel_ceiling=tp / max(tp + fn, 1))


def main():
    barrier = read(ROOT / 'predictions_complete.json')
    for record in barrier['records']: check(record)
    cfg = read(ROOT / 'prediction_config.json'); scenes = {}; inputs = []
    for j in cfg['jobs']:
        path = ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json'
        evaluation = read(path); inputs.append(receipt(path)); methods = {}
        for arm in ARMS:
            rows = []
            for item in evaluation['metrics'][arm]['selected']['objects']:
                assert not item['missing']
                g = load_binary_mask(Path(item['gt_path']), threshold=10)
                p = load_binary_mask(Path(item['pred_path']), threshold=10, grayscale=True)
                value = bounds(p, g)
                assert abs(value['actual_iou'] - item['iou']) < 1e-12
                rows.append(dict(frame=item['frame'], query=item['query'], **value))
            methods[arm] = dict(cases=len(rows), rows=rows,
                **{k: float(np.mean([r[k] for r in rows])) for k in
                   ['actual_iou', 'addition_only_pixel_ceiling', 'deletion_only_pixel_ceiling']})
        scenes[j['scene']] = methods
    macro = {a: {k: sum(s[a][k] for s in scenes.values()) / len(scenes) for k in
        ['actual_iou', 'addition_only_pixel_ceiling', 'deletion_only_pixel_ceiling']} for a in ARMS}
    report = dict(posthoc_oracle_diagnosis=True, is_method_result=False, model_or_predictions_changed=False,
        target_labels_used_only_in_analysis=True, bound_domain='arbitrary pixel masks; optimistic upper bound for monotone rendered additions',
        assumptions='current rendered positives cannot be removed; same evaluation masks and macro weighting',
        attainable_gaussian_bound=False, ideal_architecture_impossibility=False, macro=macro, scenes=scenes,
        inputs=inputs, code=receipt(__file__))
    write(ROOT / 'monotone_capacity_audit.json', report)
    write('docs/experiments/2026-10-11-monotone-capacity-audit.json', report)
    print(macro, flush=True)


if __name__ == '__main__': main()
