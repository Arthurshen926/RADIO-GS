"""Compare complete common-system runs and describe their consumed-output errors.

This is post-prediction analysis. It does not select thresholds, training targets,
queries, scenes or an inference branch. The original five scorers remain intact.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask


def read(path):
    return json.loads(Path(path).read_text())


def set_metrics(prediction, truth):
    intersection = int((prediction & truth).sum())
    predicted, target = int(prediction.sum()), int(truth.sum())
    return dict(iou=intersection / max(predicted + target - intersection, 1),
                purity=intersection / max(predicted, 1),
                coverage=intersection / max(target, 1),
                false_positive=predicted - intersection, false_negative=target - intersection)


def mean(rows, keys=('iou', 'purity', 'coverage')):
    return {key: float(np.mean([r[key] for r in rows])) for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('output/system_v8'))
    args = p.parse_args()
    summary = read(args.system / 'summary.json')
    reference = read(args.reference / 'summary.json')
    result = dict(system=str(args.system), reference=str(args.reference),
                  completed_jobs=summary['completed'],
                  primary=summary['macro'], delta_pp={}, per_scene_delta_pp={},
                  errors={}, evaluation_only=True, evaluation_contracts_unchanged=True,
                  independent_test=False, method_promotion_automatic=False)
    old = {(r['job']['benchmark'], r['job']['name']): r for r in reference['rows']}
    for benchmark, values in summary['macro'].items():
        result['delta_pp'][benchmark] = {
            k: 100 * (v - reference['macro'][benchmark][k]) for k, v in values.items()}
    lerf, nvos, scannet = [], [], []
    for row in summary['rows']:
        job = row['job']; benchmark, name = job['benchmark'], job['name']
        folder = args.system / benchmark / name / 'evaluation'
        result['per_scene_delta_pp'][f'{benchmark}/{name}'] = {
            k: 100 * (v - old[(benchmark, name)]['metrics'][k]) for k, v in row['metrics'].items()}
        if benchmark == 'lerf':
            metrics = read(folder / 'evaluation.json')['metrics']['hierarchical_pipeline']
            response = read(folder / 'query_results.json')['hierarchical_pipeline']
            observations = []
            for obj in metrics['selected']['objects']:
                truth = load_binary_mask(Path(obj['gt_path']), threshold=10)
                prediction = load_binary_mask(Path(obj['pred_path']), threshold=10, grayscale=True)
                observations.append(dict(frame=obj['frame'], query=obj['query'],
                                         **set_metrics(prediction, truth)))
            queries = []
            for text in sorted({r['query'] for r in observations}):
                masks = [r for r in observations if r['query'] == text]
                scores = [r for r in response if r['query'] == text]
                queries.append(dict(query=text, selected=mean(masks),
                    response=mean(scores, ('ap', 'iou', 'localization_hit')),
                    interpretation='frozen response and selected-projection scoring contracts; no query-specific tuning'))
            lerf.append(dict(scene=name, observations=observations, queries=queries, **mean(observations)))
        elif benchmark == 'nvos':
            spec = read(job['evaluation'])
            response = torch.load(folder / 'response.pt', map_location='cpu')['response'].numpy()
            gt = np.asarray(Image.open(spec['gt_mask']))
            truth = np.any(gt > 127, axis=2) if gt.ndim == 3 else gt > 127
            response = cv2.resize(response, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_LINEAR)
            nvos.append(dict(scene=name, **set_metrics(response >= spec['threshold'], truth)))
        elif benchmark == 'scannet':
            metrics = read(folder / 'report.json')['point_metrics']
            splits = {}
            for split, final in metrics['source'].items():
                classes = []
                for class_id, value in final['per_class'].items():
                    if not value['gt_count']:
                        continue
                    tp = value['intersection']
                    classes.append(dict(class_id=class_id, name=value['name'], iou=value['iou'],
                        purity=tp / max(value['pred_count'], 1), coverage=tp / value['gt_count'],
                        false_positive=value['pred_count'] - tp, false_negative=value['gt_count'] - tp))
                splits[split] = dict(**mean(classes), classes=classes,
                    native_miou=metrics['map'][split]['miou'], final_miou=final['miou'])
            scannet.append(dict(scene=name, splits=splits))
    result['errors']['lerf_selected_projection'] = dict(scene_macro=mean(lerf), scenes=lerf,
        domain='2D projection of actually selected Gaussians; not independent 3D truth')
    result['errors']['nvos_response'] = dict(scene_macro=mean(nvos), scenes=nvos,
        domain='retained positive-reference target-view masks')
    result['errors']['scannet'] = dict(scenes=scannet, scene_macro={
        split: mean([s['splits'][split] for s in scannet],
                    ('iou', 'purity', 'coverage', 'native_miou', 'final_miou')) for split in ['19', '15', '10']})
    uqis = read(args.system / 'uqis/identity_scope/result.json')
    result['errors']['uqis'] = dict(modalities={}, paired=uqis['explicit_selection']['paired_query_robustness'])
    for modality, value in uqis['explicit_selection']['modalities'].items():
        rows = [r for r in uqis['explicit_selection']['per_query'] if r['modality'] == modality]
        correct = [r for r in rows if r['target_is_unique_dominant_instance']]
        result['errors']['uqis']['modalities'][modality] = dict(
            scene_macro=value['scene_macro'], target_dominant=len(correct), queries=len(rows),
            identity_correct_subset=dict(query_macro=mean(correct), count=len(correct),
                interpretation='conditional diagnostic; not a method score or an identity oracle'),
            empty_selections=value['empty_selections'])
    result['uqis_stages'] = {}
    for name in ['native_semantic', 'identity_region', 'identity_scope']:
        value = read(args.system / f'uqis/{name}/result.json')
        result['uqis_stages'][name] = dict(
            ap={m: v['scene_macro']['average_precision'] for m, v in value['modalities'].items()},
            actual_iou={m: v['scene_macro']['iou'] for m, v in value.get('explicit_selection', {}).get('modalities', {}).items()})
    result['uqis_repeatability'] = {}
    for name in summary['uqis_readouts']:
        comparison = {}
        for kind in ['predictions', 'selections']:
            files = sorted((args.system / 'uqis' / name / kind).glob('*.npy'))
            if files:
                maximum = 0.; changed = 0
                for file in files:
                    current = np.load(file)
                    previous = np.load(args.reference / 'uqis' / name / kind / file.name)
                    changed += int(not np.array_equal(current, previous))
                    maximum = max(maximum, float(np.max(np.abs(current.astype(np.float32) - previous.astype(np.float32)))))
                comparison[kind] = dict(queries=len(files), changed_queries=changed, max_absolute_difference=maximum)
        result['uqis_repeatability'][name] = comparison
    (args.system / 'common_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(primary=result['primary'], delta_pp=result['delta_pp'],
        output_errors={k: v.get('scene_macro', v.get('modalities')) for k, v in result['errors'].items()}), indent=2))


if __name__ == '__main__':
    main()
