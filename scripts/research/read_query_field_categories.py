"""Frozen typed category readout for the parent and both relation-training arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.query_field.category_evidence import corroborated_category_scores
from radio_gs.query_field.storage import sha256
from scripts.research.run_query_field_relations import ROOT, ARMS, read, write, receipt, check

OUT = ROOT / 'category_evidence'
PROTOCOL = Path('docs/experiments/2026-10-08-typed-category-evidence-protocol.md')


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'design.json', dict(protocol=receipt(PROTOCOL), code=[receipt(Path(__file__)),
        receipt(Path('radio_gs/query_field/category_evidence.py')),
        receipt(Path('radio_gs/querying/observation_readout.py'))],
        v10_temperature_source=receipt(Path('configs/pipeline/scannet_dual_native/scene0000_00.scene.json')),
        fields=['parent', *ARMS], labels_used=False))


def cohorts():
    result = {'parent': read('configs/query_field/five_benchmark_v1/cohort.json')}
    result.update({a: read(ROOT / a / 'cohort.json') for a in ARMS})
    return result


def predict():
    read(ROOT / 'predictions_complete.json')
    design = read(OUT / 'design.json')
    for r in design['code'] + [design['protocol'], design['v10_temperature_source']]: check(r)
    records = []
    for arm, cohort in cohorts().items():
        for job in cohort['scannet']:
            source = Path(cohort['root']) / 'predictions/scannet' / job['name']
            original = read(source / 'prediction.json')
            if sha256(source / 'predictions.pt') != original['predictions_sha256']:
                raise ValueError('category source predictions changed')
            p = torch.load(source / 'predictions.pt', map_location='cpu', weights_only=True)
            out = OUT / arm / job['name']; out.mkdir(parents=True, exist_ok=False)
            evidence = corroborated_category_scores(p['native'], p['member'])
            torch.save(dict(xyz=p['xyz'], observed=p['observed'], queries=p['queries'],
                            native=p['native'], evidence=evidence), out / 'predictions.pt')
            write(out / 'prediction.json', dict(source=receipt(source / 'predictions.pt'),
                payload=receipt(out / 'predictions.pt'), source_record=receipt(source / 'prediction.json'),
                source_access=False, evaluator_access=False, complete_menu=[q['class_id'] for q in p['queries']],
                changed_vs_native=int(((evidence.argmax(1) != p['native'].argmax(1)) & p['observed']).sum()),
                observed=int(p['observed'].sum()), scores_calibrated=False))
            records.append(receipt(out / 'prediction.json'))
    write(OUT / 'predictions_complete.json', dict(records=records, design=receipt(OUT / 'design.json')))


def score():
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from radio_gs.scannet_constants import compute_split_metrics
    barrier = read(OUT / 'predictions_complete.json'); check(barrier['design'])
    if len(barrier['records']) != 30: raise ValueError('typed category cohort incomplete')
    for r in barrier['records']:
        check(r); check(read(r['path'])['payload'])
    reports = []
    for arm, cohort in cohorts().items():
        for job in cohort['scannet']:
            out = OUT / arm / job['name']; report = read(out / 'prediction.json')
            p = torch.load(out / 'predictions.pt', map_location='cpu', weights_only=True)
            spec = read(job['evaluation'])
            gt = PlyData.read(spec['ground_truth'])['vertex'].data
            xyz = np.stack([gt[k] for k in ['x', 'y', 'z']], 1)
            _, index = cKDTree(p['xyz'].numpy()).query(xyz, workers=4)
            metrics = {}
            for method in ['native', 'evidence']:
                metrics[method] = {}
                for split, ids in spec['splits'].items():
                    columns = [next(i for i, q in enumerate(p['queries']) if q['class_id'] == c) for c in ids]
                    labels = np.asarray(ids)[p[method][:, columns].argmax(1).numpy()]
                    labels[~p['observed'].numpy()] = 0
                    metrics[method][split] = compute_split_metrics(labels[index], np.asarray(gt['label']), ids)
            value = dict(arm=arm, scene=job['name'], metrics=metrics, prediction=receipt(out / 'prediction.json'),
                         ground_truth=receipt(Path(spec['ground_truth'])), changed_vs_native=report['changed_vs_native'],
                         observed=report['observed'])
            write(out / 'evaluation.json', value); reports.append(value)
    write(OUT / 'evaluation.json', dict(complete=True, records=reports,
        macro={a: {m: {s: float(np.mean([r['metrics'][m][s]['miou'] for r in reports if r['arm'] == a]))
                       for s in ['19', '15', '10']} for m in ['native', 'evidence']} for a in ['parent', *ARMS]},
        independent_final_test=False, accuracy_promotion=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare', 'predict', 'score'])
    args = p.parse_args(); torch.set_num_threads(4)
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
