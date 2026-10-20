"""Rescore frozen v10 menus on VALA's paper-eight Gaussian metric.

This changes only the evaluator, not predictions, supervision, or providers.
The provided-mesh Gaussian carrier remains different from VALA RGB geometry.
No pseudo labels or dense predictions are persisted; complete prediction hashes
are sealed before opening label PLYs. Original ten-scene mesh scores stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from plyfile import PlyData
import torch
import yaml

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    assign_vala_pseudo_labels, volume_weighted_split_metrics)
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

REPO = Path(__file__).resolve().parents[2]
SYSTEM = REPO / 'output/query_conditioned_membership_20261006/image_convergence/system'
FREEZE = REPO / 'paper/artifacts/evaluation_protocol_freeze_20260801.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='scannet_paper8_metric_bridge_v1')
    args = parser.parse_args()
    os.chdir(REPO)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    torch.set_num_threads(4)
    started = time.monotonic()
    run = BoundedExperiment(STUDY, args.run, POLICY)
    run.preflight(2**20)
    freeze = yaml.safe_load(FREEZE.read_text())['canonical_tasks']['concept_scannet_ovs_vala_paper8']
    scenes = freeze['cohort']['scenes']
    inputs = {str(FREEZE.relative_to(REPO)): sha(FREEZE),
              str(Path(__file__).relative_to(REPO)): sha(__file__),
              'radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py':
              sha('radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py')}
    predictions, geometries, specs, seals = {}, {}, {}, {}
    # Produce and bind ALL predictions before any target labels are opened.
    for scene in scenes:
        path = SYSTEM / 'scannet' / scene / 'query/category/typed_readout.pt'
        spec_path = REPO / f'configs/consolidated/scannet_native/{scene}.evaluation.json'
        data = torch.load(path, map_location='cpu', weights_only=False)
        spec = json.loads(spec_path.read_text())
        inputs[str(path.relative_to(REPO))] = sha(path)
        inputs[str(spec_path.relative_to(REPO))] = sha(spec_path)
        geometry = data['semantic_geometry']
        geo = FrozenGaussianGeometry.load(geometry['path'], expected_sha256=geometry['sha256'])
        inputs[geometry['path']] = geometry['sha256']
        classes = [q['class_id'] for q in data['queries']]
        predictions[scene] = {}
        for split, ids in spec['splits'].items():
            columns = [classes.index(c) for c in ids]
            # Match the retained categorical readout, including unobserved voids.
            pred = np.asarray(ids, np.int32)[data['category_logits'][:, columns].argmax(1).numpy()]
            pred[~data['semantic_observed'].numpy()] = 0
            predictions[scene][split] = pred
        geometries[scene], specs[scene] = geo, spec
        seals[scene] = {split: hashlib.sha256(p.astype('<i4').tobytes()).hexdigest()
                        for split, p in predictions[scene].items()}
        print('sealed', scene, len(geo.xyz), flush=True)
    run.json('prediction_seal.json', dict(scenes=scenes, predictions=seals, inputs=inputs,
        predictions_before_labels=True, provider='retained v10', model_selection=False,
        input_track='provided_unlabelled_mesh; source-assisted teachers',
        exact_vala_input_equivalence=False, durable_dense_prediction_bytes=0))
    rows = {}
    for scene in scenes:
        geo, spec = geometries.pop(scene), specs[scene]
        label_path = Path(spec['ground_truth'])
        vertices = PlyData.read(label_path)['vertex'].data
        xyz = np.stack([vertices[a] for a in ('x', 'y', 'z')], 1)
        pseudo, stats = assign_vala_pseudo_labels(geo.xyz.numpy(), geo.scales.numpy(),
            geo.quaternions.numpy(), xyz, np.asarray(vertices['label'], dtype=np.int32),
            radius_factor=5., candidate_k=1000, fallback_k=1, class_balance=True)
        weights = (geo.opacities.flatten()*geo.scales.prod(1)).numpy()
        splits = {split: volume_weighted_split_metrics(pseudo, predictions[scene][split], weights, ids)
                  for split, ids in spec['splits'].items()}
        rows[scene] = dict(splits=splits, pseudo_gt=stats,
            label_file=str(label_path), label_sha256=sha(label_path),
            pseudo_gt_sha256=hashlib.sha256(pseudo.astype('<i4').tobytes()).hexdigest())
        run.json(scene+'.json', rows[scene])
        print('scored', scene, {k:v['miou'] for k,v in splits.items()}, flush=True)
        del predictions[scene]
    macro = {split: {metric: float(np.mean([r['splits'][split][metric] for r in rows.values()]))
                    for metric in ('miou', 'macc')} for split in ('19', '15', '10')}
    summary = dict(complete=True, scenes=scenes, macro=macro, rows=rows,
        prediction_seal_sha256=sha(run.root/'prediction_seal.json'),
        metric_protocol='VALA paper8; Mahalanobis pseudo labels; opacity-volume weights',
        exact_vala_input_equivalence=False, new_method_gain=False,
        meaning='metric bridge on existing provided-mesh predictions; not RGB-only VALA reproduction',
        durable_dense_prediction_bytes=0, elapsed_seconds=time.monotonic()-started)
    run.json('summary.json', summary)
    print(json.dumps(macro), flush=True)


if __name__ == '__main__':
    main()
