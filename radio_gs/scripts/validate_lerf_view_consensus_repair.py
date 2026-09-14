"""Frozen-association ablation of source-view-diverse prototype receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.build_lerf_object_hypotheses import _select_hypothesis_prototypes, validate_hypothesis_memory
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import _load_fragment_prototypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    jobs = []
    for scene in ('figurines', 'ramen', 'teatime', 'waldo_kitchen'):
        base = args.result_root / 'v4_affinity_lerf_transfer_20260914/full_v2'
        original = base / (scene + '.pt')
        hypothesis = torch.load(original, weights_only=False, map_location='cpu')
        fragment_path = Path(hypothesis['fragment_memory'])
        if sha256_file(fragment_path) != hypothesis['fragment_memory_sha256']:
            raise ValueError('fragment hash differs')
        fragment = torch.load(fragment_path, weights_only=False, map_location='cpu')
        state = _load_state(Path(hypothesis['scene_state']), expected_sha256=hypothesis['scene_state_sha256'])
        flat, _, _ = _load_fragment_prototypes(Path(hypothesis['appearance_audit']['fragment_language_manifest']), state=state)
        count = hypothesis['fragment_count']
        if flat.shape[0] != 2 * count:
            raise ValueError('source prototype axes differ')
        prototypes = _select_hypothesis_prototypes(torch.stack((flat[:count], flat[count:]), 1),
            hypothesis['fragment_assignment'], fragment['quality'],
            maximum_prototypes=hypothesis['object_prototype_valid'].shape[1],
            fragment_view_index=fragment['fragment_view_index'])
        prototypes['object_prototype_descriptors'] = prototypes['object_prototype_descriptors'].half()
        hypothesis.update(prototypes)
        hypothesis['prototype_repair'] = {'parent_sha256': sha256_file(original),
            'selection': 'distinct_source_views_first', 'consensus': 'distinct_source_views',
            'association_frozen': True, 'development_only': True}
        validate_hypothesis_memory(hypothesis)
        path = args.output / (scene + '.pt')
        torch.save(hypothesis, path)
        repaired_text = args.result_root / 'v4_repaired_query_inputs_20260914'
        cmd = [sys.executable, '-m', 'radio_gs.scripts.validate_lerf_resolution_repair',
            '--text-report', str(base / (scene + '_text.json')),
            '--scene-root', str(args.data_root / scene), '--label-root', str(args.data_root / 'label'),
            '--gt-root', str(args.result_root / 'protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt' / scene / 'gt'),
            '--output-directory', str(args.output / scene), '--hypothesis-memory', str(path),
            '--positive-text-cache', str(repaired_text / (scene + '_text_query_cache.pt')),
            '--negative-text-cache', str(repaired_text / (scene + '_negative_text_query_cache.pt')),
            '--formal-fragment-memory', str(args.result_root / 'v4_native_fragment_repair_20260914' / (scene + '.pt')),
            '--fixed-only']
        jobs.append((scene, cmd))
    (args.output / 'commands.json').write_text(json.dumps(jobs, indent=2))
    def worker(lane):
        for scene, cmd in jobs[lane::2]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(2 + lane))
            with (args.output / (scene + '.log')).open('x') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(scene, 'completed', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, (0, 1)))


if __name__ == '__main__':
    main()
