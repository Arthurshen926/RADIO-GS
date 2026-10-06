"""Ablate finite constraint during fitting against the same final mixture only."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from radio_gs.querying.finite_evidence import combine_with_native

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
p = argparse.ArgumentParser()
p.add_argument('--root', default='output/finite_source_margin')
p.add_argument('--gpus', nargs='+', type=int, default=list(range(6)))
p.add_argument('--jobs', nargs='+', help='Explicit cohort:scene pairs')
a = p.parse_args()


def worker(job):
    gpu, cohort, scene = job
    root = Path(a.root) / cohort / scene
    dest = root / 'readout_only'
    dest.mkdir(parents=True, exist_ok=True)
    source = (Path('output/conditioned_identity/full_source') / scene / 'sets' / ('lerf_' + scene) / 'typed_readout.pt'
              if cohort == 'original' else Path('output/tiled_source_evidence') / scene / 'global/sets' / ('lerf_' + scene) / 'typed_readout.pt')
    old = torch.load(source, weights_only=False, map_location='cpu')
    data = torch.load(root / 'response/native_protected/typed_readout.pt', weights_only=False, map_location='cpu')
    # The explicit readout includes its exact query-aligned native response,
    # including the extra unlabelled ramen query; never read a scene-only cache.
    prior = old['scores']
    fit = json.loads((root / 'response/fit.json').read_text())
    strength = torch.tensor(fit['source_strength'])[None]
    data['selection_scores'] = combine_with_native(old['selection_scores'], prior, strength)
    data['membership_control'] = 'final readout mixture only; proposal fitted without finite bound'
    torch.save(data, dest / 'typed_readout.pt')
    cmd = [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', f'configs/consolidated/{scene}.scene.json',
           '--readout', str(dest / 'typed_readout.pt'), '--output', str(dest / 'evaluation')]
    (dest / 'control.json').write_text(json.dumps(dict(source=str(source), prior='current native response',
        strength=fit['source_strength'], evidence_strength=fit['evidence_strength'], new_fits=0, command=cmd), indent=2) + '\n')
    with (dest / 'run.log').open('w') as log:
        subprocess.run(cmd, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(cohort, scene, 'readout control complete', flush=True)


jobs = ([tuple(job.split(':')) for job in a.jobs] if a.jobs else
        [(cohort, scene) for cohort in ['original', 'tiled'] for scene in SCENES])
lanes = [[(gpu, *job) for job in jobs[i::len(a.gpus)]] for i, gpu in enumerate(a.gpus)]
def lane(jobs):
    for job in jobs:
        worker(job)
with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
    list(pool.map(lane, lanes))
