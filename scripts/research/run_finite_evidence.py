"""Matched full-cohort original/tiled evidence controls, no new teacher calls."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def worker(job):
    gpu, cohort, scene = job
    dest = Path(args.output) / cohort / scene
    dest.mkdir(parents=True, exist_ok=True)
    inputs = (Path('output/conditioned_identity/full_source') / scene / 'identity_evidence/inputs.json'
              if cohort == 'original' else Path('output/tiled_source_evidence') / scene / 'admitted/inputs.json')
    commands = [
        [PY, 'scripts/research/decode_expanded_extent_next72.py', '--inputs', str(inputs),
         '--key', 'lerf_' + scene, '--output-base', str(dest / 'sets'), '--mode', 'admission',
         '--scope-initialization', 'source', '--pixel-loss', 'balanced', '--support-prior',
         'footprint', '--raster-scale', '2', '--steps', '240', '--finite-evidence',
         '--evidence-strength', args.evidence_strength],
        [PY, 'scripts/research/fit_consolidated_response.py', '--inputs', str(inputs),
         '--readout', str(dest / 'sets' / ('lerf_' + scene) / 'typed_readout.pt'),
         '--output', str(dest / 'response'), '--policies', 'native_protected',
         '--solver', 'backprojection', '--finite-evidence', '--evidence-strength', args.evidence_strength],
        [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', f'configs/consolidated/{scene}.scene.json',
         '--readout', str(dest / 'response/native_protected/typed_readout.pt'), '--output', str(dest / 'evaluation')],
    ]
    (dest / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (dest / 'run.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print(cohort, scene, 'complete', flush=True)


p = argparse.ArgumentParser()
p.add_argument('--gpus', type=int, nargs='+', default=list(range(6)))
p.add_argument('--output', default='output/finite_source_evidence')
p.add_argument('--evidence-strength', choices=['confidence', 'positive_logit'], default='confidence')
args = p.parse_args()
jobs = [(cohort, scene) for cohort in ['original', 'tiled'] for scene in SCENES]
# Each lane owns one GPU; no overlapping worker can silently reuse its device.
lanes = [[(gpu, *job) for job in jobs[i::len(args.gpus)]] for i, gpu in enumerate(args.gpus)]
def lane(jobs):
    for job in jobs:
        worker(job)
with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
    list(pool.map(lane, lanes))
