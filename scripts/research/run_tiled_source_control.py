"""All-query shared-evidence control: global versus viewport-bounded negatives."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT = Path('output/tiled_source_evidence')
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def worker(job):
    gpu, scene = job
    out = ROOT / scene
    raw_inputs = Path('output/conditioned_identity/full_source') / scene / 'inputs.json'
    commands = [
        [PY, 'scripts/research/merge_tiled_source_evidence.py', '--inputs', str(raw_inputs),
         '--tiles', str(out / 'tiles'), '--output', str(out / 'combined')],
        [PY, 'scripts/research/filter_source_identity.py', '--inputs', str(out / 'combined/inputs.json'),
         '--output', str(out / 'identity_records'), '--heldout-stride', '0', '--record-only'],
        [PY, 'scripts/research/condition_source_identity.py', '--inputs', str(out / 'combined/inputs.json'),
         '--identity-record', str(out / 'identity_records/identity.json'), '--output', str(out / 'admitted')],
        [PY, 'scripts/research/bound_viewport_evidence.py', '--inputs', str(out / 'admitted/inputs.json'),
         '--output', str(out / 'bounded_evidence')],
    ]
    for arm, evidence in [('global', 'admitted'), ('bounded', 'bounded_evidence')]:
        dest = out / arm
        manifest = str(out / evidence / 'inputs.json')
        commands += [
            [PY, 'scripts/research/decode_expanded_extent_next72.py', '--inputs', manifest,
             '--key', 'lerf_' + scene, '--output-base', str(dest / 'sets'), '--mode', 'admission',
             '--scope-initialization', 'source', '--pixel-loss', 'balanced', '--support-prior',
             'footprint', '--raster-scale', '2', '--steps', '240'],
            [PY, 'scripts/research/fit_consolidated_response.py', '--inputs', manifest,
             '--readout', str(dest / 'sets' / ('lerf_' + scene) / 'typed_readout.pt'),
             '--output', str(dest / 'response'), '--policies', 'native_protected', '--solver', 'backprojection'],
            [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', f'configs/consolidated/{scene}.scene.json',
             '--readout', str(dest / 'response/native_protected/typed_readout.pt'), '--output', str(dest / 'evaluation')],
        ]
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (out / 'fit.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print(scene, 'dual output controls complete', flush=True)


p = argparse.ArgumentParser()
p.add_argument('--scenes', nargs='+', default=SCENES)
p.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
a = p.parse_args()
with ThreadPoolExecutor(max_workers=len(a.scenes)) as pool:
    list(pool.map(worker, zip(a.gpus, a.scenes)))
