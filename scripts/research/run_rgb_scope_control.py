"""Matched-pixel RGB correspondence intervention under the frozen LERF scorer."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--root', default='output/rgb_scope_correspondence')
parser.add_argument('--scenes', nargs='+', default=['ramen', 'teatime'])
parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5])
args = parser.parse_args()


def run(gpu, scene):
    root = Path(args.root) / scene
    inputs = root / 'evidence/inputs.json'
    commands = [
        [PY, 'scripts/research/decode_expanded_extent_next72.py', '--inputs', str(inputs), '--key', 'lerf_' + scene,
         '--output-base', str(root / 'sets'), '--mode', 'admission', '--scope-initialization', 'source',
         '--pixel-loss', 'balanced', '--support-prior', 'footprint', '--raster-scale', '2', '--steps', '240'],
        [PY, 'scripts/research/fit_consolidated_response.py', '--inputs', str(inputs),
         '--readout', str(root / 'sets' / ('lerf_' + scene) / 'typed_readout.pt'), '--output', str(root / 'response'),
         '--policies', 'native_protected', '--solver', 'backprojection'],
        [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', f'configs/consolidated/{scene}.scene.json',
         '--readout', str(root / 'response/native_protected/typed_readout.pt'), '--output', str(root / 'evaluation')],
    ]
    (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (root / 'run.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(scene, 'complete', flush=True)


def lane(item):
    gpu, scenes = item
    for scene in scenes:
        run(gpu, scene)


with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
    list(pool.map(lane, [(gpu, args.scenes[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]))
