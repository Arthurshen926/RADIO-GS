"""Direct-versus-proxy identity: original detections, same density rule and solver."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--context-preserving', action='store_true')
args = parser.parse_args()
ROOT = Path('output/direct_source_context' if args.context_preserving else 'output/direct_source_identity')
JOBS = [(4, 'figurines'), (1, 'ramen'), (3, 'teatime'), (5, 'waldo_kitchen')]


def worker(job):
    gpu, scene = job
    out = ROOT / scene
    out.mkdir(parents=True, exist_ok=True)
    original = str(Path('output/conditioned_identity/full_source') / scene / 'inputs.json')
    inputs = str(out / 'admitted/inputs.json')
    commands = [
        [PY, 'scripts/research/encode_direct_source_identity.py', '--inputs', original, '--output', str(out / 'native')],
        [PY, 'scripts/research/condition_source_identity.py', '--inputs', original,
         '--identity-record', str(out / 'native/identity.json'), '--output', str(out / 'admitted')],
        [PY, 'scripts/research/decode_expanded_extent_next72.py', '--inputs', inputs,
         '--key', 'lerf_' + scene, '--output-base', str(out / 'sets'), '--mode', 'admission',
         '--scope-initialization', 'source', '--pixel-loss', 'balanced', '--support-prior',
         'footprint', '--raster-scale', '2', '--steps', '240'],
        [PY, 'scripts/research/fit_consolidated_response.py', '--inputs', inputs,
         '--readout', str(out / 'sets' / ('lerf_' + scene) / 'typed_readout.pt'),
         '--output', str(out / 'response'), '--policies', 'native_protected', '--solver', 'backprojection'],
        [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', f'configs/consolidated/{scene}.scene.json',
         '--readout', str(out / 'response/native_protected/typed_readout.pt'), '--output', str(out / 'evaluation')],
    ]
    if args.context_preserving:
        commands[0].append('--context-preserving')
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (out / 'run.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print(scene, 'direct identity complete', flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, JOBS))
