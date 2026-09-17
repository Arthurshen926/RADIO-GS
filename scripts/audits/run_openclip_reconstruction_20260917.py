"""Rebuild all encoder-dependent assets for the functional replacement control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    old = base / 'preserved_source_inventory_20260916' / a.scene
    root = base / 'openclip_replacement_20260917' / a.scene
    marker = root / 'reconstruction_execution.json'
    if marker.exists():
        raise FileExistsError(marker)
    jobs = json.loads(Path('paper/artifacts/preserved_inventory_training_protocol_20260916.json').read_text())['jobs']
    job = next(j for j in jobs if j['scene'] == a.scene)
    train = [x.replace(str(old), str(root)).replace('preserved96_protected', 'openclip96_protected') for x in job['command']]
    train[train.index('--device') + 1] = a.device
    commands = [
        [sys.executable, 'scripts/audits/build_source_fusion_initialization.py', '--root', str(root), '--output', str(root / 'source_fusion_initialization.pt')],
        [sys.executable, 'scripts/audits/compile_grounded_text_supervision.py', '--root', str(root), '--grounding', str(root / 'source_grounded_text.pt'), '--output', str(root / 'grounded_text_supervision.pt'), '--device', a.device],
        train,
        [sys.executable, 'scripts/audits/calibrate_frozen_source_images.py', '--root', str(root), '--run', 'openclip96_protected', '--device', a.device],
    ]
    record = dict(scene=a.scene, commands=commands, completed=[], matched_parent_steps=job['steps'])
    marker.write_text(json.dumps(record, indent=2) + '\n')
    for i, command in enumerate(commands):
        print('starting', i, flush=True)
        with (root / f'reconstruction_stage_{i}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        record['completed'].append(i)
        marker.write_text(json.dumps(record, indent=2) + '\n')
        print('completed', i, flush=True)


if __name__ == '__main__':
    main()
