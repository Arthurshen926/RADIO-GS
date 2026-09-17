"""Fixed-budget development controls; each result retains its complete parent."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--variants', nargs='+', default=['explicit', 'mlp', 'hash', 'moge_output', 'moge_refiner'])
    a = p.parse_args()
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base / 'scope_consistent_episodes_20260916' / a.scene
    for variant in a.variants:
        command = [sys.executable, 'scripts/audits/train_bounded_structure_exploration.py',
                   '--root', str(root), '--geometry', str(base / 'geometry_only_repair_final_20260914' / a.scene / 'independent_scale.pt'),
                   '--parent', 'scope96_protected_imagecal_balanced', '--name', 'bounded_' + variant,
                   '--variant', variant, '--passes', '2', '--device', a.device]
        if variant == 'moge_refiner':
            command += ['--moge-root', str(base / 'moge_feature_teacher_20260917' / a.scene)]
        print('starting', variant, flush=True)
        with (root / ('bounded_' + variant + '.log')).open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        print('completed', variant, flush=True)


if __name__ == '__main__':
    main()
