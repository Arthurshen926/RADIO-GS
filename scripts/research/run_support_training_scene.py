"""Run both fixed source-only fits and teacher diagnostics for one scene."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    study = json.loads(Path('configs/query_field/support_training_study.json').read_text())
    jobs = [j for j in study['fits'] if j['scene'] == args.scene]
    if not jobs:
        raise ValueError('scene is not in the fixed study')
    for job in jobs:
        for stage in ['fit', 'evaluate']:
            command = [sys.executable, '-m', 'radio_gs.query_field', stage,
                       '--config', job['config'], '--output', job['output']]
            with (Path(job['output'])/f'{stage}.log').open('w') as handle:
                subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
            print(job['name'], stage, 'complete', flush=True)


if __name__ == '__main__':
    main()
