"""Fixed LERF/NVOS predictions must all precede new target-mask scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/support_training_v1')
LERF = Path('configs/query_field/support_training_study.json')
NVOS = Path('configs/query_field/support_training_nvos.json')


def execute(command, log):
    with Path(log).open('w') as handle:
        subprocess.run([sys.executable, *command], check=True, stdout=handle, stderr=subprocess.STDOUT)


def complete_prediction_barrier():
    receipts = []
    for path, directory in [(LERF, ROOT/'lerf'), (NVOS, ROOT/'nvos')]:
        config = json.loads(path.read_text())
        for job in config['jobs']:
            root = directory/job['name']
            record = json.loads((root/'prediction.json').read_text())
            if record['configuration'] != config or sha256(root/'predictions.pt') != record['predictions_sha256']:
                raise ValueError('fixed prediction cohort changed')
            receipts.append(dict(job=job['name'], prediction_sha256=record['predictions_sha256']))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['predict', 'score', 'cold'])
    parser.add_argument('--scene', choices=['figurines', 'ramen', 'fern'], required=True)
    args = parser.parse_args()
    path = NVOS if args.scene == 'fern' else LERF
    config = json.loads(path.read_text())
    directory = ROOT/('nvos' if args.scene == 'fern' else 'lerf')
    directory.mkdir(exist_ok=True)
    jobs = config['jobs'] if args.scene == 'fern' else [j for j in config['jobs'] if j['scene'] == args.scene]
    if args.stage == 'cold':
        if args.scene != 'fern':
            raise ValueError('use normal four-input smoke for LERF')
        complete_prediction_barrier()
        from scripts.research.run_query_field_nvos import replay
        replay(path, directory, ROOT)
        return
    if args.stage == 'score':
        receipts = complete_prediction_barrier()
        cold = json.loads((ROOT/'cold_replay.json').read_text())
        if not all(all(r['exact'].values()) for r in cold['comparisons']):
            raise ValueError('cold prompt replay failed')
        (ROOT/f'{args.scene}_prescore_barrier.json').write_text(json.dumps(receipts, indent=2)+'\n')
    if args.scene == 'fern' and args.stage == 'score':
        execute(['-m', 'radio_gs.query_field.public_nvos', 'score', '--config',
                 'configs/query_field/support_training_nvos_scoring.json', '--output', str(directory)], ROOT/'nvos_score.log')
    else:
        module = 'radio_gs.query_field.public_nvos' if args.scene == 'fern' else 'radio_gs.query_field.metric_study'
        for job in jobs:
            execute(['-m', module, args.stage, '--config', str(path), '--job', job['name'],
                     '--output', str(directory)], ROOT/f"{args.stage}_{job['name']}.log")
            print(args.stage, job['name'], 'complete', flush=True)


if __name__ == '__main__':
    main()
