"""Execute the declared study; all prediction cohorts precede target scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from radio_gs.query_field.storage import load_field, save_field, sha256


ROOT = Path('output/query_field_20261007/expected_support_v1')
STUDY = Path('configs/query_field/expected_support_study.json')
NVOS = Path('configs/query_field/expected_support_nvos.json')


def execute(command, log):
    if Path(log).exists():
        raise FileExistsError(log)
    with Path(log).open('w') as handle:
        subprocess.run([sys.executable, *command], check=True, stdout=handle, stderr=subprocess.STDOUT)


def barrier():
    receipts = []
    for path, output in [(STUDY, ROOT/'lerf'), (NVOS, ROOT/'nvos')]:
        config = json.loads(path.read_text())
        for job in config['jobs']:
            directory = output/job['name']
            report = json.loads((directory/'prediction.json').read_text())
            if report['configuration'] != config or sha256(directory/'predictions.pt') != report['predictions_sha256']:
                raise ValueError('fixed prediction cohort changed')
            receipts.append(dict(job=job['name'], sha256=report['predictions_sha256']))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'evaluate', 'predict', 'cold', 'score', 'smoke'])
    parser.add_argument('--scene', required=True, choices=['figurines', 'ramen', 'fern'])
    args = parser.parse_args()
    study = json.loads(STUDY.read_text())
    models = [j for j in study['models'] if j['scene'] == args.scene]
    if args.stage in {'fit', 'evaluate', 'smoke'}:
        for job in models:
            output = Path(job['output'])
            if args.stage == 'fit' and job['arm'] == 'frozen_expected':
                field, parent = load_field(job['parent_field'])
                field.support_aggregation = 'expected_energy'
                save_field(field, job['field'], dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
                    observations_sha256=parent['provenance']['observations_sha256'],
                    operation='Frozen expected-support energy; all parent tensors retained',
                    extra_3d_labels=False, benchmark_queries_used=False))
                del field
            else:
                execute(['-m', 'radio_gs.query_field', args.stage, '--config', job['config'], '--output', str(output)],
                        output/f'{args.stage}.log')
            print(args.stage, job['name'], 'complete', flush=True)
        return
    if args.stage == 'cold':
        barrier()
        from scripts.research.run_query_field_nvos import replay
        replay(NVOS, ROOT/'nvos', ROOT)
        return
    if args.stage == 'score':
        receipts = barrier()
        cold = json.loads((ROOT/'cold_replay.json').read_text())
        if len(cold['comparisons']) != 6 or not cold['original_open_rejected'] or not all(all(r['exact'].values()) for r in cold['comparisons']):
            raise ValueError('source-denied prompt replay incomplete')
        receipt = ROOT/f'{args.scene}_prescore_barrier.json'
        if receipt.exists():
            raise FileExistsError(receipt)
        receipt.write_text(json.dumps(receipts, indent=2)+'\n')
    if args.scene == 'fern' and args.stage == 'score':
        execute(['-m', 'radio_gs.query_field.public_nvos', 'score', '--config',
                 'configs/query_field/expected_support_nvos_scoring.json', '--output', str(ROOT/'nvos')], ROOT/'score_fern.log')
        return
    path = NVOS if args.scene == 'fern' else STUDY
    config = json.loads(path.read_text())
    jobs = config['jobs'] if args.scene == 'fern' else [j for j in config['jobs'] if j['scene'] == args.scene]
    module = 'radio_gs.query_field.public_nvos' if args.scene == 'fern' else 'radio_gs.query_field.metric_study'
    output = ROOT/('nvos' if args.scene == 'fern' else 'lerf')
    for job in jobs:
        execute(['-m', module, args.stage, '--config', str(path), '--job', job['name'], '--output', str(output)],
                ROOT/f"{args.stage}_{job['name']}.log")
        print(args.stage, job['name'], 'complete', flush=True)


if __name__ == '__main__':
    main()
