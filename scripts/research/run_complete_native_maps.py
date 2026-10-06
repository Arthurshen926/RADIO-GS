"""Rebuild every LERF source region and evaluate one complete, fixed recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess

PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES = ['ramen', 'figurines', 'teatime', 'waldo_kitchen']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--evaluate-built-maps', action='store_true')
    args = parser.parse_args()
    root = Path('output/system_map_rebuild')
    results = []
    for name in SCENES:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        commands = []

        def run(arguments, log):
            command = [PYTHON, *map(str, arguments)]
            commands.append(command)
            with (folder / log).open('w') as stream:
                subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=args.gpu),
                               stdout=stream, stderr=subprocess.STDOUT, check=True)

        spec = folder / 'complete/scene.json'
        queries = f'configs/consolidated/{name}.integration-queries.json' if name == 'ramen' else f'configs/consolidated/{name}.queries.json'
        if not args.evaluate_built_maps:
            run(['-m', 'radio_gs.pipeline.region_teacher', '--scene', f'configs/pipeline/reconstruction/{name}.json',
                 '--output', folder / 'regions'], 'regions.log')
            run(['-m', 'radio_gs.pipeline.build_map', '--scene', f'configs/pipeline/reconstruction/{name}.json',
                 '--regions', folder / 'regions', '--output', folder / 'complete'], 'complete.log')
        run(['-m', 'radio_gs.pipeline', '--scene', spec, '--queries', queries, '--access', 'map-only',
             '--gpu', args.gpu, '--output', folder / 'map_query'], 'map_query.log')
        response = folder / 'map_query/native/response.pt'
        run(['scripts/research/evaluate_consolidated_query.py', '--scene', spec, '--readout', response,
             '--direct-map', '--output', folder / 'map_evaluation'], 'map_evaluation.log')
        raw = json.loads(Path(f'output/conditioned_identity/full_source/{name}/inputs.json').read_text())
        run(['-m', 'radio_gs.pipeline', '--scene', spec, '--queries', queries, '--access', 'source-assisted',
             '--native-response', response, '--source-evidence', raw['detections'], '--gpu', args.gpu,
             '--output', folder / 'source_query'], 'source_query.log')
        run(['scripts/research/evaluate_consolidated_query.py', '--scene', spec,
             '--readout', folder / 'source_query/typed_readout.pt', '--output', folder / 'source_evaluation'],
            'source_evaluation.log')
        admitted = json.loads(Path(f'output/conditioned_identity/full_source/{name}/identity_evidence/inputs.json').read_text())
        run(['-m', 'radio_gs.pipeline', '--scene', spec, '--queries', queries, '--access', 'source-assisted',
             '--native-response', response, '--source-evidence', admitted['detections'], '--provider-admitted',
             '--gpu', args.gpu, '--output', folder / 'fixed_evidence_query'], 'fixed_evidence_query.log')
        run(['scripts/research/evaluate_consolidated_query.py', '--scene', spec,
             '--readout', folder / 'fixed_evidence_query/typed_readout.pt', '--output', folder / 'fixed_evidence_evaluation'],
            'fixed_evidence_evaluation.log')
        def metrics(path):
            return json.loads(Path(path).read_text())['metrics']['hierarchical_pipeline']
        row = dict(scene=name, encoding=json.loads((folder / 'regions/encoding.json').read_text()),
                   map_only=metrics(folder / 'map_evaluation/evaluation.json'),
                   source_assisted=metrics(folder / 'source_evaluation/evaluation.json'),
                   fixed_source_evidence=metrics(folder / 'fixed_evidence_evaluation/evaluation.json'),
                   retained_source=metrics(f'output/consolidated_response_control/{name}/fusion/native_protected/evaluation/evaluation.json'))
        (folder / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        (folder / 'comparison.json').write_text(json.dumps(row, indent=2) + '\n')
        results.append(row)
        print(name, 'complete', flush=True)
    summary = dict(scenes=results, scope='all nonempty source SAM regions; unchanged geometry, native encoder and frozen calibration',
                   cached_raw_source_detections=True, identity_and_both_readouts_reconstructed=True,
                   independent_test=False, historical_query_predictions_read=False)
    summary['macro'] = {kind: {metric: sum(row[kind][metric] for row in results) / len(results)
                              for metric in ['iou', 'ap', 'localization']}
                        for kind in ['map_only', 'source_assisted', 'fixed_source_evidence', 'retained_source']}
    for kind in summary['macro']:
        summary['macro'][kind]['selected_projection_iou'] = sum(row[kind]['selected']['miou'] for row in results) / len(results)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
