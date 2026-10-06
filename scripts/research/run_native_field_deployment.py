"""Full-scene deployment comparison in unchanged native encoder coordinates."""
import json
import argparse
import math
import os
from pathlib import Path
import subprocess

import torch

from radio_gs.pipeline.native_field import compress_field, run_map_query, score_field

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('--scenes', nargs='+', default=['ramen', 'figurines', 'teatime', 'waldo_kitchen'])
parser.add_argument('--output', default='output/system_native_field')
parser.add_argument('--rebuilt-maps', help='Complete-map cohort root, including its fixed source-evidence outputs')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
root = Path(args.output)
scenes = args.scenes
rows = []
for scene in scenes:
    folder = root / scene
    spec = str(Path(args.rebuilt_maps) / scene / 'complete/scene.json') if args.rebuilt_maps else f'configs/consolidated/{scene}.scene.json'
    queries = f'configs/consolidated/{scene}.integration-queries.json' if scene == 'ramen' else f'configs/consolidated/{scene}.queries.json'
    report = run_map_query(spec, queries, folder / 'float/map')
    compression = compress_field(spec, folder / 'int8')
    baseline = torch.load(folder / 'float/map/response.pt', map_location='cpu', weights_only=False)
    field = torch.load(folder / 'int8/native_field.pt', map_location='cpu', weights_only=False)
    cosine = score_field(field, baseline['encodings'], 'cuda')
    probability = (math.exp(field['log_scale']) * cosine + field['bias']).sigmoid()
    probability *= field['supported'].reshape(-1, 1)
    dest = folder / 'int8/map'
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(dict(baseline, cosine=cosine, semantic_probability=probability), dest / 'response.pt')
    known = baseline['observed'].bool()
    delta = (cosine[known] - baseline['cosine'][known]).abs()
    row = dict(scene=scene, compression=compression, maximum_cosine_error=float(delta.max()),
               mean_cosine_error=float(delta.mean()),
               text_element_decision_agreement=float(((probability[known] > .5) ==
                                                     (baseline['semantic_probability'][known] > .5)).float().mean()),
               same_query_encoding=True, map_only={})
    commands = []
    for storage in ['float', 'int8']:
        output = folder / storage / 'map/evaluation'
        command = [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', spec,
                   '--readout', str(folder / storage / 'map/response.pt'), '--output', str(output), '--direct-map']
        commands.append(command)
        with (folder / storage / 'map/evaluation.log').open('w') as log:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
        row['map_only'][storage] = json.loads((output / 'evaluation.json').read_text())['metrics']['hierarchical_pipeline']
    admission_path = Path(args.rebuilt_maps) / scene / 'source_query/text/identity_evidence/inputs.json' if args.rebuilt_maps else Path(f'output/conditioned_identity/full_source/{scene}/identity_evidence/inputs.json')
    admitted = json.loads(admission_path.read_text())
    output = folder / 'int8/source'
    command = [PY, '-m', 'radio_gs.pipeline', '--scene', str(folder / 'int8/scene.json'),
               '--queries', queries, '--access', 'source-assisted', '--native-response', str(dest / 'response.pt'),
               '--source-evidence', admitted['detections'], '--provider-admitted', '--gpu', args.gpu, '--output', str(output)]
    commands.append(command)
    with (folder / 'int8/source.log').open('w') as log:
        subprocess.run(command, env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [PY, 'scripts/research/evaluate_consolidated_query.py', '--scene', spec,
                   '--readout', str(output / 'typed_readout.pt'), '--output', str(output / 'evaluation')]
        commands.append(command)
        subprocess.run(command, env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
    row['source_int8'] = json.loads((output / 'evaluation/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
    float_evaluation = Path(args.rebuilt_maps) / scene / 'source_evaluation/evaluation.json' if args.rebuilt_maps else Path(f'output/consolidated_response_control/{scene}/fusion/native_protected/evaluation/evaluation.json')
    row['source_float'] = json.loads(float_evaluation.read_text())['metrics']['hierarchical_pipeline']
    (folder / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    (folder / 'comparison.json').write_text(json.dumps(row, indent=2) + '\n')
    rows.append(row)
    print(scene, 'complete', flush=True)
summary = dict(scenes=rows, encoder_coordinates_preserved=True, field_storage_only=True,
               native_query_dimension_reduced=False, independent_test=False,
               macro={kind: {metric: sum(row[kind][metric] for row in rows) / len(rows)
                             for metric in ['iou', 'ap', 'localization']}
                      for kind in ['source_float', 'source_int8']})
for kind in ['source_float', 'source_int8']:
    summary['macro'][kind]['selected_projection_iou'] = sum(row[kind]['selected']['miou'] for row in rows) / len(rows)
(root / ('summary.json' if len(scenes) == 4 else 'summary_' + '_'.join(scenes) + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['macro'], indent=2))
