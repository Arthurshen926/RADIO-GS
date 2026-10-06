"""Use all already-authorized registered NVOS tracks in both reconstruction heads.

The retained configuration excluded two geometry-validation source frames from
fitting although they already entered tracking. Geometry is frozen at query
time; this control removes that inherited fitting exclusion without new RGB,
teacher outputs, prompt changes or target access.
"""
import json
import os
from pathlib import Path
import subprocess

PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'


def summarize(root):
    rows = [json.loads(path.read_text()) for path in sorted(root.glob('*/comparison.json'))]
    summary = dict(rows=rows, new_teacher_or_rgb_inputs=False, geometry_changed=False,
                   target_rgb_used=False, same_positive_reference_prompt=True, response_steps=120, selection_steps=240,
                   scope='all eight retained positive-reference NVOS queries; geometry-validation frames now contribute to frozen-map readout fitting',
                   mean={kind: {metric: sum(row[kind][metric] for row in rows) / len(rows)
                                for metric in ['iou', 'average_precision']}
                         for kind in ['retained', 'complete']})
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['mean'], indent=2), flush=True)


def main():
    root = Path('output/unified_system/nvos_all_sources')
    rows = []
    for path in sorted(Path('configs/consolidated/nvos').glob('*.scene.json')):
        name = path.name.removesuffix('.scene.json')
        previous = json.loads(path.read_text())['source_frames']
        scene = json.loads((Path('configs/pipeline/nvos_all_sources') / path.name).read_text())
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        spec = folder / 'scene.json'
        spec.write_text(json.dumps(scene, indent=2) + '\n')
        queries = f'configs/consolidated/nvos/{name}.query.json'
        plan = json.loads(Path(f'output/unified_system/nvos/{name}/execution.json').read_text())
        tracks = plan['groups'][0]['tracks']
        commands = [
            [PYTHON, '-m', 'radio_gs.pipeline', '--scene', str(spec), '--queries', queries,
             '--access', 'source-assisted', '--reference-tracks', tracks, '--gpu', '0', '--output', str(folder / 'query')],
            [PYTHON, 'scripts/research/evaluate_reference_instance.py', '--scene', str(spec),
             '--evaluation', f'configs/consolidated/nvos/{name}.evaluation.json',
             '--field', str(folder / 'query/typed_readout.pt'), '--typed-readout', '--output', str(folder / 'evaluation')],
        ]
        for index, command in enumerate(commands):
            with (folder / f'{index}.log').open('w') as log:
                subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='0'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        old = json.loads(Path(f'output/unified_system/nvos/{name}/evaluation/evaluation.json').read_text())['metrics']
        new = json.loads((folder / 'evaluation/evaluation.json').read_text())['metrics']
        row = dict(scene=name, previous_frames=previous, added_frames=sorted(set(scene['source_frames']) - set(previous)),
                   source_frames=scene['source_frames'], retained=old, complete=new)
        rows.append(row)
        (folder / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        (folder / 'comparison.json').write_text(json.dumps(row, indent=2) + '\n')
        print(name, old['iou'], new['iou'], flush=True)
    summarize(root)


if __name__ == '__main__':
    main()
