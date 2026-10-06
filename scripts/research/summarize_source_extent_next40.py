"""Summarize completed, fixed-cohort method controls against next38."""
import json
from pathlib import Path


def mean(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def text_results(base, calibrated=False):
    suffix = 'pipeline_calibrated' if calibrated else 'pipeline'
    result = {}
    for path in sorted(Path(base).glob(f'*/anchored/{suffix}/evaluation.json')):
        data = json.loads(path.read_text())
        m = data['metrics']['hierarchical_pipeline']
        result[data['key']] = (dict(iou=m['iou'], ap=m['ap'], selected_iou=m['selected']['miou'])
                               if data['key'].startswith('lerf_') else
                               {f'iou_{split}': m[split]['miou'] for split in ['19', '15', '10']})
    return dict(scenes=result, macro={family: mean([v for k, v in result.items() if k.startswith(family)])
                                    for family in ['lerf_', 'scannet_'] if any(k.startswith(family) for k in result)})


def prompt_results(base):
    rows = {}
    for path in sorted(Path(base).glob('nvos_*/evaluation_*/evaluation.json')):
        data = json.loads(path.read_text())
        m = data['metrics']['trained']
        rows[data['scene']] = dict(iou=m['iou'], ap=m['average_precision'])
    return dict(scenes=rows, macro=mean(list(rows.values())))


def image_results(base):
    rows = {}
    for path in sorted(Path(base).glob('lerf_*/reference_image_evaluation.json')):
        data = json.loads(path.read_text())
        rows[data['key']] = data['macro']['trained']
    return dict(scenes=rows, macro=mean(list(rows.values())))


if __name__ == '__main__':
    parent = 'output/taskwise_scope_composition_next38'
    result = dict(development=True, promotion=False,
                  units='fractions; scene-equal means; fixed existing scorers',
                  selected_contract='selected Gaussians rendered against 2D masks, not physical 3D GT',
                  parent=text_results(parent, calibrated=True),
                  odds_kernel=text_results('output/odds_kernel_extent_next40'),
                  odds_all_positive=text_results('output/odds_all_positive_kernel_extent_next40'),
                  prompt_parent=prompt_results(parent),
                  images_parent=image_results(parent),
                  images_kernel=image_results('output/odds_kernel_extent_next40'))
    for name, directory in [('reference_scopes', 'reference_scopes'), ('cut_local', '1.0'),
                            ('cut_retained', 'retained_learned_1.0'), ('cut_scopes', 'retained_scopes_1.0')]:
        result[name] = prompt_results('output/seeded_extent_cut_next40/' + directory)
    out = Path('output/source_extent_methods_next40')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['macro'] for k, v in result.items() if isinstance(v, dict) and 'macro' in v}, indent=2))
