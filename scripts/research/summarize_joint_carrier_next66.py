"""Summarize the completed reconstruction and selected-output controls."""
import json
from pathlib import Path
from statistics import mean

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
ROOT = Path('output/joint_semantic_next66')

def metrics(base):
    scenes = {}
    for scene in SCENES:
        path = base / ('lerf_' + scene) / 'anchored/pipeline_calibrated/evaluation.json'
        if not path.exists():
            continue
        m = json.loads(path.read_text())['metrics']['hierarchical_pipeline']
        scenes[scene] = {**{k: m[k] for k in ['iou', 'ap', 'localization']},
                         'selected': m['selected']['miou']}
    return dict(scenes=scenes, scene_count=len(scenes),
                macro={k: mean(v[k] for v in scenes.values()) for k in
                       ['iou', 'ap', 'localization', 'selected']} if len(scenes) == 4 else None)

controls = {'retained_next63': metrics(Path('output/text_video_next63/decision'))}
geometry = {}
for arm in ['fixed', 'adaptive', 'adaptive_allrgb']:
    geometry[arm] = {}
    for scene in SCENES:
        fit = json.loads((Path('output/joint_carrier_next66') / arm / scene / 'fit.json').read_text())
        geometry[arm][scene] = {k: fit[k] for k in ['initial_gaussians', 'final_gaussians', 'steps', 'seconds', 'before', 'after', 'source_frames', 'validation_frames']}
        geometry[arm][scene]['original_depth_training_frames'] = fit['original_depth_training_frames'] if arm == 'adaptive_allrgb' else fit['source_frames']
        geometry[arm][scene]['expanded_rgb_budget'] = arm == 'adaptive_allrgb'
    for mode in ['decision', 'decision_transfer', 'decision_exclusion', 'decision_output', 'decision_continuation']:
        if (ROOT / arm / mode).exists():
            controls[arm + '/' + mode] = metrics(ROOT / arm / mode)
result = dict(controls=controls, geometry=geometry, development_only=True,
              default_changed=False, selected_metric='2D projection of selected-only Gaussian subset; not 3D GT IoU',
              expanded_rgb_is_different_input_budget=True, new_other_benchmark_results=False)
(ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({a: v['macro'] for a, v in controls.items()}, indent=2))
