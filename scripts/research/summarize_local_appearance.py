"""Full-cohort correspondence ablation, with actual sets and matched controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/local_appearance_study'))
    args = parser.parse_args()
    roots = {'v7': Path('output/system_v7')}
    roots.update({mode: args.root/mode/'system' for mode in ['pooled', 'chamfer', 'reciprocal']})
    results = {'v7': read(Path('output/observation_hypotheses_study/identity_diagnostics/v7.json'))}
    results.update({mode: read(root/'uqis/identity_scope/result.json')
                    for mode, root in roots.items() if mode != 'v7'})
    baseline = results['v7']
    baseline_sets = {q['query_id']: q for q in baseline['explicit_selection']['per_query']}
    summary = {}
    for name, result in results.items():
        row = dict(modalities={}, paired=result['explicit_selection']['paired_query_robustness'])
        query_records = {q['query_id']: q for q in result['per_query']}
        row['image_reference_groups'] = {}
        for label, flagged in [('review_flagged', True), ('not_flagged', False)]:
            selected_group = [q for q in result['explicit_selection']['per_query']
                if q['modality'] == 'image' and bool(query_records[q['query_id']]['reference_difficulty_flags']) == flagged]
            scenes = sorted({q['scene_id'] for q in selected_group})
            row['image_reference_groups'][label] = dict(queries=len(selected_group), scenes=len(scenes),
                ap=100*result['image_reference_difficulty_groups'][label]['scene_macro']['average_precision'],
                iou=100*float(np.mean([np.mean([q['iou'] for q in selected_group if q['scene_id'] == scene]) for scene in scenes])))
        for modality, value in result['modalities'].items():
            selected = result['explicit_selection']['modalities'][modality]
            differences = {scene: 100*(v['iou']-baseline['explicit_selection']['modalities'][modality]['per_scene'][scene]['iou'])
                           for scene, v in selected['per_scene'].items()}
            pooled_differences = {scene: 100*(v['iou']-results['pooled']['explicit_selection']['modalities'][modality]['per_scene'][scene]['iou'])
                                  for scene, v in selected['per_scene'].items()}
            row['modalities'][modality] = dict(
                ap=100*value['scene_macro']['average_precision'],
                **{key: 100*v for key, v in selected['scene_macro'].items()},
                per_scene_iou_gain_pp=differences,
                iou_gain_pp=scene_clustered_bootstrap(differences),
                iou_gain_over_pooled_pp=scene_clustered_bootstrap(pooled_differences),
                identity=result['explicit_selection']['identity_diagnosis'][modality],
                empty_selections=selected['empty_selections'])
        if name != 'v7':
            row['control_parity'] = {}
            for control in ['native_semantic', 'anchor_comembership']:
                differences = [float(np.abs(np.load(path)-np.load(roots['v7']/'uqis'/control/'predictions'/path.name)).max())
                               for path in (roots[name]/'uqis'/control/'predictions').glob('*.npy')]
                row['control_parity'][control] = dict(queries=len(differences), max_absolute_difference=max(differences))
            reports = [read(path) for path in (roots[name]/'uqis').glob('*/query/registered/runs/*.json')]
            row['cost'] = dict(query_seconds_sum=sum(r['seconds'] for r in reports),
                               map_mib=sum(r['map_bytes'] for r in reports)/2**20,
                               source_images_read_at_query=sum(int(r['source_rgb_read']) for r in reports))
            images = [q for q in result['explicit_selection']['per_query'] if q['modality'] == 'image']
            row['image_identity_changes'] = dict(
                corrected=sum(q['target_is_unique_dominant_instance'] and not baseline_sets[q['query_id']]['target_is_unique_dominant_instance'] for q in images),
                broken=sum(not q['target_is_unique_dominant_instance'] and baseline_sets[q['query_id']]['target_is_unique_dominant_instance'] for q in images))
        summary[name] = row
    controls = {}
    for readout in ['native_semantic', 'anchor_comembership', 'identity_region', 'identity_scope']:
        files = sorted((roots['pooled']/'uqis'/readout/'predictions').glob('*.npy'))
        differences = [float(np.abs(np.load(p)-np.load(Path('output/dual_native_identity_study/system/uqis')/readout/'predictions'/p.name)).max()) for p in files]
        controls[readout] = dict(queries=len(files), max_absolute_difference=max(differences))
    builds = [read(p) for p in (args.root/'maps').glob('*/build.json')]
    descriptor_diff = max(float(np.abs(np.load(p.parent/'appearance_descriptors.npy').astype(np.float32)-
        np.load(Path('output/region_appearance_study/maps')/p.parent.name/'appearance_descriptors.npy').astype(np.float32)).max())
        for p in (args.root/'maps').glob('*/build.json'))
    report = dict(scope='9 opened-development scenes; 67 targets; all 268 independent queries per candidate',
        completed_scene_jobs=27, completed_queries=804,
        results=summary, rebuilt_pooled_control_parity=controls, pooled_descriptor_max_difference=descriptor_diff,
        compile_cost=dict(**{key: sum(b[key] for b in builds) for key in ['source_frames', 'regions', 'patches', 'seconds']},
            shared_checkpoint_mib=builds[0]['shared_checkpoint_bytes']/2**20,
            shared_half_parameters_mib=builds[0]['shared_parameter_bytes']/2**20),
        all_query_selections={name: r['explicit_selection']['per_query'] for name, r in results.items()},
        original_geometry_membership_and_scope_unchanged=True, source_images_added=0,
        independent_test=False, public_benchmarks_reexecuted=False, default_changed=False)
    (args.root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: {m: {k: v[k] for k in ['ap', 'iou', 'purity', 'coverage']}
        for m, v in row['modalities'].items()} for name, row in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
