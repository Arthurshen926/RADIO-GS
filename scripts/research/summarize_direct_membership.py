"""Report matched observation-domain controls and consumed four-query sets."""
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.uqis.metrics import scene_clustered_bootstrap
from scripts.research.fit_real_direct_membership import ROOT, write


def read(path):return json.loads(path.read_text())


def main():
    systems = {'native': Path('output/identity_bottleneck_20261005/baseline_system'),
        **{c: ROOT / c / 'isolated_system' for c in ['rendered', 'real']}}
    results = {c: read(p / 'uqis/identity_scope/result.json') for c, p in systems.items()}
    fits = {c: read(ROOT / c / 'fit.json') for c in ['rendered', 'real']}
    report = dict(default_changed=False, independent_final_test=False, declaration='configs/recipes/real_direct_membership.json',
        additional_supervision='official3D instance labels,22 disjoint rooms; same anchor/sample budget',
        development_gate=read(ROOT / 'development_gate.json'), fits={}, conditions={}, matched_comparisons={})
    for c, fit in fits.items():
        report['fits'][c] = {k: v for k, v in fit.items() if k not in ['calibration', 'validation']}
        report['fits'][c]['development'] = [{k: v for k, v in r.items() if k != 'rows'} for r in fit['validation']]
        report['fits'][c]['checkpoint_mib'] = (ROOT / c / 'model.joblib').stat().st_size / 2**20
    evaluator = Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    dataset = read(evaluator / 'targets.json')
    labels = {d['scene_id']: np.load(evaluator / d['mesh_instance_ids_path']) for d in dataset['scene_domains']}
    target_lookup = {q: t for t in dataset['targets'] for q in t['queries'].values()}
    for condition, result in results.items():
        metrics = {};unchanged = {};changes = {}
        for modality in ['text', 'image', 'point_2d', 'point_3d']:
            selected = result['explicit_selection']['modalities'][modality]
            original = results['native']['explicit_selection']['modalities'][modality]
            metrics[modality] = dict(ap=result['modalities'][modality]['scene_macro']['average_precision'],
                **selected['scene_macro'], per_scene=selected['per_scene'],
                iou_delta_pp=scene_clustered_bootstrap({s: 100 * (r['iou'] - original['per_scene'][s]['iou'])
                    for s, r in selected['per_scene'].items()}))
            queries = [t['queries'][modality] for t in dataset['targets']]
            unchanged[modality] = dict(selection_changes=sum(not np.array_equal(
                np.load(systems[condition] / 'uqis/identity_scope/selections' / (q + '.npy')),
                np.load(systems['native'] / 'uqis/identity_scope/selections' / (q + '.npy'))) for q in queries),
                native_max_difference=max(float(np.max(np.abs(
                    np.load(systems[condition] / 'uqis/native_semantic/predictions' / (q + '.npy')) -
                    np.load(systems['native'] / 'uqis/native_semantic/predictions' / (q + '.npy'))))) for q in queries),
                rank_max_difference=max(float(np.max(np.abs(
                    np.load(systems[condition] / 'uqis/identity_scope/predictions' / (q + '.npy')) -
                    np.load(systems['native'] / 'uqis/identity_scope/predictions' / (q + '.npy'))))) for q in queries))
            if modality.startswith('point_'):
                edits = []
                for q in queries:
                    target = target_lookup[q];truth = labels[target['scene_id']] == target['instance_id']
                    old = np.load(systems['native'] / 'uqis/identity_scope/selections' / (q + '.npy'))
                    new = np.load(systems[condition] / 'uqis/identity_scope/selections' / (q + '.npy'))
                    added = new & ~old;removed = old & ~new
                    values, counts = np.unique(labels[target['scene_id']][new], return_counts=True)
                    edits.append(dict(scene=target['scene_id'], query=q, added=int(added.sum()),
                        added_true=int((added & truth).sum()), removed=int(removed.sum()), removed_true=int((removed & truth).sum()),
                        correct_dominant=bool(len(values) and values[counts.argmax()] == target['instance_id'])))
                changes[modality] = dict(rows=edits, correct_dominant=sum(r['correct_dominant'] for r in edits),
                    **{k: sum(r[k] for r in edits) for k in ['added', 'added_true', 'removed', 'removed_true']})
        report['conditions'][condition] = dict(metrics=metrics, controls=unchanged, point_edits=changes,
            all_four_success=sum(t['all_four_iou_at_least_050'] for t in result['explicit_selection']['paired_query_robustness']['per_target']))
    for candidate, control in [('real', 'rendered'), ('real', 'native'), ('rendered', 'native')]:
        report['matched_comparisons'][candidate + ':' + control] = {}
        for modality in ['point_2d', 'point_3d']:
            a = results[candidate]['explicit_selection']['modalities'][modality]['per_scene']
            b = results[control]['explicit_selection']['modalities'][modality]['per_scene']
            delta = {s: 100 * (a[s]['iou'] - b[s]['iou']) for s in a}
            report['matched_comparisons'][candidate + ':' + control][modality] = dict(
                delta_pp=scene_clustered_bootstrap(delta), per_scene_delta_pp=delta,
                improved_scenes=sum(v > 1e-10 for v in delta.values()), degraded_scenes=sum(v < -1e-10 for v in delta.values()))
    original = {c: read(ROOT / c / 'system/summary.json') for c in ['real', 'rendered']}
    corrected = {c: read(ROOT / c / 'isolated_system/summary.json') for c in ['real', 'rendered']}
    report['fresh_scene_jobs'] = sum(r['completed'] for r in [*original.values(), *corrected.values()])
    report['valid_scene_jobs'] = 40
    report['uqis_query_executions'] = 4 * len(results['real']['explicit_selection']['per_query'])
    report['retained_uqis_query_executions'] = 2 * len(results['real']['explicit_selection']['per_query'])
    report['native_array_repair'] = dict(invalidated_uqis_jobs=18,
        reason='auxiliary reader renormalized a shared native array in place; observed5.96e-8 native score perturbation',
        fix='copy auxiliary descriptors before normalizing; no changes to fitted model or budgets',
        regenerated='all9 rooms in each condition, all four queries and readouts',
        public='same-phase22 public predictions do not call this reader and remain valid; no score-based selection')
    public = {k: v for k, v in original['real']['macro'].items() if k != 'uqis'}
    report['public_replay'] = public
    final = dict(original['real'], rows=[r for r in original['real']['rows'] if r['job']['benchmark'] != 'uqis'] + corrected['real']['rows'],
        macro={**public, 'uqis': corrected['real']['macro']['uqis']},
        uqis_readouts=corrected['real']['uqis_readouts'], uqis_selections=corrected['real']['uqis_selections'],
        completion='22 freshly recomputed public tasks plus9 fully regenerated corrected UQIS tasks; same declared recipe',
        all_predictions_before_scoring=False, each_execution_predictions_before_scoring=True,
        separate_execution_reason='isolate auxiliary native-array mutation; public readers unaffected; no score-dependent numerical revision',
        correction=report['native_array_repair'])
    recipe=read(ROOT / 'real_recipe.json')
    jobs={(j['benchmark'],j['name']):j for j in recipe['jobs']}
    for row in final['rows']:
        row['job']=jobs[row['job']['benchmark'],row['job']['name']]
    write(ROOT / 'final_validation.json', final)
    def joint_success(result):
        return {(r['scene_id'],r['instance_id']) for r in result['explicit_selection']['paired_query_robustness']['per_target']
                if r['all_four_iou_at_least_050']}
    old,new=joint_success(results['native']),joint_success(results['real'])
    report['joint_transitions']=dict(stable=len(old&new),lost=[list(x) for x in sorted(old-new)],gained=[list(x) for x in sorted(new-old)])
    selected=report['conditions']['real'];checks={}
    for modality in ['point_2d','point_3d']:
        delta=selected['metrics'][modality]['iou_delta_pp']
        checks[modality]=delta['estimate']>=2 and delta['lower']>0
    checks['native_exact']=all(r['native_max_difference']==0 for r in selected['controls'].values())
    checks['text_image_exact']=all(selected['controls'][m]['selection_changes']==0 and
        selected['controls'][m]['rank_max_difference']==0 for m in ['text','image'])
    checks['joint_count_preserved']=selected['all_four_success']>=7
    checks['public_retained']=(abs(public['lerf']['semantic_iou']-.7040067569509945)<1e-10 and
        abs(public['lerf']['selected_projection_iou']-.6696765063684671)<.0001 and
        abs(public['nvos']['iou']-.865134172141552)<1e-10 and abs(public['scannet']['miou_19']-.6101149054806628)<1e-10)
    report['promotion_gate']=dict(checks=checks,passed=all(checks.values()),declaration='configs/recipes/real_direct_membership.json')
    report['default_changed']=read(Path('configs/pipeline/current.json'))['active_recipe']=='configs/pipeline/system_v9.json'
    if report['default_changed']:
        report['promoted_recipe']='configs/pipeline/system_v9.json'
    write(ROOT / 'comparison.json', report)
    print(json.dumps({c: dict(metrics={m: {k: v for k, v in r.items() if k != 'per_scene'}
        for m, r in d['metrics'].items()}, joint=d['all_four_success']) for c, d in report['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
