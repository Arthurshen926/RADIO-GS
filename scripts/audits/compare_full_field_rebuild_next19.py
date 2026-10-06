"""Post-hoc paired diagnostics; never chooses parameters, fields or query routes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')
ARMS = {'native_compact_calibrated': 'native_compact_calibrated',
        'final': 'final', 'structure32_graph': 'distilled_structure32_calibrated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/full_field_rebuild_next19'))
    args = parser.parse_args()
    destination = args.root / 'paired_comparison.json'
    assert not destination.exists(), destination
    bindings = {}

    def read(path):
        path = Path(path)
        bindings[str(path)] = sha256_file(path)
        return json.loads(path.read_text())

    def index(rows):
        result = {(r['frame'], r['query']): r for r in rows}
        assert len(result) == len(rows), 'duplicate observation'
        return result

    support = read('output/lerf_support_decomposition_next19/report.json')
    scenes, observations = {}, []
    for scene in SCENES:
        old = read(f'output/graph_transfer_final_20260919/{scene}/report.json')
        selected = read(f'output/graph_transfer_final_20260919/{scene}/selected/report.json')
        frozen = read(f'output/frozen_method_lerf2d_20260918/{scene}/report.json')
        new = read(args.root / scene / 'evaluation/report.json')
        assert new['complete'] and new['geometry_sha256'] == frozen['geometry_sha256']
        old_score_path = Path('output/graph_transfer_final_20260919') / scene / 'scores.pt'
        new_score_path = args.root / scene / 'evaluation/scores.pt'
        old_preparation = read(old_score_path.parent / 'preparation.json')
        packs = []
        for path, expected in [(old_score_path, old_preparation['scores_sha256']),
                               (new_score_path, new['scores_sha256'])]:
            bindings[str(path)] = sha256_file(path)
            assert bindings[str(path)] == expected
            packs.append(torch.load(path, map_location='cpu', weights_only=False))
        oldpack, newpack = packs
        assert oldpack['geometry_binding'] == newpack['geometry_binding']
        assert set(oldpack['texts']) == set(newpack['texts'])
        oldrows = index(old['observations'])
        if 'native_compact_calibrated' not in oldpack['scores']:
            supplement_root = Path('output/old_native_control_next19') / scene
            supplement = read(supplement_root / 'report.json')
            supplement_selected = read(supplement_root / 'selected/report.json')
            supplement_preparation = read(supplement_root / 'preparation.json')
            supplement_identity = read(supplement_root / 'sealed_input_identity.json')
            assert supplement_identity['verified']
            assert supplement_identity['field_sha256'] == supplement_preparation['old_field_sha256']
            assert supplement_identity['query_sha256'] == supplement_preparation['old_query_sha256']
            supplement_path = supplement_root / 'scores.pt'
            bindings[str(supplement_path)] = sha256_file(supplement_path)
            assert bindings[str(supplement_path)] == supplement_preparation['scores_sha256']
            supplement_pack = torch.load(supplement_path, map_location='cpu', weights_only=False)
            assert supplement_pack['geometry_binding'] == oldpack['geometry_binding']
            assert supplement_pack['texts'] == oldpack['texts']
            assert torch.equal(supplement_pack['observed'], oldpack['observed'])
            supplement_rows = index(supplement['observations'])
            assert supplement_rows.keys() == oldrows.keys()
            for key, row in oldrows.items():
                for metric in ('ap', 'membership_iou'):
                    row[metric]['native_compact_calibrated'] = supplement_rows[key][metric]['native_compact_calibrated']
            selected['results']['native_compact_calibrated'] = supplement_selected['results']['native_compact_calibrated']
            oldpack['scores']['native_compact_calibrated'] = supplement_pack['scores']['native_compact_calibrated']
            del supplement_pack
        supportrows = index([r for r in support['rows'] if r['scene'] == scene])
        populations = {}
        for side, fieldroot in [('before', Path('output/depth_opacity_pipeline_20260918') / scene / 'reconstruction/field'),
                                ('after', args.root / scene / 'reconstruction/field')]:
            captions = read(fieldroot / 'caption_source_phrases.json')
            calibration = read(fieldroot / 'exact_render_query_threshold96/report.json')
            admission = read(fieldroot / 'bounded_fallback_admission/report.json')
            scope_groups = {}
            for row in admission['observations']:
                key = (row['kind'], bool(row['concept_accepted']), row['selected_scope_relation'])
                scope_groups.setdefault(key, []).append(row)
            scope_by_kind = {}
            for (kind, accepted, relation), rows in sorted(scope_groups.items()):
                group = scope_by_kind.setdefault(kind, {}).setdefault(
                    'concept_accepted' if accepted else 'concept_rejected', {})
                group[relation] = {
                    'observations': len(rows),
                    'native_iou': float(np.mean([r['native_iou'] for r in rows])),
                    'extent_iou': float(np.mean([r['extent_iou'] for r in rows])),
                    'harmful': sum(r['gain'] < -1e-6 for r in rows),
                }
            populations[side] = {'caption_phrases': len(captions['phrases']),
                                 'native_threshold_supervised_queries': len(calibration['query_indices']),
                                 'threshold_query_indices': calibration['query_indices'],
                                 'selected_ridge': calibration['selected_ridge'],
                                 'native_harm_protection_certified': admission['native_harm_protection_certified'],
                                 'source_admission_diagnostics': admission['admission_diagnostics'],
                                 'source_scope_granularity': admission['granularity'],
                                 'source_scope_by_kind': scope_by_kind,
                                 'source_validation_scope': admission['validation_scope']}
        scene_result = {'source_acceptance': new['source_acceptance'], 'source_populations': populations, 'arms': {}}
        old_reference = read(Path('output/bounded_fallback_repair_20260918') / scene / 'reference_online.json')
        new_reference = read(args.root / scene / 'reference_online.json')
        def reference_inputs(value):
            return [(r['category'], r['frame'], r['image_sha256'], r['pixels']) for r in value['references']]
        assert reference_inputs(old_reference) == reference_inputs(new_reference)
        reference_runs = [r['runs']['bounded_fallback_admission'] for r in (old_reference, new_reference)]
        def reference_rows(value):
            return [(r['reference_frame'], r['target_frame'], r['category'], r['prompt_available'])
                    for r in value['observations']]
        assert reference_rows(reference_runs[0]) == reference_rows(reference_runs[1])
        assert reference_runs[0]['tasks_per_modality'] == reference_runs[1]['tasks_per_modality']
        scene_result['reference_development'] = {
            'tasks_per_modality': reference_runs[0]['tasks_per_modality'],
            'prompt_seed_rows_changed': sum(a['seedrows'] != b['seedrows']
                                            for a, b in zip(old_reference['references'], new_reference['references'])),
            'official_four_query': False,
            'before': reference_runs[0]['macro'], 'after': reference_runs[1]['macro'],
            'delta': {modality: {metric: reference_runs[1]['macro'][modality][metric] - reference_runs[0]['macro'][modality][metric]
                                for metric in ('iou', 'average_precision')}
                      for modality in ('text', 'image', 'prompt')},
        }
        for arm, oldarm in ARMS.items():
            selection_changes = {}
            for query in newpack['texts']:
                oldmask = (oldpack['scores'][oldarm][:, oldpack['texts'].index(query)] > .5) & oldpack['observed']
                newmask = (newpack['scores'][arm][:, newpack['texts'].index(query)] > .5) & newpack['observed']
                assert oldmask.shape == newmask.shape
                intersection, union = int((oldmask & newmask).sum()), int((oldmask | newmask).sum())
                selection_changes[query] = {
                    'before': int(oldmask.sum()), 'after': int(newmask.sum()),
                    'retained': intersection, 'added': int((newmask & ~oldmask).sum()),
                    'removed': int((oldmask & ~newmask).sum()),
                    'jaccard': intersection / union if union else 1.,
                }
            newrows = index(new['lerf2d'][arm]['observations'])
            oldselected = index(selected['results'][oldarm]['objects'])
            newselected = index(new['selected'][arm]['objects'])
            assert oldrows.keys() == newrows.keys() == supportrows.keys() == oldselected.keys() == newselected.keys()
            pairs = []
            for key, row in newrows.items():
                prior = oldrows[key]
                before = {'iou': prior['membership_iou'][oldarm], 'ap': prior['ap'][oldarm],
                          'selected_iou': oldselected[key]['iou'],
                          'support_recall': supportrows[key]['valid_semantic_recall']}
                after = {'iou': row['iou'], 'ap': row['ap'], 'selected_iou': newselected[key]['iou'],
                         'support_recall': row['positive_support_recall']}
                assert abs(row['alpha_nonzero_recall'] - supportrows[key]['alpha_nonzero_recall']) < 1e-6
                assert row['selected_elements'] == selection_changes[key[1]]['after']
                pair = dict(scene=scene, arm=arm, frame=key[0], query=key[1], before=before, after=after,
                            delta={k: after[k] - before[k] for k in before}, selected_elements=row['selected_elements'],
                            selection_change=selection_changes[key[1]])
                pairs.append(pair)
                observations.append(pair)
            scene_result['arms'][arm] = {
                'before': {k: float(np.mean([r['before'][k] for r in pairs])) for k in before},
                'after': {k: float(np.mean([r['after'][k] for r in pairs])) for k in after},
                'delta': {k: float(np.mean([r['delta'][k] for r in pairs])) for k in before},
                'query_gains': {k: sum(r['delta'][k] > 1e-6 for r in pairs) for k in before},
                'query_losses': {k: sum(r['delta'][k] < -1e-6 for r in pairs) for k in before},
                'probability_iou_up_selection_iou_down': sum(r['delta']['iou'] > 1e-6 and r['delta']['selected_iou'] < -1e-6 for r in pairs),
            }
        scenes[scene] = scene_result
        del packs, oldpack, newpack
    assert len(observations) == 208 * len(ARMS)
    result = dict(complete=True, diagnostic_only=True, parameter_selection=False,
                  independent_test=False, source_budget_changed=True, scenes=scenes, observations=observations,
                  macro={a: {side: {k: float(np.mean([scenes[s]['arms'][a][side][k] for s in SCENES]))
                                   for k in ('iou', 'ap', 'selected_iou', 'support_recall')}
                             for side in ('before', 'after', 'delta')} for a in ARMS},
                  reference_development_macro={side: {modality: {metric: float(np.mean([
                      scenes[s]['reference_development'][side][modality][metric] for s in SCENES]))
                      for metric in ('iou', 'average_precision')} for modality in ('text', 'image', 'prompt')}
                      for side in ('before', 'after', 'delta')},
                  interpretation='Scene-equal means. Native-grid probability and selected-alpha JPEG IoU are different contracts. Nonzero alpha is not target geometry correctness. Query-wise changes do not identify a single causal change because source count, training budget and admission cohorts changed together.',
                  source_bindings=bindings, implementation_sha256=sha256_file(__file__))
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['macro'], indent=2))


if __name__ == '__main__':
    main()
