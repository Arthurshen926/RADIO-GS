"""Posthoc mask replay and functional attribution for the privileged reference."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from scripts.research.run_query_field_canonical_subset_reference import ROOT, PARENT, ARMS, CONTROLS, design, initializer_from_state
from scripts.research.run_query_field_relations import read, write, receipt, check
from scripts.research.audit_query_field_extension_quality import compare_masks
from scripts.research.summarize_query_field_relations import paired


def main():
    d = design(inputs=False); cfg = read(d['configuration']['path']); summary = read(ROOT / 'summary.json')
    for r in read(ROOT / 'predictions_complete.json')['records']: check(r)
    comparisons = [('admitted_joint', 'seed_addition'), ('raw_joint', 'seed_addition'),
                   ('raw_joint', 'raw_fusion'), ('admitted_joint', 'admitted_fusion'),
                   ('admitted_joint', 'raw_joint'), ('admitted_fusion', 'raw_fusion')]
    scenes = {}
    for j in cfg['jobs']:
        out = ROOT / 'predictions' / j['name']; record = read(out / 'prediction.json')
        check(dict(path=str(out / 'predictions.pt'), sha256=record['predictions_sha256']))
        data = torch.load(out / 'predictions.pt', weights_only=True)
        previous = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
        assert record['source_access'] and record['transient_query_fitting'] and not record['persistent_field_updated']
        assert not record['target_labels_opened'] and record['canonical_subset_used']
        for k, v in data['geometry'].items(): assert torch.equal(v, previous['geometry'][k])
        assert torch.equal(data['observed'], previous['observed'])
        binding = read(ROOT / 'inputs' / j['scene'] / 'binding.json')
        source = read(ROOT / 'inputs' / j['scene'] / 'scene.json')
        camera_authority = MogeGeometryPrior(source['camera_report'], source['camera_report_sha256'])
        assert camera_authority.scale == binding['source_world_units_per_metre']
        for frame in binding['source_frames']:
            raw = torch.load(Path(j['observations']) / f'view_{frame:05d}.pt', weights_only=True)
            camera = camera_authority.cameras[frame]
            assert torch.equal(raw['world_to_camera'], camera.world_to_camera)
            assert torch.equal(raw['intrinsic'], camera.intrinsic)
            assert raw['original_hw'] == [camera.height, camera.width]
            del raw
        for a in ARMS:
            assert torch.equal(data['selected_by_control'][a], data['membership_by_control'][a] > .5)
            assert torch.equal(data['scores'][a], previous['scores']['semantic_initial'])
            assert not bool(data['selected_by_control'][a][~data['observed']].any())
        for a in CONTROLS:
            assert torch.equal(data['membership_by_control'][a], previous['membership_by_control'][a])
        functional = {}
        for a in ['raw', 'admitted']:
            info = record['diagnostics'][a]; check(info['fit']); check(info['state'])
            fit = read(info['fit']['path']); state = torch.load(info['state']['path'], weights_only=True)
            initial = initializer_from_state(data['membership_by_control']['semantic_initial'], state, fit['history'])
            assert torch.equal(initial, data['membership_by_control'][a + '_fusion'])
            assert fit['steps'] == 240 and fit['pixel_loss'] == 'balanced'
            assert fit['support_prior'] == 'footprint' and fit['support_prior_weight'] == .1
            assert fit['scope_initialization'] == 'source' and fit['source_raster'] == [384, 512]
            assert fit['sampled_raster'] == [192, 256] and fit['interleaved_phases'] == 4
            assert not fit['explicit_tracks'] or all(x is None for x in fit['explicit_tracks'])
            assert not fit['implicit_legacy_query_artifacts_read']
            normalization_errors = []
            for s in state.values():
                assert len(s['view_indices']) == len(s['confidence']) == len(s['negative'])
                assert all(0 < c <= 1 for c in s['confidence'])
                assert all(0 <= i < 18 for i in s['view_indices'])
                assert not bool(s['support_cost'][~data['observed']].any())
                error = abs(float(torch.dot(s['support_cost'], s['start_membership'])) - 1.)
                assert error <= 2e-5
                normalization_errors.append(error)
            functional[a] = dict(active_queries=len(state), active_query_views=sum(r['views'] for r in fit['history']),
                fit_seconds=fit['seconds'], initializers_exact=True,
                source_footprint_normalization_max_error=max(normalization_errors, default=0.),
                added_gaussian_occurrences=int((data['selected_by_control'][a+'_joint'] & ~data['selected_by_control'][a+'_fusion']).sum()),
                removed_gaussian_occurrences=int((data['selected_by_control'][a+'_fusion'] & ~data['selected_by_control'][a+'_joint']).sum()))
        admission = read(out / 'identity_evidence/admission.json')
        partition_checks = 0
        for frame in read(ROOT / 'inputs' / j['scene'] / 'binding.json')['source_frames']:
            raw = torch.load(Path(j['teacher_detections']) / f'identity_{frame}.pt', weights_only=True)
            admitted = torch.load(out / 'identity_evidence' / f'identity_{frame}.pt', weights_only=True)
            assert raw['queries'] == admitted['queries'] == data['texts']
            assert raw['frame'] == admitted['frame'] == frame
            for original, retained in zip(raw['instances'], admitted['instances']):
                def mask_ids(masks):
                    return Counter(hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in masks)
                unknown = retained.get('unknown_masks', retained['masks'][:0])
                assert mask_ids(original['masks']) == mask_ids(retained['masks']) + mask_ids(unknown)
                assert len(retained['scores']) == len(retained['masks'])
                partition_checks += 1
        e = read(out / 'evaluation/evaluation.json'); objects = {}; masks = {}; metrics = {}
        for a in ARMS:
            rows = e['metrics'][a]['selected']['objects']; objects[a] = {(r['frame'], r['query']): r for r in rows}
            assert len(objects[a]) == len(rows)
            masks[a] = {}; ious = []; precision = []; recall = []
            for key, r in objects[a].items():
                assert not r['missing']
                gt = load_binary_mask(Path(r['gt_path']), threshold=10)
                pred = load_binary_mask(Path(r['pred_path']), threshold=10, grayscale=True)
                tp = int((gt & pred).sum()); iou = tp / max(int((gt | pred).sum()), 1)
                assert abs(iou - r['iou']) < 1e-12
                ious.append(iou); precision.append(tp / max(int(pred.sum()), 1)); recall.append(tp / max(int(gt.sum()), 1))
                masks[a][key] = pred
            assert abs(float(np.mean(ious)) - summary['per_scene'][a][j['scene']]) < 1e-12
            metrics[a] = dict(iou=float(np.mean(ious)), precision=float(np.mean(precision)), recall=float(np.mean(recall)))
        changes = {}
        for a, b in comparisons:
            assert objects[a].keys() == objects[b].keys()
            rows = []
            for key, r in objects[b].items():
                assert objects[a][key]['gt_path'] == r['gt_path']
                gt = load_binary_mask(Path(r['gt_path']), threshold=10)
                rows.append(dict(frame=key[0], query=key[1], **compare_masks(masks[b][key], masks[a][key], gt)))
            totals = {k: sum(r[k] for r in rows) for k in ['added_tp', 'added_fp', 'removed_tp', 'removed_fp']}
            changes[a + '_vs_' + b] = dict(**totals, beneficial_cases=sum(r['beneficial'] for r in rows),
                harmful_cases=sum(r['harmful'] for r in rows), rows=rows)
        scenes[j['scene']] = dict(metrics=metrics, changes=changes, functions=functional,
            original_detections=sum(r['original'] for r in admission['queries']),
            admitted_detections=sum(r['admitted'] for r in admission['queries']),
            rejected_unknown=sum(r['unknown'] for r in admission['queries']),
            exact_retained_unknown_mask_partitions=partition_checks,
            source_cameras_exact=len(binding['source_frames']), input_binding=binding)
        print('audited', j['scene'], flush=True)
    macro = {a: {k: sum(r['metrics'][a][k] for r in scenes.values()) / 4 for k in ['iou', 'precision', 'recall']} for a in ARMS}
    deltas = {a + '_vs_' + b: paired({s: r['metrics'][b]['iou'] for s, r in scenes.items()},
        {s: r['metrics'][a]['iou'] for s, r in scenes.items()}, 'actual selected-set IoU') for a, b in comparisons}
    report = dict(passed=True, macro=macro, scenes=scenes, comparisons=deltas, design=receipt(ROOT / 'design.json'),
        code=receipt(__file__), all_ious_replayed=True, source_assisted=True, posthoc_diagnosis=True,
        canonical_function_reference_not_source_free_deployment=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'audit.json', report); write('docs/experiments/2026-10-11-canonical-subset-reference-audit.json', report)
    print(macro, flush=True); print(deltas, flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
