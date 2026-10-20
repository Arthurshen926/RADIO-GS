"""Independent compact-receipt checks for the N04 input-access experiment."""
import hashlib
import json
import math
from statistics import mean

from .reference_study import read, sha, require, check_binary_metrics

ARMS = ['v10', 'render_box', 'rgb_box', 'render_mask', 'rgb_mask', 'render_mask_lift', 'rgb_mask_lift']
PARTIAL_ARMS = ['v10','signed_core','positive_core','sparse_core','signed_core_lift','positive_core_lift','sparse_core_lift']
UNBOXED_ARMS = ['v10','boxed_sparse','unboxed_sparse','unboxed_signed','boxed_sparse_lift','unboxed_sparse_lift','unboxed_signed_lift']
MEMORY_ARMS = ['v10','static_image','pair_memory','sequence_memory','static_image_lift','pair_memory_lift','sequence_memory_lift']


def check_image_response_study(repo, study, item, jobs):
    memory = item['id']=='N07'
    unboxed = item['id']=='N06'
    partial = item['id'] in ('N05','N06')
    observed_only = partial or memory
    arms_expected = MEMORY_ARMS if memory else UNBOXED_ARMS if unboxed else PARTIAL_ARMS if partial else ARMS
    primary = 'sequence_memory_lift' if memory else 'unboxed_sparse_lift' if unboxed else 'positive_core_lift' if partial else 'rgb_mask_lift'
    lifted = {a for a in arms_expected if a.endswith('_lift')}
    root = study / item['run']
    if not (root / 'summary.json').exists():
        return dict(**item, complete=False, status='failed' if (root/'failure.json').exists()
                    else 'in_progress' if root.exists() else 'planned')
    manifest, summary, seal, snapshot = [read(root / p) for p in
        ('manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json')]
    config = manifest['config']
    nvos = [j for j in jobs if j['benchmark'] == 'nvos']
    names = {j['name'] for j in nvos}
    require(len(names) == 8 and manifest['jobs'] == nvos and config['arms'] == arms_expected, 'image response cohort/arms changed')
    require(config['primary_arm'] == summary['primary_arm'] == primary
            and config['threshold'] == seal['threshold'] == .5 and config['solver_steps'] == 120
            and config['source_hw'] == [384, 512], 'N04 design changed')
    config_path='configs/research/nvos_memory_response.json' if memory else 'configs/research/nvos_unboxed_image_response.json' if unboxed else 'configs/research/nvos_partial_image_response.json' if partial else 'configs/research/nvos_image_response.json'
    require(config == json.loads(snapshot[config_path]['text']), 'image response archived recipe differs')
    require(summary['complete'] and not summary['target_gt_used_for_prediction']
            and not summary['default_changed'] and not summary['independent_test']
            and not summary['per_task_winners'] and summary['durable_dense_prediction_bytes'] == 0, 'N04 scope changed')
    require(seal['jobs'] == 8 and seal['images'] == 56 and not seal['target_gt_read']
            and set(seal['predictions']) == set(seal['report_hashes']) == set(summary['rows']) == names,
            'N04 incomplete prediction seal')
    require(summary['prediction_seal_sha256'] == sha(root/'predictions_complete.json'), 'N04 seal changed')
    if observed_only:
        require(config['target_rgb_used'] and summary['target_rgb_used'] and not summary['average_precision_measured']
                and (memory or config['point_budget']==8) and config['metrics']==['iou','precision','recall','accuracy'],
                'N05 input, point budget or metric contract changed')
    else:
        require(config['target_rgb_arms'] == summary['target_rgb_arms'] == ['rgb_box', 'rgb_mask', 'rgb_mask_lift'],
                'N04 input access changed')
    receipts = {str(root/p): sha(root/p) for p in ('manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json')}
    for path, digest in manifest['source_hashes'].items():
        require(hashlib.sha256(snapshot[path]['text'].encode()).hexdigest() == digest == snapshot[path]['sha256'],
                'N04 source snapshot changed')
    inputs = dict(summary['gt_hashes'])
    inputs[config['image_checkpoint']] = manifest['checkpoint_sha256']
    source_checks, implementation_checks = {}, {}
    ref_path = repo / read(repo/'configs/research/tmm_response.json')['reference_summary']
    reference = {r['job']['name']: r['metrics'] for r in read(ref_path)['rows'] if r['job']['benchmark']=='nvos'}
    receipts[str(ref_path)] = sha(ref_path)
    for name, digest in seal['report_hashes'].items():
        p = root / (name + '_prediction.json')
        require(sha(p) == digest, 'N04 prediction report changed')
        report = read(p)
        require(report['job'] in nvos and report['job']['name'] == name and not report['target_gt_read'], 'prediction authority changed')
        require(report['target_rgb_used'] if observed_only else report['target_rgb_arms']==config['target_rgb_arms'], 'image access changed')
        require(report['predictions'] == seal['predictions'][name] and set(report['predictions']) == set(arms_expected),
                'N04 prediction subset changed')
        decoder_sources=report['decoder']['source_hashes'] if observed_only else report['decoder_source_hashes']
        require(decoder_sources and all(manifest['source_hashes'][p] == h
                for p,h in decoder_sources.items()), 'runtime decoder source differs from archive')
        if partial:
            prompt=report['decoder']['common_prompt']
            require(prompt is None or (1<=len(prompt['point_labels'])<=8 and
                    prompt['point_labels']==[1]*len(prompt['point_coords'])), 'N05 inferred identity prompt changed')
        if memory:
            from radio_gs.querying.reference_memory import reference_memory_paths
            job=report['job'];scene=read(repo/job['scene']);query=read(repo/job['queries']);spec=read(repo/job['evaluation'])
            paths=reference_memory_paths(scene['tracking_frames'],query['reference_frame'],spec['target_frame'])
            require(report['memory_paths']==paths and not query['reference_mask_is_gt']
                    and report['decoder']['model_eval'] and report['decoder']['memory_selection']
                    and report['decoder']['matched_rgb_preprocessing']
                    and not report['decoder']['target_gt_available'], 'N07 reference memory contract changed')
            for arm,order in paths.items():
                d=report['decoder']['decoder'][arm]
                require(d['order']==order and len(d['object_logits'])==len(order)
                        and all(math.isfinite(x) for x in d['object_logits']), 'N07 path inference incomplete')
            if paths['pair_memory']==paths['sequence_memory']:
                require(report['predictions']['pair_memory']['sha256']==report['predictions']['sequence_memory']['sha256'],
                        'N07 identical memory paths differ')
        c = report['implementation_checks']
        require(c['v10_binary_exact'] and (observed_only or c['rgb_geometry_domain_matches']) and
                all(0 <= c[k] < 1e-4 for k in ('v10_render_max_error','compositor_alpha_max_error','normalized_operator_max_error')),
                'N04 rendering implementation check failed')
        require(set(report['lifting']) == lifted, '3D branches missing')
        for value in report['lifting'].values():
            require(value['unsupported_changed'] == 0 and value['source_image_loss_after'] <= value['source_image_loss_before']+1e-5,
                    'N04 lift support/loss check failed')
        implementation_checks[name] = c
        source_checks[name] = report['source_teacher_checks']
        require(source_checks[name] and all(set(r['confidence_weighted_mean_huber']) == {'v10'}|lifted
                for r in source_checks[name]), 'N04 missing source checks')
        for path, value in report['input_hashes'].items():
            require(path not in inputs or inputs[path] == value, 'N04 shared inputs differ')
            inputs[path] = value
        receipts[str(p)] = digest
        p = root / (name + '_evaluation.json')
        row = summary['rows'][name]
        require(read(p) == row and set(row) == set(arms_expected), 'evaluation subset/receipt changed')
        receipts[str(p)] = sha(p)
        for arm, tracks in row.items():
            require(set(tracks) == set(config['gt_tracks']), 'N04 GT tracks changed')
            for track, value in tracks.items():
                check_binary_metrics(value)
                require(type(value['tn']) is int and value['tn'] >= 0 and value['independent_confusion_exact'],
                        'independent evaluator check failed')
                if not observed_only:
                    require(value['ap_implementation_difference'] < 2e-6 and math.isfinite(value['average_precision'])
                            and 0 <= value['average_precision'] <= 1, 'N04 AP invalid')
                count = sum(value[k] for k in ('tn','tp','fn','fp'))
                require(abs(value['accuracy'] - (value['tp']+value['tn'])/count) < 1e-12, 'N04 accuracy/count mismatch')
                old = row['v10'][track]
                require(value['tp']+value['fn'] == old['tp']+old['fn'] and value['tn']+value['fp'] == old['tn']+old['fp'],
                        'N04 GT pixels changed across arms')
        require(abs(row['v10']['retained_gt_gt127']['iou'] - reference[name]['iou']) < 1e-7, 'N04 v10 replay failed')
    require(set(summary['macro']) == set(config['gt_tracks']), 'missing aggregate GT track')
    for track, arms in summary['macro'].items():
        require(track in config['gt_tracks'] and set(arms) == set(arms_expected), 'aggregate subset changed')
        for arm, metrics in arms.items():
            for metric, value in metrics.items():
                require(abs(value - mean(r[arm][track][metric] for r in summary['rows'].values())) < 1e-12,
                        'N04 aggregate differs from tasks')
    require(set(summary['other_benchmarks']) == {j['benchmark']+'/'+j['name'] for j in jobs if j['benchmark']!='nvos'}
            and not summary['other_benchmarks_freshly_evaluated'], 'N04 other benchmark coverage changed')
    for value in summary['other_benchmarks'].values():
        require(not value['intervention'], 'N04 other benchmark falsely marked fresh')
        inputs[value['path']] = value['sha256']
    for path, digest in inputs.items():
        require(sha(path) == digest, 'N04 input changed: '+path)
        receipts[path] = digest
    return dict(**item, complete=True, status='completed_not_promoted', fresh_prediction_jobs=8,
                images=56, macro=summary['macro'], rows=summary['rows'], receipts=receipts,
                implementation_checks=implementation_checks, source_teacher_checks=source_checks,
                primary_arm=primary, target_rgb_arms=arms_expected[1:] if observed_only else config['target_rgb_arms'],
                independent_test=False, other_jobs_fresh_inference=False, default_changed=False)
