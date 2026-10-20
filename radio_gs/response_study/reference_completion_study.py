"""Integrity and matched-control validation for N12's complete eight-task run."""
import hashlib
import json
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics

POLICIES = ['legacy_memory', 'semantic_memory', 'constrained_memory']
ARMS = ['v10', 'static_image', 'static_native'] + sum(([a + '_image', a + '_native'] for a in POLICIES), [])


def check_reference_completion_study(repo, study, item, jobs):
    root = study / item['run']; old = repo / 'output/tmm_response_20261016/study'
    if not (root / 'summary.json').exists():
        return dict(**item, complete=False, status='failed' if (root / 'failure.json').exists() else 'in_progress' if root.exists() else 'planned')
    m, s, seal, snapshot = [read(root / p) for p in ['manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json']]
    c = m['config']; parent = old / c['parent_run']; pm, ps = [read(parent / p) for p in ['manifest.json', 'summary.json']]
    memory = read(old / 'nvos_memory_response_v2/summary.json')
    incumbent = read(old / 'nvos_instance_extent_v2/summary.json')
    expected = [j for j in jobs if j['benchmark'] == 'nvos']; names = {j['name'] for j in expected}
    require(m['jobs'] == pm['jobs'] == expected and len(names) == 8 and c['arms'] == ARMS, 'N12 cohort/arms differ')
    require(c == json.loads(snapshot['configs/research/nvos_reference_completion.json']['text']) and
            c['native_wh'] == [1008, 756] and c['source_hw'] == [384, 512] and c['solver_steps'] == 120 and
            c['points_per_sign'] == 16 and c['footprint_backend'] == 'gsplat_cuda' and c['threshold'] == seal['threshold'] == .5 and
            c['primary_arm'] == s['primary_arm'] == 'constrained_memory_native', 'N12 recipe differs')
    require(read(study / 'storage_policy.json') == c['storage_policy'] and c['storage_policy']['study_bytes'] == 8*2**20 and
            c['maximum_new_durable_bytes'] == 2**20 and c['maximum_volatile_prediction_bytes'] == 256*2**20, 'N12 budget differs')
    require(s['complete'] and not s['default_changed'] and not s['target_gt_used_for_prediction'] and not s['per_task_winners'] and
            not s['independent_test'] and s['target_rgb_used'] and s['negative_scribble_used'] and s['average_precision_measured'] and
            s['durable_dense_prediction_bytes'] == 0 and not s['other_benchmarks_freshly_evaluated'], 'N12 scope differs')
    require(seal['jobs'] == 8 and seal['images'] == 72 and not seal['target_gt_read'] and
            set(seal['report_hashes']) == set(seal['predictions']) == set(s['rows']) == set(s['transitions_diagnostic_only']) == names and
            s['prediction_seal_sha256'] == sha(root / 'predictions_complete.json'), 'N12 seal incomplete')
    require(m['parent_manifest_sha256'] == sha(parent / 'manifest.json') and s['parent_summary_sha256'] == sha(parent / 'summary.json') and
            m['inherited_source_snapshot_sha256'] == sha(parent / 'source_snapshot.json') and
            s['previous_candidate_summary_sha256'] == sha(old / 'nvos_instance_extent_v2/summary.json'), 'N12 parent differs')
    receipts = {str(p): sha(p) for p in root.iterdir() if p.is_file()}
    for run, key in [('nvos_memory_response_v2', 'inherited_memory_snapshot_sha256'), ('nvos_instance_extent_v2', 'inherited_instance_snapshot_sha256')]:
        require(m[key] == sha(old / run / 'source_snapshot.json'), 'N12 dependency archive differs')
        for file in ['summary.json', 'source_snapshot.json']:
            receipts[str(old / run / file)] = sha(old / run / file)
    inherited = read(parent / 'source_snapshot.json')
    for run in ['nvos_memory_response_v2', 'nvos_instance_extent_v2']:
        inherited.update(read(old / run / 'source_snapshot.json'))
    inherited.update(snapshot)
    for path, entry in inherited.items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'N12 archived source inconsistent')
    for path, digest in m['source_hashes'].items():
        require(snapshot[path]['sha256'] == digest, 'N12 source binding differs')
    for path, digest in m['worker_source_hashes'].items():
        require(path in inherited and inherited[path]['sha256'] == digest, 'N12 worker dependency not archived: ' + path)
    inputs = dict(s['gt_hashes']); sources, checks, decoder = {}, {}, {}
    for name in names:
        rp = root / (name + '_prediction.json'); r = read(rp); pr = read(parent / (name + '_prediction.json'))
        row = s['rows'][name]
        require(r['name'] == name and sha(rp) == seal['report_hashes'][name] and not r['target_gt_read'] and
                r['predictions'] == seal['predictions'][name] and set(r['predictions']) == set(ARMS), 'N12 prediction differs')
        require(r['parent_report_sha256'] == sha(parent / (name + '_prediction.json')) and
                r['parent_arrays'] == {a: pr['predictions'][a]['sha256'] for a in ['v10', 'static_image']}, 'N12 control replay differs')
        require(row == read(root / (name + '_evaluation.json')) and set(row) == set(ARMS), 'N12 scoring differs')
        require(r['reference_figure_sha256'] == sha(root / (name + '_reference.png')), 'N12 reference diagnostic differs')
        checks[name] = cc = r['implementation_checks']
        require(cc['v10_binary_exact'] and all(0 <= cc[k] < 1e-4 for k in ['v10_render_max_error', 'target_alpha_max_error', 'normalized_operator_max_error']), 'N12 renderer check failed')
        decoder[name] = d = r['decoder']
        require(d['source_hashes'] == m['worker_source_hashes'] and not d['target_gt_available'] and not d['reference_gt_available'] and
                d['sam_eval'] and d['dino_eval'] and d['negative_scribble_used'] and d['memory_selection'] and
                not d['target_dino_features_used'] and set(d['decoder']) == set(POLICIES), 'N12 decoder contract differs')
        require(d['reference_affinity']['source_only_scale'] and not d['reference_affinity']['reference_extent_gate'], 'N12 authority differs')
        require(d['point_labels'] == [1]*16 + [0]*16 and len(d['point_coords']) == 32, 'N12 point budget/signs differ')
        for policy in POLICIES:
            require(len(d['decoder'][policy]['object_logits']) == 2, 'N12 transport differs')
        require(d['decoder']['constrained_memory']['positive_coverage'] == 1 and d['decoder']['constrained_memory']['negative_violation'] == 0,
                'N12 source constraints violated')
        require(d['decoder']['legacy_memory']['added_reference_pixels'] == d['decoder']['legacy_memory']['removed_reference_pixels'] == 0,
                'N12 legacy reference changed')
        fields = {'static_native', *[a + '_native' for a in POLICIES]}
        require(set(r['lifting']) == fields and all(v['unsupported_changed'] == 0 and
                v['source_image_loss_after'] <= v['source_image_loss_before'] + 1e-5 for v in r['lifting'].values()), 'N12 fit check failed')
        sources[name] = r['source_teacher_checks']
        require(sources[name] and all(set(v['confidence_weighted_mean_huber']) == fields | {'v10'} for v in sources[name]), 'N12 source checks missing')
        for arm, tracks in row.items():
            require(set(tracks) == set(c['gt_tracks']), 'N12 missing track')
            for track, v in tracks.items():
                check_binary_metrics(v); size = sum(v[k] for k in ['tp', 'fp', 'fn', 'tn']); control = row['v10'][track]
                require(type(v['tn']) is int and v['tn'] >= 0 and v['independent_confusion_exact'] and 0 <= v['average_precision'] <= 1 and
                        abs(v['accuracy'] - (v['tp'] + v['tn']) / size) < 1e-12 and
                        v['tp'] + v['fn'] == control['tp'] + control['fn'] and v['tn'] + v['fp'] == control['tn'] + control['fp'], 'N12 confusion mismatch')
                if arm in ['v10', 'static_image', 'static_native']:
                    require(all(v[k] == ps['rows'][name][arm][track][k] for k in ['tp', 'fp', 'fn', 'tn']), 'N12 static counts drift')
                if arm in ['v10', 'static_image']:
                    require(v == ps['rows'][name][arm][track] and r['predictions'][arm]['sha256'] == pr['predictions'][arm]['sha256'], 'N12 exact control differs')
                if arm == 'legacy_memory_image':
                    require(all(v[k] == memory['rows'][name]['pair_memory'][track][k] for k in ['tp', 'fp', 'fn']), 'N12 old memory counts drift')
        for arm, delta in s['transitions_diagnostic_only'][name].items():
            a, b = row[arm]['retained_gt_gt127'], row['static_native']['retained_gt_gt127']
            require(delta['recovered_foreground']-delta['lost_foreground'] == a['tp']-b['tp'] and
                    delta['added_false_positive']-delta['removed_false_positive'] == a['fp']-b['fp'], 'N12 transitions differ')
        receipts[str(parent / (name + '_prediction.json'))] = sha(parent / (name + '_prediction.json'))
        for path, digest in pr['input_hashes'].items():
            require(path not in inputs or inputs[path] == digest, 'N12 conflicting input'); inputs[path] = digest
    require(set(s['macro']) == set(c['gt_tracks']) and s['other_benchmarks'] == ps['other_benchmarks'], 'N12 other benchmarks differ')
    for track, arms in s['macro'].items():
        require(set(arms) == set(ARMS), 'N12 missing macro arm')
        for arm, metrics in arms.items():
            require(set(metrics) == set(c['metrics']), 'N12 missing metric')
            for k, v in metrics.items(): require(abs(v - mean(row[arm][track][k] for row in s['rows'].values())) < 1e-12, 'N12 macro differs')
    for path, digest in inputs.items(): require(sha(path) == digest, 'N12 changed input: ' + path); receipts[path] = digest
    spatial = study / 'nvos_reference_completion_spatial_v1'; spatial_record = None
    if (spatial / 'summary.json').exists():
        spatial_record = read(spatial / 'summary.json'); src = read(spatial / 'source_snapshot.json')
        require(spatial_record['complete'] and spatial_record['diagnostic_only'] and spatial_record['tasks'] == 8 and
                spatial_record['target_gt_after_parent_scoring'] and spatial_record['parent_summary_sha256'] == sha(root / 'summary.json') and
                spatial_record['parent_seal_sha256'] == sha(root / 'predictions_complete.json') and
                spatial_record['figure_sha256'] == sha(spatial / 'errors.png') and
                spatial_record['source_snapshot_sha256'] == sha(spatial / 'source_snapshot.json') and
                hashlib.sha256(src['text'].encode()).hexdigest() == src['sha256'], 'N12 spatial record differs')
        for path in spatial.iterdir():
            if path.is_file(): receipts[str(path)] = sha(path)
    return dict(**item, complete=True, status='completed_not_promoted', fresh_prediction_jobs=8, images=72,
        primary_arm=c['primary_arm'], rows=s['rows'], macro=s['macro'], receipts=receipts, implementation_checks=checks,
        decoder=decoder, source_teacher_checks=sources, transitions=s['transitions_diagnostic_only'], spatial=spatial_record,
        previous_candidate_rows=incumbent['rows'], previous_candidate_macro=incumbent['macro'],
        average_precision_measured=True, negative_scribble_used=True, independent_test=False, other_jobs_fresh_inference=False, default_changed=False)
