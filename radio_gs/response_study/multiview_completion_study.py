"""Validate N15 lineage, full-cohort scoring, source access and matched controls."""
import hashlib
import json
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics

ARMS = ['v10', 'quality_image', 'quality_native', 'legacy_source_native',
        'completed_source_native', 'completed_joint_native']


def check_multiview_completion_study(repo, study, item, jobs):
    root = study / item['run']; old = repo / 'output/tmm_response_20261016/study'
    if not (root / 'summary.json').exists():
        return dict(**item, complete=False, status='failed' if (root / 'failure.json').exists()
                    else 'in_progress' if root.exists() else 'planned')
    m, s, seal, snapshot = [read(root / p) for p in
                           ['manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json']]
    c = m['config']; parent = old / c['parent_run']; candidate = study / c['candidate_run']
    ps, cs = read(parent / 'summary.json'), read(candidate / 'summary.json')
    expected = [j for j in jobs if j['benchmark'] == 'nvos']; names = {j['name'] for j in expected}
    require(m['jobs'] == read(parent / 'manifest.json')['jobs'] == expected and len(names) == 8 and
            c['arms'] == ARMS, 'N15 cohort differs')
    require(c == json.loads(snapshot['configs/research/nvos_multiview_completion.json']['text']) and
            c['primary_arm'] == s['primary_arm'] == 'completed_joint_native' and c['solver_steps'] == 120 and
            c['threshold'] == seal['threshold'] == .5 and c['native_wh'] == [1008, 756] and
            c['source_hw'] == [384, 512] and c['points_per_sign'] == 16, 'N15 recipe differs')
    require(read(study / 'storage_policy.json') == c['storage_policy'] and
            c['storage_policy']['study_bytes'] == 8*2**20 and c['maximum_new_durable_bytes'] == 2**20 and
            c['maximum_volatile_prediction_bytes'] == c['maximum_source_evidence_bytes'] == 256*2**20,
            'N15 storage contract differs')
    require(s['complete'] and not s['default_changed'] and not s['target_gt_used_for_prediction'] and
            not s['per_task_winners'] and not s['independent_test'] and s['target_rgb_used'] and
            s['negative_scribble_used'] and s['average_precision_measured'] and
            s['durable_dense_prediction_bytes'] == 0 and not s['other_benchmarks_freshly_evaluated'], 'N15 scope differs')
    require(seal['jobs'] == 8 and seal['images'] == 48 and not seal['target_gt_read'] and
            set(seal['report_hashes']) == set(seal['predictions']) == set(s['rows']) == names and
            s['prediction_seal_sha256'] == sha(root / 'predictions_complete.json'), 'N15 prediction seal differs')
    require(m['parent_manifest_sha256'] == sha(parent / 'manifest.json') and
            s['parent_summary_sha256'] == sha(parent / 'summary.json') and
            m['candidate_manifest_sha256'] == sha(candidate / 'manifest.json') and
            s['candidate_summary_sha256'] == s['previous_candidate_summary_sha256'] == sha(candidate / 'summary.json'),
            'N15 parent binding differs')
    receipts = {str(p): sha(p) for p in root.iterdir() if p.is_file()}
    archives = [(parent, 'inherited_source_snapshot_sha256'),
                (old / 'nvos_memory_response_v2', 'inherited_memory_snapshot_sha256'),
                (old / 'nvos_instance_extent_v2', 'inherited_instance_snapshot_sha256'),
                (candidate, 'inherited_completion_snapshot_sha256')]
    inherited = {}
    for directory, key in archives:
        require(m[key] == sha(directory / 'source_snapshot.json'), 'N15 ancestor archive differs')
        inherited.update(read(directory / 'source_snapshot.json'))
        for file in ['source_snapshot.json', 'manifest.json', 'summary.json']:
            receipts[str(directory / file)] = sha(directory / file)
    for run, digest in read(root / 'completion_lineage.json').items():
        p = study / run / 'source_snapshot.json'
        require(sha(p) == digest, 'N15 completion archive changed')
        inherited.update(read(p)); receipts[str(p)] = digest
    inherited.update(snapshot)
    for path, entry in inherited.items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'N15 archived source inconsistent')
    for path, digest in m['source_hashes'].items():
        require(snapshot[path]['sha256'] == digest, 'N15 source binding differs')
    for path, digest in m['worker_source_hashes'].items():
        require(path in inherited and inherited[path]['sha256'] == digest, 'N15 worker source not archived: ' + path)
    inputs = dict(s['gt_hashes']); diagnostics = {}; checks = {}; decoders = {}
    for job in expected:
        name = job['name']; path = root / (name + '_prediction.json'); r = read(path); row = s['rows'][name]
        require(r['name'] == name and sha(path) == seal['report_hashes'][name] and not r['target_gt_read'] and
                r['predictions'] == seal['predictions'][name] and set(r['predictions']) == set(ARMS) and
                row == read(root / (name + '_evaluation.json')) and set(row) == set(ARMS), 'N15 task seal differs')
        pp = parent / (name + '_prediction.json'); cp = candidate / (name + '_prediction.json')
        pr, cr = read(pp), read(cp)
        require(r['parent_report_sha256'] == sha(pp) and r['candidate_report_sha256'] == sha(cp), 'N15 task lineage differs')
        require(r['parent_arrays'] == {a: cr['predictions'][a]['sha256'] for a in ['v10', 'quality_image']}, 'N15 replay differs')
        scene = read(repo / job['scene']); spec = read(repo / job['evaluation'])
        require(r['source_frames'] == scene['source_frames'] and r['target_frame'] == spec['target_frame'] and
                r['target_frame'] not in r['source_frames'] and
                set(r['source_rgb_hashes']) == set(scene['source_rgb'].values()), 'N15 source access differs')
        d = r['decoder']; decoders[name] = d
        require(d['source_hashes'] == m['worker_source_hashes'] and not d['target_gt_available'] and
                not d['reference_gt_available'] and d['source_evidence_before_target_rgb'] and
                d['sam_eval'] and d['dino_eval'] and d['choices'] == cr['decoder']['choices'] and
                set(d['sources']) == set(map(str, r['source_frames'])), 'N15 decoder contract differs')
        completion_path = study / 'nvos_reference_completion_v1' / (name + '_prediction.json')
        original = read(completion_path)['decoder']['decoder'][d['choices']['quality']]
        require(d['selected_reference_sha256'] == original['reference_sha256'], 'N15 reference changed')
        require(r['lifting']['target_confidence_mass'][:2] == [0., 0.] and
                r['lifting']['unsupported_changed'] == r['incumbent_fit']['unsupported_changed'] == 0 and
                r['lifting']['objective_after'] <= r['lifting']['objective_before'] + 1e-5,
                'N15 target-independent controls or fit differ')
        require(r['transient_source_bytes'] < c['maximum_source_evidence_bytes'] and
                r['sparse_operator_bytes'] < c['maximum_sparse_operator_bytes'], 'N15 transient budget exceeded')
        cc = r['implementation_checks']; checks[name] = cc
        require(cc['v10_binary_exact'] and 0 <= cc['v10_render_max_error'] < 1e-4 and
                [x['frame'] for x in r['source_operator_checks']] == r['source_frames'], 'N15 operator checks missing')
        for check in [cc, *r['source_operator_checks']]:
            require(all(0 <= check[k] < 1e-4 for k in ['alpha_max_error', 'normalized_operator_max_error']), 'N15 renderer differs')
        diagnostics[name] = r['source_consistency']
        require([x['frame'] for x in diagnostics[name]] == r['source_frames'], 'N15 source diagnostics incomplete')
        for arm, tracks in row.items():
            require(set(tracks) == set(c['gt_tracks']), 'N15 GT track missing')
            for track, v in tracks.items():
                check_binary_metrics(v); size = sum(v[k] for k in ['tp', 'fp', 'fn', 'tn']); control = row['v10'][track]
                require(type(v['tn']) is int and v['tn'] >= 0 and v['independent_confusion_exact'] and
                        0 <= v['average_precision'] <= 1 and abs(v['accuracy']-(v['tp']+v['tn'])/size) < 1e-12 and
                        v['tp']+v['fn'] == control['tp']+control['fn'] and v['tn']+v['fp'] == control['tn']+control['fp'],
                        'N15 confusion differs')
                if arm in ['v10', 'quality_image', 'quality_native']:
                    require(all(v[k] == cs['rows'][name][arm][track][k] for k in ['tp', 'fp', 'fn', 'tn']), 'N15 N14 control differs')
        for arm, delta in s['transitions_diagnostic_only'][name].items():
            a, b = row[arm]['retained_gt_gt127'], row['quality_native']['retained_gt_gt127']
            require(delta['recovered_foreground']-delta['lost_foreground'] == a['tp']-b['tp'] and
                    delta['added_false_positive']-delta['removed_false_positive'] == a['fp']-b['fp'], 'N15 transitions differ')
        for p in [pp, cp, completion_path]: receipts[str(p)] = sha(p)
        for path, digest in {**pr['input_hashes'], **r['source_rgb_hashes']}.items():
            require(path not in inputs or inputs[path] == digest, 'N15 conflicting input'); inputs[path] = digest
    require(s['other_benchmarks'] == cs['other_benchmarks'] == ps['other_benchmarks'] and
            set(s['macro']) == set(c['gt_tracks']), 'N15 other outputs changed')
    for track, arms in s['macro'].items():
        require(set(arms) == set(ARMS), 'N15 macro arm missing')
        for arm, metrics in arms.items():
            require(set(metrics) == set(c['metrics']), 'N15 macro metric missing')
            for k, v in metrics.items():
                require(abs(v-mean(row[arm][track][k] for row in s['rows'].values())) < 1e-12, 'N15 macro differs')
    for path, digest in inputs.items():
        require(sha(path) == digest, 'N15 input changed: ' + path); receipts[path] = digest
    spatial = study / 'nvos_multiview_completion_spatial_v1'; spatial_record = None
    if (spatial / 'summary.json').exists():
        spatial_record = read(spatial / 'summary.json'); src = read(spatial / 'source_snapshot.json')
        require(spatial_record['complete'] and spatial_record['tasks'] == 8 and spatial_record['diagnostic_only'] and
                spatial_record['target_gt_after_parent_scoring'] and spatial_record['parent_summary_sha256'] == sha(root / 'summary.json') and
                spatial_record['parent_seal_sha256'] == sha(root / 'predictions_complete.json') and
                spatial_record['figure_sha256'] == sha(spatial / 'errors.png') and
                spatial_record['source_snapshot_sha256'] == sha(spatial / 'source_snapshot.json') and
                hashlib.sha256(src['text'].encode()).hexdigest() == src['sha256'], 'N15 spatial provenance differs')
        for p in spatial.iterdir():
            if p.is_file(): receipts[str(p)] = sha(p)
    return dict(**item, complete=True, status='completed_not_promoted', fresh_prediction_jobs=8, images=48,
                primary_arm=c['primary_arm'], rows=s['rows'], macro=s['macro'], receipts=receipts,
                decoder=decoders, implementation_checks=checks, source_consistency=diagnostics,
                transitions=s['transitions_diagnostic_only'], spatial=spatial_record,
                independent_test=False, other_jobs_fresh_inference=False, default_changed=False)
