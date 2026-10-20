"""Validate full-cohort instance admission with immutable parent proposals."""
import hashlib
import json
from statistics import mean
from .reference_study import read, sha, require, check_binary_metrics

ARMS = ['v10', 'static_image', 'static_native', 'union_image', 'union_native',
        'prior_anchor_image', 'prior_anchor_native', 'scribble_anchor_image', 'scribble_anchor_native']
POLICIES = ['union', 'prior_anchor', 'scribble_anchor']


def check_instance_extent_study(repo, study, item, jobs):
    root = study / item['run']
    if not (root / 'summary.json').exists():
        return dict(**item, complete=False, status='failed' if (root / 'failure.json').exists() else 'in_progress' if root.exists() else 'planned')
    m, s, seal, snapshot = [read(root / p) for p in ['manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json']]
    c = m['config']; parent = study / c['parent_run']; pm, ps, pseal = [read(parent / p) for p in ['manifest.json', 'summary.json', 'predictions_complete.json']]
    expected = [j for j in jobs if j['benchmark'] == 'nvos']; names = {j['name'] for j in expected}
    require(len(names) == 8 and m['jobs'] == pm['jobs'] == expected and c['arms'] == ARMS, 'N11 cohort/arms differ')
    require(c == json.loads(snapshot['configs/research/nvos_instance_extent.json']['text']) and c['solver_steps'] == 120
            and c['native_wh'] == [1008, 756] and c['source_hw'] == [384, 512] and c['footprint_backend'] == 'gsplat_cuda'
            and c['threshold'] == seal['threshold'] == .5 and c['primary_arm'] == s['primary_arm'] == 'scribble_anchor_native', 'N11 recipe differs')
    require(s['complete'] and not s['default_changed'] and not s['target_gt_used_for_prediction'] and not s['per_task_winners']
            and not s['independent_test'] and s['target_rgb_used'] and s['negative_scribble_used'] and s['average_precision_measured']
            and s['durable_dense_prediction_bytes'] == 0, 'N11 scope differs')
    require(seal['jobs'] == 8 and seal['images'] == 72 and not seal['target_gt_read'] and
            set(seal['report_hashes']) == set(seal['predictions']) == set(s['rows']) == set(s['transitions_diagnostic_only']) == names
            and s['prediction_seal_sha256'] == sha(root / 'predictions_complete.json'), 'N11 incomplete seal')
    require(m['parent_manifest_sha256'] == sha(parent / 'manifest.json') and s['parent_summary_sha256'] == sha(parent / 'summary.json')
            and m['inherited_source_snapshot_sha256'] == sha(parent / 'source_snapshot.json'), 'N11 parent binding differs')
    for p, h in m['source_hashes'].items():
        require(hashlib.sha256(snapshot[p]['text'].encode()).hexdigest() == h == snapshot[p]['sha256'], 'N11 source snapshot differs')
    receipts = {str(p): sha(p) for p in root.iterdir() if p.is_file()}
    for filename in ['manifest.json', 'summary.json', 'predictions_complete.json', 'source_snapshot.json']:
        receipts[str(parent / filename)] = sha(parent / filename)
    inputs = dict(s['gt_hashes']); sources, checks, admission, witnesses = {}, {}, {}, {}
    for name in names:
        rp = root / (name + '_prediction.json'); ep = root / (name + '_evaluation.json'); pp = parent / (name + '_prediction.json')
        r, pr = read(rp), read(pp); row = s['rows'][name]
        require(r['name'] == name and sha(rp) == seal['report_hashes'][name] and not r['target_gt_read']
                and r['predictions'] == seal['predictions'][name] and set(r['predictions']) == set(ARMS), 'N11 prediction differs')
        require(sha(pp) == r['parent_report_sha256'] == pseal['report_hashes'][name], 'N11 parent report differs')
        require(r['parent_arrays'] == {a: pr['predictions'][a]['sha256'] for a in ['v10', 'static_image', 'signed_image']}, 'N11 proposal replay differs')
        require(row == read(ep) and set(row) == set(ARMS), 'N11 evaluation differs')
        checks[name] = cc = r['implementation_checks']
        require(cc['v10_binary_exact'] and all(0 <= cc[k] < 1e-4 for k in ['v10_render_max_error', 'reference_alpha_max_error', 'target_alpha_max_error', 'normalized_operator_max_error']), 'N11 renderer check failed')
        witnesses[name] = w = r['witness']
        require(not w['unknown_is_negative'] and 0 <= w['maximum_signed_mass'] <= 1 + 1e-6 and w['visible_gaussians'] > 0, 'N11 witness authority differs')
        admission[name] = r['admission']; require(set(r['admission']) == set(POLICIES), 'N11 admission arms missing')
        require(r['admission']['union']['components'] == r['admission']['union']['admitted_components'], 'N11 unrestricted control differs')
        for v in r['admission'].values():
            require(v['component_sha256'] == r['admission']['union']['component_sha256'] and v['image_monotone']
                    and 0 <= v['admitted_components'] <= v['components'], 'N11 component support differs')
        fields = {'static_native', *[a + '_native' for a in POLICIES]}
        require(set(r['lifting']) == fields and all(v['unsupported_changed'] == 0 and v['source_image_loss_after'] <= v['source_image_loss_before'] + 1e-5 for v in r['lifting'].values()), 'N11 lifting check failed')
        sources[name] = r['source_teacher_checks']
        require(sources[name] and all(set(v['confidence_weighted_mean_huber']) == fields | {'v10'} for v in sources[name]), 'N11 source checks missing')
        for a, tracks in row.items():
            require(set(tracks) == set(c['gt_tracks']), 'N11 missing GT track')
            for t, v in tracks.items():
                check_binary_metrics(v); size = sum(v[k] for k in ['tp', 'fp', 'fn', 'tn']); old = row['v10'][t]
                require(type(v['tn']) is int and v['tn'] >= 0 and v['independent_confusion_exact'] and 0 <= v['average_precision'] <= 1
                        and abs(v['accuracy'] - (v['tp'] + v['tn']) / size) < 1e-12
                        and v['tp'] + v['fn'] == old['tp'] + old['fn'] and v['tn'] + v['fp'] == old['tn'] + old['fp'], 'N11 confusion check failed')
                if a in ['v10', 'static_image', 'static_native']:
                    require(all(v[k] == ps['rows'][name][a][t][k] for k in ['tp', 'fp', 'fn', 'tn']), 'N11 control counts differ')
                if a in ['v10', 'static_image']:
                    require(v == ps['rows'][name][a][t] and r['predictions'][a]['sha256'] == pr['predictions'][a]['sha256'], 'N11 exact control differs')
                if a in [a + '_image' for a in POLICIES]:
                    require(v['tp'] >= row['static_image'][t]['tp'] and v['fp'] >= row['static_image'][t]['fp'], 'N11 additive image property failed')
        for a, transition in s['transitions_diagnostic_only'][name].items():
            new, old = row[a]['retained_gt_gt127'], row['static_native']['retained_gt_gt127']
            require(transition['recovered_foreground'] - transition['lost_foreground'] == new['tp'] - old['tp'] and
                    transition['added_false_positive'] - transition['removed_false_positive'] == new['fp'] - old['fp'], 'N11 transition counts differ')
        receipts[str(pp)] = sha(pp)
        for p, h in pr['input_hashes'].items():
            require(p not in inputs or inputs[p] == h, 'N11 inputs inconsistent'); inputs[p] = h
    for t, arms in s['macro'].items():
        require(set(arms) == set(ARMS), 'N11 macro arms missing')
        for a, metrics in arms.items():
            require(set(metrics) == set(c['metrics']), 'N11 metrics differ')
            for k, v in metrics.items(): require(abs(v - mean(r[a][t][k] for r in s['rows'].values())) < 1e-12, 'N11 mean differs')
    require(set(s['macro']) == set(c['gt_tracks']) and s['other_benchmarks'] == ps['other_benchmarks'] and not s['other_benchmarks_freshly_evaluated'], 'N11 cohort scope differs')
    for p, h in inputs.items(): require(sha(p) == h, 'N11 input changed: ' + p); receipts[p] = h
    spatial = study / 'nvos_instance_extent_spatial_v1'
    spatial_record = None
    if (spatial / 'summary.json').exists():
        spatial_record = read(spatial / 'summary.json'); src = read(spatial / 'source_snapshot.json')
        require(spatial_record['complete'] and spatial_record['diagnostic_only'] and spatial_record['tasks'] == 8
                and spatial_record['target_gt_after_parent_scoring'] and spatial_record['parent_summary_sha256'] == sha(root / 'summary.json')
                and spatial_record['parent_seal_sha256'] == sha(root / 'predictions_complete.json')
                and spatial_record['figure_sha256'] == sha(spatial / 'errors.png')
                and spatial_record['source_snapshot_sha256'] == sha(spatial / 'source_snapshot.json')
                and hashlib.sha256(src['text'].encode()).hexdigest() == src['sha256'], 'N11 spatial binding differs')
        for p in spatial.iterdir():
            if p.is_file(): receipts[str(p)] = sha(p)
    return dict(**item, complete=True, status='completed_not_promoted', fresh_prediction_jobs=8, images=72,
        primary_arm='scribble_anchor_native', rows=s['rows'], macro=s['macro'], receipts=receipts, implementation_checks=checks,
        admission=admission, witnesses=witnesses, source_teacher_checks=sources, transitions=s['transitions_diagnostic_only'], spatial=spatial_record,
        average_precision_measured=True, negative_scribble_used=True, independent_test=False, other_jobs_fresh_inference=False, default_changed=False)
