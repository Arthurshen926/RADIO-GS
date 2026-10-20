"""Read-only evidence checks and a normalized results ledger (standard library).

These checks validate retained receipts and aggregates, not predictions against
ground truth. Historical source availability is reported separately from result
integrity. Inference and dense-array verification belong to the sealed runners.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = 'configs/research/tmm_response.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) < 1e-8, f'{label}: {actual} != {expected}')


def job_key(job):
    return job['benchmark'], job['name']


def check_cohort(jobs, reference, counts):
    keys = [job_key(job) for job in jobs]
    require(len(keys) == len(set(keys)), 'duplicate cohort job')
    require(Counter(job['benchmark'] for job in jobs) == counts, 'incomplete cohort')
    require(sorted(jobs, key=job_key) == sorted(reference, key=job_key),
            'cohort contracts differ from frozen recipe')


def check_source_decision_audit(study, reference_jobs):
    root = study/'source_decision_audit_v1'
    summary, seal = read(root/'summary.json'), read(root/'decisions_complete.json')
    expected = [j for j in reference_jobs if j['benchmark'] != 'uqis']
    check_cohort([r['job'] for r in summary['rows']], expected, {'lerf': 4, 'nvos': 8, 'scannet': 10})
    require(summary['complete'] and summary['diagnostic_only'] and not summary['default_changed']
            and summary['dense_prediction_bytes'] == 0 and not summary['evidence_held_out'], 'A08: scope changed')
    require(seal['jobs'] == len(seal['report_hashes']) == 22 and not seal['ground_truth_read'], 'A08: invalid decision seal')
    for name, digest in seal['report_hashes'].items():
        require(sha(root/name) == digest, 'A08: changed decision receipt')
        report = read(root/name)
        require(not report['ground_truth_read'] and report['complete_candidates_exact'], 'A08: input status changed')
        units = 1 if report['job']['benchmark'] == 'scannet' else len(report['columns'])
        for arm in ('full', 'permuted'):
            value = report[arm]
            require(len(value['choices']) == len(value['views']) == units, 'A08: incorrect decision unit')
            for i, choice in enumerate(value['choices']):
                require(value['gains'][0][i] == 0 and choice == int(value['gains'][1][i] > 0), 'A08: changed label-free choice')
    for row in summary['rows']:
        for path, digest in row['cached_score_hashes'].items():
            require(sha(path) == digest, 'A08: changed historical scoring record')
    for family, arms in summary['macro'].items():
        local = [r for r in summary['rows'] if r['job']['benchmark'] == family]
        for arm, values in arms.items():
            for metric, value in values.items():
                close(value, sum(r['metrics'][arm][metric] for r in local)/len(local), 'A08: aggregate changed')
    return dict(source_jobs=22, dense_E02_candidates_verified_by_runner=True,
                fresh_target_scoring=False, historical_pool_is_diagnostic=True)


def check_cross_task_targets(study, benchmark_jobs):
    root = study/'cross_task_targets_v1'
    summary, manifest = read(root/'summary.json'), read(root/'manifest.json')
    require(summary['complete'] and summary['target_manifests_ready'] and not summary['utility_training_ready']
            and summary['dense_artifact_bytes'] == 0, 'A09: targets misrepresented as utility training')
    split_names = sum(manifest['splits'].values(), [])
    require(len(set(split_names)) == len(split_names) == summary['rooms'] == 29, 'A09: split changed')
    benchmark_rooms = {j['name'].split('_')[0] for j in benchmark_jobs}
    require(not {s.split('_')[0] for s in split_names}&benchmark_rooms, 'A09: benchmark room overlap')
    require(set(summary['report_hashes']) == {s+'.json' for s in split_names}, 'A09: missing target room')
    coverage = {split: {str(c): 0 for c in manifest['complete_menu']} for split in manifest['splits']}
    for name, digest in summary['report_hashes'].items():
        require(sha(root/name) == digest, 'A09: changed target receipt')
        room = read(root/name)
        require(room['scene'] in manifest['splits'][room['split']] and not room['utility_training_ready'], 'A09: room split/status changed')
        require([t['class_id'] for t in room['targets']] == manifest['complete_menu']
                and room['absent_classes_retained'], 'A09: incomplete menu')
        for target in room['targets']:
            coverage[room['split']][str(target['class_id'])] += target['vertices'] > 0
        for fold in room['view_folds']:
            a, b = set(fold['candidate_view_indices']), set(fold['evidence_view_indices'])
            require(not a&b and a|b == set(range(room['source_views'])), 'A09: views overlap or are missing')
    require(coverage == summary['class_room_coverage'], 'A09: stale class coverage')
    return dict(rooms=29, utility_training_ready=False,
                missing_calibration_classes=summary['missing_calibration_classes'])


def check_category_preparation(root, manifest, summary, recipe):
    expected=[j for j in recipe['jobs'] if j['benchmark']=='scannet']
    check_cohort(manifest['jobs'],expected,{'scannet':10})
    require(summary['jobs']==10 and not summary['ground_truth_read'] and not summary['model_read']
            and summary['durable_dense_prediction_bytes']==0, 'category preparation: scope changed')
    require(set(summary['report_hashes'])=={j['name']+'.json' for j in expected}, 'category preparation: missing scenes')
    views=regions=0
    for name,digest in summary['report_hashes'].items():
        require(sha(root/name)==digest,'category preparation: changed report')
        row=read(root/name);views+=row['views'];regions+=row['regions']
        require(not row['ground_truth_read'] and not row['model_read'] and row['original_gaussian_carrier']
                and row['all_applicable_teachers_retained'] and len(row['columns'])==19
                and not row['independent_mesh_signature_domain_equivalence'], 'category preparation: changed input domain')
        require(len(row['features'])==3 and all(len(x)==39 for x in row['features']), 'category preparation: incomplete signatures')
        for key in ('primary','fusion'):
            value=row['compatibility'][key]
            require(set(value['changed_subset_labels'])=={'19','15','10'}
                    and not any(value['changed_subset_labels'].values())
                    and value['max_abs']<=2e-5 and value['mean_abs']<=1e-6, 'category preparation: failed incumbent replay')
    return dict(scenes=10,views=views,regions=regions,evaluation_result=False,
                ground_truth_read=False,original_incumbents_replayed=True)


def check_multitask_transfer(study,root,manifest,summary,recipe,reference):
    expected=[j for j in recipe['jobs'] if j['benchmark'] in ('scannet','uqis')]
    check_cohort(manifest['jobs'],expected,{'scannet':10,'uqis':9})
    seal=read(root/'predictions_complete.json');evaluation=read(root/'evaluation_manifest.json')
    require(seal['category_jobs']==10 and seal['uqis_intervention_queries']==134 and not seal['ground_truth_read'],
            'joint transfer: incomplete label-free decisions')
    names={j['benchmark']+'_'+j['name']+'.json' for j in expected}
    require(set(seal['report_hashes'])==names,'joint transfer: incomplete decision reports')
    for name,digest in seal['report_hashes'].items():
        require(sha(root/name)==digest and not read(root/name)['ground_truth_read'], 'joint transfer: changed report')
    require(evaluation['prediction_seal_sha256']==sha(root/'predictions_complete.json')
            and evaluation['inputs_rechecked'], 'joint transfer: invalid scoring seal')
    trained=study/manifest['training_run'];prepared=study/manifest['prepared_run']
    require(sha(trained/'model.joblib')==manifest['model_sha256']
            and sha(trained/'calibration.json')==manifest['calibration_sha256']
            and sha(prepared/'summary.json')==manifest['prepared_summary_sha256'], 'joint transfer: changed shared policy or pools')
    require(summary['freshly_scored_category_scenes']==10 and summary['freshly_scored_uqis_queries']==268
            and summary['one_shared_checkpoint'] and not summary['all_five_transfer_complete']
            and not summary['independent_final_test'], 'joint transfer: overstated scientific scope')
    require(summary['one_shared_full_policy_margin']==manifest['margins']['full'], 'joint transfer: margin changed')
    calibrated=read(trained/'calibration.json')
    require(manifest['margins']=={a:v['selection']['margin'] for a,v in calibrated.items()},
            'joint transfer: model-specific calibration changed')
    if manifest.get('matched_no_null_control'):
        require(set(manifest['arms'])=={'full','query_blind','permuted','unrestricted','unregularized'},
                'joint transfer: missing matched no-null control')
        for name in names:
            report=read(root/name)
            rows=report['rows'] if name.startswith('uqis_') else [report]
            for row in rows:
                require(set(row['arms'])==set(manifest['arms']), 'joint transfer: incomplete control decisions')
                for arm,decision in row['arms'].items():
                    key=arm if arm in ('query_blind','unregularized') else 'full'
                    expected_margin=0. if arm=='unrestricted' else manifest['margins'][key]
                    require(decision['margin']==expected_margin, 'joint transfer: per-task control margin changed')
    require(len(summary['category_rows'])==10,'joint transfer: incomplete category scoring')
    for arm,values in summary['category_macro'].items():
        for key,value in values.items():
            close(value,sum(r['metrics'][arm][key] for r in summary['category_rows'])/10,'joint category aggregate')
            if arm=='baseline':close(value,reference['macro']['scannet'][key],'joint category baseline')
    for arm,values in summary['uqis_macro'].items():
        for suffix,filename,key in [('ap',f'uqis_{arm}.json','average_precision'),
                                    ('selected_iou',f'uqis_{arm}_sets.json','iou')]:
            scored=read(root/filename)
            require(scored['query_count']==268,'joint transfer: missing UQIS scores')
            for modality,record in scored['modalities'].items():
                require(len(record['per_scene'])==9,'joint transfer: missing UQIS rooms')
                metric=modality+'_'+suffix
                close(values[metric],sum(v[key] for v in record['per_scene'].values())/9,'joint UQIS aggregate')
                if arm=='baseline' or modality.startswith('point_'):
                    close(values[metric],reference['macro']['uqis'][metric],'joint UQIS preserved output')
    return dict(category_scenes=10,intervention_queries=134,scored_queries=268,
                other_benchmarks='transfer_pending',category_macro=summary['category_macro'],uqis_macro=summary['uqis_macro'])


def provider_audit(repo, recipe):
    """Resolve the actual configured providers, without loading model weights."""
    records = []
    for job in recipe['jobs']:
        if job['benchmark'] != 'uqis':
            continue
        path = repo / job['scene']
        spec = read(path)
        direct = spec.get('direct_anchor_membership')
        require(direct and direct['applies_to'] == ['point_2d', 'point_3d'],
                f'{job["name"]}: actual point provider changed')
        require(spec.get('inferred_anchor_membership') == ['image'],
                f'{job["name"]}: inferred-anchor provider changed')
        require(not spec.get('query_conditioned_membership'),
                f'{job["name"]}: optional membership candidate became active')
        records.append(dict(scene=job['name'], path=job['scene'], sha256=sha(path),
                            direct_anchor_membership=direct))
    require(len(records) == 9, 'expected all nine UQIS maps')
    return dict(maps=records, active_image_provider='DirectAnchorMembership.infer_region',
        active_point_provider='DirectAnchorMembership.infer',
        query_conditioned_membership_candidate_active=False,
        implementation_observation='infer_region calls infer, which forces the anchor to probability one',
        limitation='estimated-image-anchor uncertainty is declared but does not change positive authority',
        runtime_source_sha256={p: sha(repo / p) for p in [
            'radio_gs/querying/direct_anchor_membership.py',
            'radio_gs/querying/registered_observation_query.py']})


def source_index(study):
    index = {}
    for path in sorted(study.glob('*/source_snapshot.json')):
        record = read(path)
        for name, item in record.get('files', record).items():
            if not isinstance(item, dict) or 'text' not in item:
                continue
            digest = hashlib.sha256(item['text'].encode()).hexdigest()
            require(digest == item['sha256'], f'corrupt source snapshot: {path}:{name}')
            index.setdefault(digest, []).append(str(path))
    return index


def source_availability(repo, stages, index):
    rows = []
    for stage, hashes in stages.items():
        for path, digest in hashes.items():
            current = repo / path
            state = ('current' if current.is_file() and sha(current) == digest else
                     'archived' if digest in index else 'missing_version')
            rows.append(dict(stage=stage, path=path, sha256=digest, availability=state,
                             snapshots=index.get(digest, [])))
    return rows


def check_seal(repo, study, run, jobs):
    root = study / run
    seal = read(root / 'predictions_complete.json')
    require(seal['jobs'] == len(jobs) and seal['all_five_present']
            and seal['no_target_gt_read'], f'{run}: invalid prediction seal')
    expected = {(study / f'{run}_{j["benchmark"]}_{j["name"]}' / 'prediction.json').resolve(): j
                for j in jobs}
    reports = {Path(p).resolve(): h for p, h in seal['reports'].items()}
    require(set(reports) == set(expected), f'{run}: incomplete prediction reports')
    for path, digest in reports.items():
        require(sha(path) == digest, f'{run}: altered prediction report: {path}')
        report = read(path)
        require(report['job'] == expected[path] and report['ground_truth_read'] is False,
                f'{run}: prediction input contract mismatch: {path}')
    evaluation = read(root / 'evaluation_manifest.json')
    require(evaluation['prediction_seal_sha256'] == sha(root / 'predictions_complete.json'),
            f'{run}: evaluation seal mismatch')
    require(evaluation.get('all_inputs_and_predictions_checked',
                           evaluation.get('all_predictions_checked', False)),
            f'{run}: no retained scoring validation receipt')
    return evaluation


def check_aggregates(root, experiment, summary, feedback, reference, counts):
    run = experiment['run']
    arms = ['baseline'] + experiment['arms']
    rows = summary['rows']
    require(len({job_key(r['job']) for r in rows}) == len(rows) == 22,
            f'{run}: duplicate/missing scored source jobs')
    require(Counter(r['job']['benchmark'] for r in rows) == {k: v for k, v in counts.items() if k != 'uqis'},
            f'{run}: scored source cohort differs')
    for family, by_arm in summary['macro'].items():
        require(set(by_arm) == set(arms), f'{run}: arm mismatch for {family}')
        for metric, value in by_arm['baseline'].items():
            close(value, reference[family][metric], f'{run}: changed baseline {family}/{metric}')
        if family == 'uqis':
            continue
        selected = [r for r in rows if r['job']['benchmark'] == family]
        for arm in arms:
            for metric, value in by_arm[arm].items():
                close(value, sum(r['metrics'][arm][metric] for r in selected)/len(selected),
                      f'{run}: source aggregation {family}/{arm}/{metric}')
    fixed = read(root / 'uqis_fixed_selections.json')
    require(fixed['query_count'] == 268, f'{run}: fixed set query count changed')
    for arm in arms:
        scored = read(root / f'uqis_{arm}.json')
        require(scored['query_count'] == 268 and scored['scene_count'] == 9,
                f'{run}: UQIS cohort incomplete')
        for modality in ('text', 'image', 'point_2d', 'point_3d'):
            for suffix, source, key in [('ap', scored, 'average_precision'),
                                         ('selected_iou', fixed, 'iou')]:
                records = source['modalities'][modality]
                require(len(records['per_scene']) == 9, f'{run}: missing UQIS scene')
                value = sum(v[key] for v in records['per_scene'].values()) / 9
                close(value, records['scene_macro'][key], f'{run}: UQIS scene macro')
                metric = f'{modality}_{suffix}'
                close(value, summary['macro']['uqis'][arm][metric], f'{run}: UQIS summary')
    expected = {(f, k) for f, a in summary['macro'].items() for k in a['baseline']}
    require(len(feedback['rows']) == len(expected) == 16 and
            {(r['family'], r['metric']) for r in feedback['rows']} == expected,
            f'{run}: feedback metrics missing or duplicated')
    for row in feedback['rows']:
        family, metric = row['family'], row['metric']
        fixed_head = metric == 'selected_projection_iou' or metric.endswith('_selected_iou')
        require(row['independent_set_head_frozen'] == fixed_head, f'{run}: set head mislabeled')
        require(set(row['values_percent']) == set(arms), f'{run}: feedback arms differ')
        for arm, value in row['values_percent'].items():
            close(value, 100 * summary['macro'][family][arm][metric], f'{run}: feedback value')
            if arm != 'baseline':
                close(row['effects'][arm]['delta_pp'], value - row['values_percent']['baseline'],
                      f'{run}: feedback effect')
                if fixed_head or (family == 'uqis' and not experiment['uqis_intervention']):
                    close(value, row['values_percent']['baseline'], f'{run}: compatibility output changed')


def result_rows(experiment, feedback, summary_sha):
    result = []
    for row in feedback['rows']:
        family, metric = row['family'], row['metric']
        for arm in experiment['arms']:
            effect = row['effects'][arm]
            interval = effect.get('scene_bootstrap95_pp', [None, None])
            result.append(dict(experiment_id=experiment['id'], phase=experiment['phase'], arm=arm,
                benchmark=('lerf_selected_projection_2d' if metric == 'selected_projection_iou' else
                           'lerf2d' if family == 'lerf' else family), metric=metric,
                value_percent=row['values_percent'][arm],
                reference_percent=row['values_percent']['baseline'], delta_pp=effect['delta_pp'],
                descriptive_ci_low_pp=interval[0], descriptive_ci_high_pp=interval[1],
                intervention_applied=not row['independent_set_head_frozen'] and
                    (family != 'uqis' or experiment['uqis_intervention']),
                independent_set_fixed=row['independent_set_head_frozen'], independent_test=False,
                input_access='map-only' if family == 'uqis' else 'source-assisted',
                status=experiment['status'], summary_sha256=summary_sha))
    return result


def check_anchor_audit(study):
    """Verify the sparse causal control without loading any dense predictions."""
    root = study / 'estimated_anchor_authority_v1'
    seal = read(root / 'patches_complete.json')
    require(seal['image_queries'] == 67 and seal['all_uqis_queries'] == 268
            and seal['ground_truth_read'] is False, 'incomplete anchor patch seal')
    require(len(seal['report_hashes']) == 9, 'anchor audit needs all nine maps')
    patches, unchanged = [], []
    for path, digest in seal['report_hashes'].items():
        require(sha(path) == digest, 'anchor patch report changed')
        report = read(path)
        require(not report['ground_truth_read'], 'anchor predictions read labels')
        patches.extend(report['patches'])
        unchanged.extend(report['unchanged_queries'])
    require(len({p['query_id'] for p in patches}) == len(patches) == 67,
            'duplicate or missing image patch')
    require(len({p['query_id'] for p in unchanged}) == len(unchanged) == 201,
            'duplicate or missing unchanged query')
    require(not ({p['query_id'] for p in patches} & {p['query_id'] for p in unchanged}),
            'intervention and preserved queries overlap')
    evaluation = read(root / 'evaluation_manifest.json')
    require(evaluation['patch_seal_sha256'] == sha(root / 'patches_complete.json')
            and evaluation['inputs_rechecked'], 'anchor evaluation seal changed')
    summary = read(root / 'summary.json')
    require(summary['complete'] and not summary['all31_fresh_inference']
            and not summary['default_changed'], 'incorrect anchor audit scope')
    for key in ['rank', 'selected']:
        patch_key = 'selection_entries_changed' if key == 'selected' else 'rank_entries_changed'
        require(sum(p[patch_key] for p in patches) == summary['changed_'+key+'_entries'],
                'anchor intervention size differs')
    for arm in ['baseline', 'unforced']:
        for suffix, filename, metric in [('ap', f'uqis_{arm}.json', 'average_precision'),
                                         ('selected_iou', f'uqis_{arm}_selections.json', 'iou')]:
            scored = read(root / filename)
            require(scored['query_count'] == 268, 'anchor scoring cohort incomplete')
            for modality in ['text', 'image', 'point_2d', 'point_3d']:
                rows = scored['modalities'][modality]['per_scene']
                require(len(rows) == 9, 'anchor scoring scene missing')
                value = sum(row[metric] for row in rows.values()) / 9
                close(value, summary['uqis_macro'][arm][modality+'_'+suffix], 'anchor summary aggregation')
    return dict(image_patches=67, fresh_uqis_queries=268, all31_fresh_inference=False,
                changed_rank_entries=summary['changed_rank_entries'],
                changed_selected_entries=summary['changed_selected_entries'])


def check_targeted_study(repo, study, item, recipe, reference, index):
    """Check a real but restricted intervention without calling it all-31 inference."""
    root = study / item['run']
    manifest, summary = read(root/'manifest.json'), read(root/'summary.json')
    seal = read(root/'predictions_complete.json')
    evaluation = read(root/'evaluation_manifest.json')
    expected = [j for j in recipe['jobs'] if j['benchmark'] == 'uqis']
    check_cohort(manifest['jobs'],recipe['jobs'],Counter(j['benchmark'] for j in recipe['jobs']))
    require(manifest['intervention_jobs'] == expected and manifest['arms'] == item['arms'],
            'targeted intervention cohort/arms differ')
    require(summary['complete'] and not summary['all31_fresh_inference']
            and not summary['default_changed'] and not summary['independent_test'],
            'targeted study scope overstated')
    require(seal['image_queries'] == 67 and seal['scored_queries'] == 268
            and seal['ground_truth_read'] is False, 'targeted prediction cohort differs')
    expected_paths = {str(root/(j['name']+'_prediction.json')) for j in expected}
    require({str(Path(p).resolve()) for p in seal['report_hashes']} ==
            {str(Path(p).resolve()) for p in expected_paths}, 'targeted prediction reports differ')
    queries = []
    for path,digest in seal['report_hashes'].items():
        require(sha(path) == digest, 'targeted prediction report changed')
        report = read(path)
        require(report['ground_truth_read'] is False and report['all_other_queries_unchanged'],
                'targeted inference contract differs')
        for row in report['queries']:
            queries.append(row['query_id'])
            if manifest.get('proposals'):
                require(set(row['proposals'])==set(manifest['proposals']), 'response proposals changed')
                first,second=[row['arms'][a] for a in item['arms']]
                require(first['common_views']==second['common_views'] and
                        first['excluded_proposal_views']==second['excluded_proposal_views'],
                        'complete-response verification domains differ')
                for detail in [first,second]:
                    close(detail['query_budget_error'],0,'verification query-score permutation')
                    winner=max(range(len(detail['scores'])),key=lambda i:detail['scores'][i])
                    require(winner==detail['selected_candidate'] and
                            detail['chosen_proposal']==manifest['proposals'][winner],
                            'verification does not return its sealed candidate')
                    require(detail['whole_candidate_preserved'], 'complete response was modified')
            else:
                for key in ['source_views','admitted_regions','source_weight_mass','normalized_view_mass']:
                    close(row['arms'][item['arms'][0]][key],row['arms'][item['arms'][1]][key],
                          'targeted matched evidence budget')
    require(len(set(queries)) == len(queries) == 67, 'duplicate/missing targeted query')
    require(evaluation['prediction_seal_sha256'] == sha(root/'predictions_complete.json')
            and evaluation['all_inputs_and_predictions_rechecked'], 'targeted evaluation seal differs')
    for arm in ['baseline'] + item['arms']:
        for suffix, filename, key in [('ap',f'uqis_{arm}.json','average_precision'),
                                     ('selected_iou',f'uqis_{arm}_selections.json','iou')]:
            scored = read(root/filename)
            require(scored['query_count'] == 268, 'targeted scoring cohort incomplete')
            for modality in ['text','image','point_2d','point_3d']:
                rows = scored['modalities'][modality]['per_scene']
                require(len(rows) == 9, 'targeted scene missing')
                value = sum(r[key] for r in rows.values()) / 9
                metric = modality+'_'+suffix
                close(value, summary['uqis_macro'][arm][metric], 'targeted metric aggregation')
                if arm == 'baseline' or modality != 'image':
                    close(value,reference['macro']['uqis'][metric], 'targeted reference/control changed')
    for family in ['lerf','nvos','scannet']:
        retained = summary['source_benchmark_compatibility'][family]
        require(not retained['changed'] and not retained['fresh_inference'], 'targeted source scope differs')
        require(retained['metrics'] == reference['macro'][family], 'targeted source reference changed')
        require(sha(retained['provenance']) == retained['sha256'], 'targeted reference receipt changed')
    for comparison, effects in summary['effects'].items():
        arm, base = comparison.split('_minus_')
        for metric,effect in effects.items():
            close(effect['delta_pp'],100*(summary['uqis_macro'][arm][metric]-summary['uqis_macro'][base][metric]),
                  'targeted paired effect differs')
    if manifest.get('proposals'):
        capacity=read(root/'proposal_capacity.json')['rows']
        require({r['query_id'] for r in capacity}==set(queries) and len(capacity)==67,
                'finite proposal capacity cohort changed')
        choices={r['query_id']:r for path in seal['report_hashes'] for r in read(path)['queries']}
        scored={a:{r['query_id']:r['iou'] for r in read(root/f'uqis_{a}_selections.json')['per_query']}
                for a in ['baseline']+item['arms']}
        for row in capacity:
            qid=row['query_id']
            close(row['proposal_iou']['joint'],scored['baseline'][qid],'capacity original response')
            close(row['oracle_iou'],max(row['proposal_iou'].values()),'finite pool maximum')
            for arm in item['arms']:
                proposal=choices[qid]['arms'][arm]['chosen_proposal']
                close(row['proposal_iou'][proposal],scored[arm][qid],'selected complete response changed')
        scenes={r['scene'] for r in capacity}
        oracle=sum(sum(r['oracle_iou'] for r in capacity if r['scene']==s)/
                   sum(r['scene']==s for r in capacity) for s in scenes)/len(scenes)
        close(oracle,summary['finite_pool_oracle_iou'],'finite-pool scene macro')
    rows=[]
    for arm in item['arms']:
        for metric,value in summary['uqis_macro'][arm].items():
            effect=summary['effects'][arm+'_minus_baseline'][metric]
            rows.append(dict(experiment_id=item['id'],arm=arm,benchmark='uqis',metric=metric,
                value_percent=100*value,reference_percent=100*summary['uqis_macro']['baseline'][metric],
                delta_pp=effect['delta_pp'],descriptive_ci_low_pp=effect['scene_bootstrap95_pp'][0],
                descriptive_ci_high_pp=effect['scene_bootstrap95_pp'][1],
                intervention_applied=metric.startswith('image_'),all31_fresh_inference=False,
                independent_test=False,status=item['status'],summary_sha256=sha(root/'summary.json')))
    versions = source_availability(repo,dict(prediction=manifest['source_hashes']),index)
    return dict(**item,source_versions=versions,summary_sha256=sha(root/'summary.json'),
                prediction_seal_sha256=sha(root/'predictions_complete.json'),
                inference_jobs=9,freshly_scored_queries=268,intervention_queries=67),rows


def inspect(repo=REPO):
    registry = read(repo / REGISTRY)
    recipe = read(repo / registry['recipe'])
    check_cohort(recipe['jobs'], recipe['jobs'], registry['cohort'])
    study, paper = repo / registry['study_root'], repo / registry['paper_root']
    reference = read(repo / registry['reference_summary'])
    index = source_index(study)
    experiments, metrics, warnings = [], [], []
    for experiment in registry['experiments']:
        run = experiment['run']
        root = study / run
        manifest, summary = read(root / 'manifest.json'), read(root / 'summary.json')
        feedback = read(paper / f'{run}_feedback.json')
        check_cohort(manifest['jobs'], recipe['jobs'], registry['cohort'])
        require(manifest['arms'] == experiment['arms'], f'{run}: arms changed')
        expected_source = sorted([j for j in manifest['jobs'] if j['benchmark'] != 'uqis'], key=job_key)
        require(sorted([r['job'] for r in summary['rows']], key=job_key) == expected_source,
                f'{run}: scored jobs differ from prediction jobs')
        require(summary['complete'] and summary['all_prediction_jobs'] == 31
                and summary['uqis']['queries'] == registry['uqis_queries'], f'{run}: incomplete')
        require(summary['selection_heads_frozen'] and not summary['default_changed']
                and not summary['independent_test'], f'{run}: changed scientific status')
        require(feedback['summary_sha256'] == sha(root / 'summary.json'), f'{run}: stale feedback')
        evaluation = check_seal(repo, study, run, manifest['jobs'])
        check_aggregates(root, experiment, summary, feedback, reference['macro'], registry['cohort'])
        stages = dict(prediction=manifest['source_hashes'], scoring=evaluation['scoring_source_hashes'])
        if (root / 'prediction_resume.json').exists():
            stages['prediction_resume'] = read(root / 'prediction_resume.json')['source_hashes']
        versions = source_availability(repo, stages, index)
        missing = [v for v in versions if v['availability'] == 'missing_version']
        if missing:
            warnings.append(f'{experiment["id"]}: {len(missing)} source-version receipts lack matching current/archive text; exact replay not certified')
        experiments.append(dict(**experiment, jobs=31, metrics=16, source_versions=versions,
            recorded_source_versions_available=not missing, summary_sha256=sha(root / 'summary.json'),
            feedback_sha256=sha(paper / f'{run}_feedback.json'),
            prediction_seal_sha256=sha(root / 'predictions_complete.json')))
        metrics.extend(result_rows(experiment, feedback, sha(root / 'summary.json')))
    evidence = read(paper / 'evidence.json')
    for path, receipt in evidence['source_files'].items():
        require(sha(repo / path) == receipt['sha256'], f'changed historical evidence: {path}')
    method = read(paper / 'method_manifest.json')
    for binding in method['code_bindings']:
        require(sha(repo / binding['path']) == binding['sha256'], f'stale method binding: {binding["symbol"]}')
    for path, digest in method['source_sha256'].items():
        require(sha(repo / path) == digest, f'stale method source: {path}; run refresh')
    providers = provider_audit(repo, recipe)
    require(method['effective_provider_audit'] == providers, 'stale effective-provider method attribution')
    failures = []
    for item in registry['failed_attempts']:
        path = study / item['receipt']
        require(path.is_file(), f'missing failure receipt: {path}')
        require(not any((study / item['root'] / f).exists() for f in ['summary.json', 'predictions_complete.json']),
                f'{item["id"]}: failed attempt now has scientific completion; reclassify')
        failures.append(dict(**item, sha256=sha(path)))
    audits = []
    for item in registry['audits']:
        if item['id'] == 'A24':
            from .score_precision_audit import check_score_precision_audit
            audits.append(check_score_precision_audit(repo,repo/item['study_root'],item,recipe['jobs']))
            continue
        record = dict(**item, sha256=sha(study / item['path']))
        if item['id'] == 'A06':
            record['verification'] = check_anchor_audit(study)
        elif item['id'] == 'A07':
            capacity = read(study / item['path'])
            transfer = study/'response_replacement_transfer_v1'
            predictions = read(study/'response_replacement_capacity_v1/predictions_complete.json')
            actual = {r['query_id']: r['iou'] for r in read(transfer/'uqis_full_sets.json')['per_query']}
            require(capacity['complete'] and capacity['exact_candidate_replay'] and capacity['diagnostic_only'],
                    'A07: candidate pool replay/diagnostic status changed')
            require(len(capacity['rows']) == predictions['queries'] == 134 and not predictions['ground_truth_read'],
                    'A07: candidate cohort changed')
            for row in capacity['rows']:
                values = row['candidate_iou']
                require(abs(max(values)-row['oracle_iou']) < 1e-12
                        and abs(values[0]-row['baseline_iou']) < 1e-12
                        and abs(values[1]-row['fixed_provider_iou']) < 1e-12
                        and abs(actual[row['query_id']]-row['selected_iou']) < 1e-12,
                        'A07: candidate utility differs from sealed evaluation')
            for modality, result in capacity['macro'].items():
                rows = [r for r in capacity['rows'] if r['modality'] == modality]
                scenes = sorted({r['scene'] for r in rows})
                for key in ('baseline_iou', 'fixed_provider_iou', 'selected_iou', 'oracle_iou'):
                    value = sum(sum(r[key] for r in rows if r['scene'] == s)/sum(r['scene'] == s for r in rows)
                                for s in scenes)/len(scenes)
                    require(abs(value-result[key]) < 1e-12, 'A07: scene aggregate changed')
                require(abs(result['selected_iou']-(result['baseline_iou']+result['available_gain']-result['decision_regret'])) < 1e-12,
                        'A07: finite-pool regret identity changed')
            record['verification'] = dict(exact_pools=134, finite_pool_oracle_is_not_a_method=True)
        elif item['id'] == 'A08':
            record['verification'] = check_source_decision_audit(study, recipe['jobs'])
        elif item['id'] == 'A09':
            record['verification'] = check_cross_task_targets(study, recipe['jobs'])
        elif item['id'] == 'A15':
            audit = read(study/item['path'])
            require(audit['complete'] and audit['durable_dense_bytes'] == 0
                    and not audit['new_model_gain_claim'] and not audit['global_no_leakage_proof'],
                    'A15: audit scope changed')
            for family, count in [('lerf',4),('nvos',8),('scannet',10)]:
                require(len(audit[family]['rows']) == count, 'A15: incomplete cohort')
            require(sum(r['count'] for r in audit['lerf']['rows'].values()) == 208,
                    'A15: incomplete LERF projection audit')
            gaussian = [r for r in audit['scannet']['rows'].values() if 'gaussian' in r]
            require(len(gaussian) == 8 and all(r['pseudo_gt_hash_match']
                    and r['official_sample_mismatches'] == 0 for r in gaussian), 'A15: Gaussian disagreement')
            for row in audit['nvos']['rows'].values():
                require(row['diagnostic_grid_oracle']['oracle']
                        and not row['diagnostic_grid_oracle']['deployable']
                        and not row['published_readout_is_full_ludvig_reproduction'],
                        'A15: diagnostic promoted to method result')
            record['verification'] = dict(retained_projection_masks=208,gaussian_scenes=8,
                official_dense_samples=sum(r['official_dense_samples'] for r in gaussian),
                semantic_rerender_in_this_audit=False)
        elif item['id'] == 'A16':
            audit = read(study/item['path'])
            require(audit['complete'] and len(audit['rows']) == 22 and not audit['default_changed']
                    and not audit['global_training_leakage_proof'], 'A16: incorrect lineage scope')
            require(audit['scoring_audit_sha256'] == sha(study/'protocol_correctness_audit_v2/summary.json'),
                    'A16: scoring evidence changed')
            root = (study/item['path']).parent
            require(audit['image_sha256'] == sha(root/'nvos_error_diagnosis.png'), 'A16: error figure changed')
            for path in ('scripts/research/audit_tmm_protocol_correctness.py',
                         'scripts/research/audit_tmm_protocol_lineage.py'):
                snapshot = read(root/(Path(path).stem+'_source.json'))
                require(hashlib.sha256(snapshot['source_text'].encode()).hexdigest() == snapshot['sha256']
                        == sha(repo/path), 'A16: audit implementation receipt changed')
            record['verification'] = dict(source_jobs=12,provided_meshes=10,
                                           archived_audit_implementations=2)
        if item['id'] == 'A17':
            from .reference_study import check_inherited_readouts
            record['verification']=check_inherited_readouts(study)
        elif item['id'] == 'A18':
            from .reference_study import check_coverage_ranking
            record['verification']=check_coverage_ranking(study)
        elif item['id'] == 'A19':
            from .reference_study import check_inherited_readouts
            record['verification']=check_inherited_readouts(study,'nvos_v10_readout_factorial_v1')
        elif item['id'] == 'A20':
            from .source_readout_study import check_score_changes
            record['verification']=check_score_changes(study)
        elif item['id'] == 'A21':
            from .local_credit_study import check_local_credit_spatial
            record['verification']=check_local_credit_spatial(study)
        elif item['id'] == 'A22':
            from .local_credit_study import check_reference_scope
            record['verification']=check_reference_scope(study)
        elif item['id'] == 'A23':
            from .reference_study import check_exact_readout_capacity
            record['verification']=check_exact_readout_capacity(study)
        audits.append(record)
    targeted, targeted_metrics = [], []
    for item in registry.get('targeted_studies', []):
        record, rows = check_targeted_study(repo,study,item,recipe,reference,index)
        targeted.append(record);targeted_metrics.extend(rows)
    learned = []
    for item in registry.get('learning_studies', []):
        root = study / item['run']
        if not (root / 'summary.json').exists():
            learned.append(dict(**item, status='in_progress' if root.exists() else 'planned', complete=False))
            continue
        summary = read(root / 'summary.json')
        require(summary['complete'] and not summary.get('default_changed', False), f'{item["id"]}: invalid promotion state')
        manifest = read(root / 'manifest.json')
        snapshot = read(root / 'source_snapshot.json')
        for path, digest in manifest['source_hashes'].items():
            archived = snapshot[path]['text'] if isinstance(snapshot[path], dict) else snapshot[path]
            require(hashlib.sha256(archived.encode()).hexdigest() == digest,
                    f'{item["id"]}: archived implementation differs: {path}')
        if item['stage'] in ('native_category_responses', 'category_supervision'):
            scenes = sum(manifest['splits'].values(), [])
            require(len(scenes) == len(set(scenes)) == summary['rooms'] == 29,
                    f'{item["id"]}: incomplete category room cohort')
            require(not {s.split('_')[0] for s in scenes} & {j['name'].split('_')[0] for j in recipe['jobs']},
                    f'{item["id"]}: benchmark room prefix overlap')
            require(set(summary['report_hashes']) == {s+'.json' for s in scenes}, f'{item["id"]}: room reports missing')
            reports = []
            for name, digest in summary['report_hashes'].items():
                require(sha(root/name) == digest, f'{item["id"]}: changed category report')
                report = read(root/name); reports.append(report)
                require(report['scene'] in manifest['splits'][report['split']], f'{item["id"]}: split changed')
                if item['stage'] == 'native_category_responses':
                    require(sha(root/report['response_file']) == report['response_sha256']
                            and report['all_original_views'] and report['original_masks_unchanged']
                            and not report['labels_read'] and not report['high_dimensional_features_saved'],
                            f'{item["id"]}: native teacher/cache contract changed')
                else:
                    require(report['queries'] == len(report['rows']) == 4
                            and len(report['teachers']) == 3 and report['all_original_views']
                            and not report['benchmark_labels_used'] and report['new_mask_bytes'] == 0
                            and report['new_dense_prediction_bytes'] == 0
                            and not report['benchmark_gaussian_renderer_equivalence'],
                            f'{item["id"]}: category supervision scope changed')
                    for row in report['rows']:
                        require(row['complete_menu'] and row['no_classwise_selection']
                                and set(row['menu_utilities']) == {'19','15','10'}
                                and len(row['features']) == 3
                                and all(len(x) == 39 for x in row['features'])
                                and row['evidence']['query_teachers'] == 2
                                and row['observation_views_disjoint'] == (row['fold'] != 'full'),
                                f'{item["id"]}: incomplete menu or evidence contract')
            if item['stage'] == 'native_category_responses':
                require(not summary['utility_training_ready'] and not summary['source_category_stage_complete'],
                        f'{item["id"]}: native cache is not full utility supervision')
                require(sum(r['frames'] for r in reports) == summary['frames']
                        and sum(r['regions'] for r in reports) == summary['regions'], f'{item["id"]}: cache counts changed')
                details = dict(rooms=29, frames=summary['frames'], regions=summary['regions'], evaluation_result=False)
            else:
                require(summary['episodes'] == 116 and summary['native_and_source_supervision_ready'],
                        f'{item["id"]}: incomplete category utility corpus')
                details = dict(rooms=29, episodes=116, evaluation_result=False)
        elif item['stage'] == 'category_preparation':
            details=check_category_preparation(root,manifest,summary,recipe)
        elif item['stage'] == 'category_instance_transfer':
            details=check_multitask_transfer(study,root,manifest,summary,recipe,reference)
        elif item['stage'] == 'transfer_diagnosis':
            for path,digest in manifest['input_hashes'].items():
                require(sha(path)==digest,'transfer diagnosis: changed input results')
            require(summary['diagnostic_only'] and summary['all_oracles_are_whole_answers']
                    and len(summary['category_rows'])==10 and len(summary['instance_rows'])==134,
                    'transfer diagnosis: scope changed')
            for row in summary['instance_rows']:
                close(row['selected_iou'],row['baseline_iou']+row['available_gain']-row['decision_regret'],
                      'transfer diagnosis: regret identity')
            details=dict(evaluation_result='post-seal capacity diagnosis; no new predictions')
        elif item['stage'] == 'utility_diagnosis':
            for path, digest in manifest['input_hashes'].items():
                require(sha(path) == digest, 'utility diagnosis: changed sealed input')
            require(summary['diagnostic_only'] and not summary['benchmark_labels_used']
                    and not manifest['target_benchmark_ground_truth_read']
                    and not manifest['model_or_margin_changed']
                    and summary['durable_dense_prediction_bytes'] == 0,
                    'utility diagnosis: diagnostic-only contract changed')
            require(summary['category_fullmap_range']['fit_candidates'] == 66
                    and summary['category_fullmap_range']['target_candidates'] == 30,
                    'utility diagnosis: incomplete category signature cohort')
            for split, arms in summary['independent_outputs'].items():
                require(split in ('calibration', 'validation')
                        and set(arms) == {'full', 'query_blind', 'permuted', 'unrestricted'},
                        'utility diagnosis: incomplete fixed-policy controls')
                for results in arms.values():
                    require(set(results) == {'text', 'image', 'category'},
                            'utility diagnosis: incomplete task coverage')
                    for result in results.values():
                        values = result['scene_macro']
                        close(values['selected'], values['baseline']
                              + values['available_gain'] - values['regret'],
                              'utility diagnosis: regret identity')
            details = dict(evaluation_result='fixed-model diagnosis; no target benchmark labels',
                           target_category_signatures=30)
        elif item['stage'] == 'residual_diagnosis':
            for path,digest in manifest['input_hashes'].items():
                require(sha(path)==digest,'residual diagnosis: changed sealed input')
            require(summary['diagnostic_only'] and summary['null_mse_is_not_accuracy']
                    and summary['all_control_utilities_checked_against_fresh_scores']
                    and not manifest['parameters_changed'] and not manifest['new_decisions']
                    and manifest['benchmark_labels_used'], 'residual diagnosis: scope changed')
            require(set(summary['outputs'])=={'calibration','validation','benchmark'},
                    'residual diagnosis: missing split')
            for split,arms in summary['outputs'].items():
                require(set(arms)=={'full','query_blind','permuted','unrestricted','unregularized'},
                        'residual diagnosis: missing control')
                for tasks in arms.values():
                    require(set(tasks)=={'text','image','category'},'residual diagnosis: incomplete tasks')
                    for task,result in tasks.items():
                        d=result['decomposition']
                        close(d['available_gain'],d['ranking_regret']+d['retention_cost']+d['actual_gain'],
                              'residual diagnosis: error decomposition')
                        if split=='benchmark':
                            require(result['guarded']['queries']==(10 if task=='category' else 67),
                                    'residual diagnosis: incomplete target decision cohort')
            details=dict(evaluation_result='post-seal ranking/retention diagnosis; no new decisions')
        elif item['stage'] == 'multitask_training':
            seal = read(root/'model_seal.json')
            require(sha(root/'model.joblib') == seal['model_sha256'] == summary['checkpoint_sha256']
                    and sha(root/'calibration.json') == seal['calibration_sha256'], f'{item["id"]}: changed trained policy')
            require(not seal['benchmark_feedback_used'] and not seal['development_feedback_used']
                    and summary['all_five_transfer_pending'], f'{item["id"]}: training mislabeled as benchmark transfer')
            for result in read(root/'calibration.json').values():
                require(len(result['selection']['per_scene']) == 15
                        and min(result['selection']['per_scene'].values()) >= -1e-12,
                        f'{item["id"]}: shared multi-output calibration guard failed')
            for path, digest in manifest['input_hashes'].items():
                require(sha(path) == digest, f'{item["id"]}: supervision changed')
            require(all(v['same_menu_decisions'] for v in summary['development'].values()),
                    f'{item["id"]}: category submenus use different candidates')
            if item['id'] == 'L09':
                design = read(root/'design_seal.json')
                require(not design['parent_complete_at_design']
                        and not design['parent_benchmark_complete_at_design'],
                        'L09: architecture control was not fixed before parent results')
                require(design['source_hashes'] == manifest['source_hashes'],
                        'L09: predeclared architecture implementation changed')
            if item['id'] == 'L10':
                require(sha(root/'fit_diagnostics.json')==summary['fit_diagnostics_sha256'],
                        'L10: changed cross-fitting receipt')
                fit=read(root/'fit_diagnostics.json');rooms=set(manifest['splits']['fit'])
                held=[]
                require(len(fit['folds'])==5 and fit['room_count']==len(rooms)==22,
                        'L10: room folds incomplete')
                for fold in fit['folds']:
                    a,b=set(fold['fit_rooms']),set(fold['held_rooms'])
                    require(not a&b and a|b==rooms and b, 'L10: room leakage in structural prior')
                    held.extend(fold['held_rooms'])
                require(Counter(held)==Counter(rooms), 'L10: rooms not held out exactly once')
                close(fit['original_weight'],fit['augmented_weight'],'L10: task weight mass changed')
                require(fit['full_trees']==fit['no_null_control_trees']==2*fit['prior_trees']
                        and not fit['benchmark_labels_used'], 'L10: unmatched control or target leakage')
                require(set(read(root/'calibration.json'))=={'full','query_blind','unregularized'},
                        'L10: missing independently calibrated no-null control')
            details = dict(development_gate=summary['development_gate'],
                           queries=summary['training_queries'], evaluation_result='opened independent-room development')
        elif item['stage'] == 'supervision':
            scenes = sum(manifest['splits'].values(), [])
            require(len(scenes) == len(set(scenes)) and not set(scenes) & {j['name'] for j in recipe['jobs']},
                    f'{item["id"]}: independent image scenes overlap')
            counts = {}
            for scene in scenes:
                report = read(root / (scene+'.json')); counts[scene] = report['queries']
                require(report['queries'] == len(report['rows']) > 0
                        and report['original_teachers_and_masks']
                        and report['reference_views_excluded_from_all_providers']
                        and not report['benchmark_labels_used'] and not report['dense_predictions_retained'],
                        f'{item["id"]}: invalid independent image episode')
                require(not set(report['retained_source_frames']) & set(report['excluded_sensor_frames'])
                        and set(report['reference_frames']) <= set(report['excluded_sensor_frames']),
                        f'{item["id"]}: query view leakage')
                require(all(row['evidence']['query_teachers'] == 2 for row in report['rows']),
                        f'{item["id"]}: an image teacher disappeared')
            require(counts == summary['scenes'], f'{item["id"]}: supervision counts changed')
            details = dict(queries=sum(counts.values()), rooms=len(counts), evaluation_result=False)
        elif item['stage'] in ('training', 'joint_training'):
            seal = read(root / 'model_seal.json')
            require(sha(root / 'model.joblib') == seal['model_sha256'] == summary['checkpoint_sha256'],
                    f'{item["id"]}: learned checkpoint changed')
            require(sha(root / 'calibration.json') == seal['calibration_sha256'], f'{item["id"]}: calibration changed')
            scenes = sum(manifest['splits'].values(), [])
            require(len(scenes) == len(set(scenes)) and not set(scenes) & {j['name'] for j in recipe['jobs']},
                    f'{item["id"]}: independent scene splits overlap')
            if item['stage'] == 'joint_training':
                roots = dict(text=study/manifest['config']['text_run'], image=study/manifest['image_run'])
                counts = {split: {m: sum(read(path / f'{s}.json')['queries'] for s in values)
                                  for m, path in roots.items()} for split, values in manifest['splits'].items()}
                require(not seal['benchmark_feedback_used'] and not seal['development_feedback_used'],
                        f'{item["id"]}: non-independent calibration')
                for r in read(root / 'calibration.json').values():
                    require(len(r['selection']['per_scene']) == 6
                            and min(r['selection']['per_scene'].values()) >= -1e-12,
                            f'{item["id"]}: modality-specific calibration harm')
                for path, digest in manifest['input_hashes'].items():
                    require(sha(path) == digest, f'{item["id"]}: changed joint supervision: {path}')
            else:
                counts = {split: sum(read(root / f'{s}.json')['queries'] for s in values)
                          for split, values in manifest['splits'].items()}
            require([counts[k] for k in ('fit', 'calibration', 'validation')] ==
                    [summary[k] for k in ('training_queries', 'calibration_queries', 'development_queries')],
                    f'{item["id"]}: query counts differ')
            details = dict(queries=counts, development_gate=summary['development_gate'])
        else:
            seal = read(root / 'predictions_complete.json')
            require(seal['intervention_queries'] == 134 and not seal['ground_truth_read'],
                    f'{item["id"]}: transfer cohort/seal changed')
            for scene, digest in seal['report_hashes'].items():
                require(sha(root / (scene+'.json')) == digest, f'{item["id"]}: transfer report changed')
            require(summary['freshly_scored_queries'] == 268 and not summary['all31_fresh_inference'],
                    f'{item["id"]}: transfer scope changed')
            training = study / manifest['training_run']
            require(sha(training / 'model.joblib') == manifest['model_sha256'], f'{item["id"]}: transfer model changed')
            details = dict(intervention_queries=134, scored_queries=268,
                           other_benchmarks='transfer_pending', uqis_macro=summary['uqis_macro'])
        learned.append(dict(**item, status='completed_candidate', complete=True,
                            summary_sha256=sha(root / 'summary.json'), **details))
    from .reference_study import check_reference_study
    from .source_readout_study import check_source_readout_study
    from .local_credit_study import check_local_credit_study
    from .image_response_study import check_image_response_study
    from .native_writeback_study import check_native_writeback_study
    from .signed_affinity_study import check_signed_affinity_study
    from .direct_affinity_study import check_direct_affinity_study
    from .instance_extent_study import check_instance_extent_study
    from .reference_completion_study import check_reference_completion_study
    from .reference_delta_study import check_reference_delta_study
    from .reference_admission_study import check_reference_admission_study
    from .multiview_completion_study import check_multiview_completion_study
    from .target_redecode_study import check_target_redecode_study
    checkers={'N01':check_reference_study,'N02':check_source_readout_study,'N03':check_local_credit_study,
              'N04':check_image_response_study,'N05':check_image_response_study,'N06':check_image_response_study,'N07':check_image_response_study,'N08':check_native_writeback_study,'N09':check_signed_affinity_study,'N10':check_direct_affinity_study,'N11':check_instance_extent_study,'N12':check_reference_completion_study,'N13':check_reference_delta_study,'N14':check_reference_admission_study,'N15':check_multiview_completion_study,'N16':check_target_redecode_study}
    reference_studies=[checkers[item['id']](repo,repo/item.get('study_root',registry['study_root']),item,recipe['jobs'])
                       for item in registry.get('reference_studies',[])]
    return dict(schema='radiogs.response_study.catalog.v1', method=registry['method'],
        scope='retained report integrity and aggregate checks; no fresh inference or GT re-scoring',
        registry_sha256=sha(repo / REGISTRY), recipe_sha256=sha(repo / registry['recipe']),
        complete_studies=len(experiments), jobs_per_study=31, uqis_queries=268,
        historical_sources_verified=len(evidence['source_files']), scientific_default='retained v10',
        independent_test=False, universal_admission_validated=False, warnings=warnings,
        reference=dict(path=registry['reference_summary'], sha256=sha(repo / registry['reference_summary']),
                       macro_percent={f: {k: 100*v for k, v in m.items()} for f, m in reference['macro'].items()}),
        experiments=experiments, failed_attempts=failures, audits=audits,
        targeted_studies=targeted,targeted_metrics=targeted_metrics, learning_studies=learned,
        reference_studies=reference_studies,
        effective_provider_audit=providers, metrics=metrics)


def export(repo=REPO):
    catalog = inspect(repo)
    target = repo / read(repo / REGISTRY)['paper_root'] / 'artifacts'
    target.mkdir(exist_ok=True)
    (target / 'study_catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    with (target / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog['metrics'][0]))
        writer.writeheader()
        writer.writerows(catalog['metrics'])
    if catalog['targeted_metrics']:
        with (target / 'targeted_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(catalog['targeted_metrics'][0]))
            writer.writeheader();writer.writerows(catalog['targeted_metrics'])
    return catalog


def replay_plan(experiment_id, run, repo=REPO):
    registry = read(repo / REGISTRY)
    experiment = next((e for e in registry['experiments'] + registry.get('targeted_studies', [])
                       if e['id'] == experiment_id), None)
    require(experiment is not None, f'unknown experiment: {experiment_id}')
    require(experiment['launch_enabled'], experiment['launch_note'])
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', run) is not None,
            'run must be a simple name of at most 80 characters')
    study = repo / registry['study_root']
    require(not any(p.name == run or p.name.startswith(run + '_') for p in study.iterdir()),
            'run already exists or collides with existing child outputs')
    manifest = read(study / experiment['run'] / 'manifest.json')
    check_cohort(manifest['jobs'], read(repo / registry['recipe'])['jobs'], registry['cohort'])
    for path, digest in manifest['source_hashes'].items():
        require(sha(repo / path) == digest, f'historical runner code differs: {path}; register a new experiment')
    return dict(command=[sys.executable, experiment['runner'], '--run', run, *experiment['runner_args']],
        cwd=str(repo), jobs=len(manifest.get('intervention_jobs',manifest['jobs'])),
        uqis_queries=268, storage=registry['storage'],
        prediction_storage=experiment['prediction_storage'], note=experiment['launch_note'],
        scientific_default_changed=False, executed=False)
