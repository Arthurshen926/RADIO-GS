"""Seal a failed cross-input screen and the complete physical-metric LERF cohort."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'output/query_field_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def cold(root, manifest):
    smoke = read(root/'runtime_smoke.json')
    if (smoke['field_sha256'] != manifest['files']['field.pt']['sha256'] or
            not all(smoke['exact_replay'].values()) or smoke['forbidden_open_attempts']):
        raise ValueError('cold replay failure or field mismatch')
    return {k: v['seconds'] for k, v in smoke['results'].items()}


def run():
    target = ROOT/'docs/experiments/2026-10-07-native-local-evidence.json'
    if target.exists():
        raise FileExistsError('do not overwrite a sealed round')
    reference = read(ROOT/'output/query_conditioned_membership_20261006/image_convergence/system/summary.json')
    v10 = {r['job']['name']: r['metrics'] for r in reference['rows'] if r['job']['benchmark'] == 'lerf'}
    config = read(ROOT/'configs/query_field/native_local_metric_study.json')
    errors = read(RUNS/'native_local_lerf_v1/subset_errors.json')['reports']
    result = dict(schema='radio_gs.native_local_evidence.v1', parent_commit='b3b934b',
        independent_final_test=False, v10_contract_equivalent=False, compact_student=False,
        candidate_scenes={}, physical_lerf_four_scene={}, artifacts={}, code={})
    artifacts = []
    for scene in ['figurines', 'ramen']:
        previous = RUNS/f'{scene}_fitted_metric_v1'
        before = read(previous/'development_trained.json')
        prior_metrics = read(RUNS/f'spatial_metric_lerf_v1/{scene}_fitted_metric/evaluation/evaluation.json')['metrics']
        arms = {}
        for job in [j for j in config['jobs'] if j['scene'] == scene]:
            root = (ROOT/job['field']).parents[1]
            after = read(root/'development_trained.json')
            keys = ['frame', 'region', 'mode', 'scale_condition', 'status', 'reason']
            if [[r.get(k) for k in keys] for r in before['rows']] != [[r.get(k) for k in keys] for r in after['rows']]:
                raise ValueError('teacher-region roster/status differs')
            public = RUNS/'native_local_lerf_v1'/job['name']
            prediction = read(public/'prediction.json')
            manifest = read(root/'models/trained/manifest.json')
            metrics = read(public/'evaluation/evaluation.json')['metrics']
            if prediction['field_sha256'] != manifest['files']['field.pt']['sha256']:
                raise ValueError('public field mismatch')
            for key in ['iou', 'ap']:
                if prior_metrics['native_generic_relevance'][key] != metrics['native_generic_relevance'][key]:
                    raise ValueError('native response mismatch')
            old = prior_metrics['shared_membership']['selected']['miou']
            new = metrics['shared_membership']['selected']['miou']
            private = {mode: dict(before=before['summary'][mode+'/default']['mean_actual_subset_iou'],
                                 after=after['summary'][mode+'/default']['mean_actual_subset_iou'])
                       for mode in ['region_image', 'point_2d', 'point_3d']}
            gate = dict(text_subset_improves=new > old,
                        **{mode+'_nonregression': x['after'] >= x['before'] for mode, x in private.items()})
            arms[job['arm']] = dict(text_subset_before=old, text_subset_after=new,
                v10_gap_after_pp=100*(v10[scene]['selected_projection_iou']-new),
                member_semantic_iou=metrics['shared_membership']['iou'], private=private,
                gate=gate, local_gate_passed=all(gate.values()),
                errors={k: v for k, v in errors[job['name']]['controls']['shared_membership'].items() if k != 'rows'})
            if job['arm'] == 'fitted_metric':
                arms[job['arm']]['single_fixture_readout_seconds'] = cold(root, manifest)
                artifacts.extend([root/'runtime_smoke.json', root/'models/training.json'])
            artifacts.extend([root/'development_trained.json', root/'models/trained/manifest.json',
                              public/'prediction.json', public/'evaluation/evaluation.json'])
        result['candidate_scenes'][scene] = dict(arms=arms,
            source_exclusion_audit=read(RUNS/f'{scene}_exclusion_audit_v1.json')['summary'],
            scope_bound=read(RUNS/f'{scene}_native_scope_bound.json')['groups'],
            readout_controls={k: v['selected']['miou'] for k, v in read(RUNS/f'physical_readout_study_v1/{scene}/evaluation/evaluation.json')['metrics'].items()})
        for name in [f'{scene}_exclusion_audit_v1.json', f'{scene}_native_point_deltas.json', f'{scene}_native_scope_bound.json']:
            artifacts.append(RUNS/name)
        artifacts.extend([RUNS/f'physical_readout_study_v1/{scene}/prediction.json',
                          RUNS/f'physical_readout_study_v1/{scene}/evaluation/evaluation.json'])
    result['passing_arms'] = [arm for arm in ['frozen_metric', 'fitted_metric']
                            if all(row['arms'][arm]['local_gate_passed'] for row in result['candidate_scenes'].values())]
    if result['passing_arms']:
        raise ValueError('this closure requires completing the declared third-scene gate for a passing arm')
    result['promoted_local_metric'] = False
    for scene in ['figurines', 'ramen', 'teatime', 'waldo_kitchen']:
        root = RUNS/f'{scene}_fitted_metric_v1'
        public = (RUNS/'spatial_metric_lerf_v1'/f'{scene}_fitted_metric' if scene in ['figurines', 'ramen'] else
                  RUNS/'spatial_metric_teatime_lerf_v1/teatime_fitted_metric' if scene == 'teatime' else
                  RUNS/'waldo_kitchen_physical_lerf_v1/waldo_kitchen')
        evaluation = read(public/'evaluation/evaluation.json')
        manifest = read(root/'models/trained/manifest.json')
        if manifest['spec'].get('local_metric', 'none') != 'none':
            raise ValueError('four-scene cohort must use the same physical member method')
        metrics = evaluation['metrics']
        result['physical_lerf_four_scene'][scene] = dict(
            actual_subset_iou=metrics['shared_membership']['selected']['miou'],
            native_semantic_iou=metrics['native_generic_relevance']['iou'],
            member_semantic_iou=metrics['shared_membership']['iou'],
            query_view_cases=metrics['shared_membership']['observations'], v10=v10[scene],
            bytes=manifest['files']['field.pt']['bytes'], elements=manifest['elements'],
            observed=manifest['observed'], eligible_anchors=manifest['eligible_anchors'],
            private=read(root/'development_trained.json')['summary'], single_fixture_readout_seconds=cold(root, manifest))
        artifacts.extend([public/'prediction.json', public/'evaluation/evaluation.json',
            root/'models/trained/manifest.json', root/'models/training.json', root/'runtime_smoke.json',
            root/'development_trained.json', root/'observations/observations.json'])
    rows = list(result['physical_lerf_four_scene'].values())
    result['physical_four_scene_macro'] = {key: sum(row[key] for row in rows)/len(rows)
        for key in ['actual_subset_iou', 'native_semantic_iou', 'member_semantic_iou']}
    result['four_scene_v10_subset_gap_pp'] = 100*(sum(r['v10']['selected_projection_iou'] for r in rows)/4-result['physical_four_scene_macro']['actual_subset_iou'])
    for name in ['native_local_metric_tests.log', 'native_local_package.json', 'native_local_integrity.json',
                 'native_local_original_worktree_check.json', 'native_local_lerf_v1/subset_errors.json',
                 'waldo_kitchen_physical_lerf_v1/subset_errors.json']:
        artifacts.append(RUNS/name)
    artifacts.append(ROOT/'docs/experiments/2026-10-07-native-local-member-protocol.md')
    for path in (ROOT/'configs/query_field').glob('*.json'):
        if any(word in path.name for word in ['native_local', 'physical_readout', 'exclusion_audit', 'waldo_kitchen']):
            artifacts.append(path)
    for path in artifacts:
        result['artifacts'][str(path.relative_to(ROOT))] = receipt(path)
    for path in list((ROOT/'radio_gs/query_field').glob('*.py')) + [Path(__file__), ROOT/'tests/test_query_field_local_metric.py']:
        result['code'][str(path.relative_to(ROOT))] = receipt(path)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['passing_arms', 'physical_four_scene_macro', 'four_scene_v10_subset_gap_pp']}, indent=2))


if __name__ == '__main__':
    run()
