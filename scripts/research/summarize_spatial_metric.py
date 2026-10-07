"""Seal the fixed physical-scale study, retaining failures and v10 contract gaps."""
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


def collect(config, public_root, scenes):
    rows = {}
    errors = read(public_root/'subset_errors.json')['reports']
    baseline = read(ROOT/'output/query_conditioned_membership_20261006/image_convergence/system/summary.json')
    baseline = {r['job']['name']: r['metrics'] for r in baseline['rows'] if r['job']['benchmark'] == 'lerf'}
    for scene in scenes:
        reference = RUNS/f'{scene}_supported_anchor_v1'
        before = read(reference/'development_trained.json')
        public_before = (RUNS/'supported_anchor_teatime_lerf_v1/teatime_supported' if scene == 'teatime'
                         else RUNS/'supported_anchor_lerf_v1'/scene)
        previous = read(public_before/'evaluation/evaluation.json')['metrics']
        arms = {}
        for job in [j for j in config['jobs'] if j['scene'] == scene]:
            field_root = (ROOT/job['field']).parents[1]
            after = read(field_root/'development_trained.json')
            current = read(public_root/job['name']/'evaluation/evaluation.json')['metrics']
            manifest = read(ROOT/job['field']/'manifest.json')
            prediction = read(public_root/job['name']/'prediction.json')
            if prediction['field_sha256'] != manifest['files']['field.pt']['sha256']:
                raise ValueError('prediction field mismatch')
            if job['arm'] == 'fitted_metric':
                smoke = read(field_root/'runtime_smoke.json')
                if (not all(smoke['exact_replay'].values()) or smoke['forbidden_open_attempts']
                        or smoke['field_sha256'] != prediction['field_sha256']):
                    raise ValueError('fitted metric source-denied cold replay failed')
            roster = ['frame', 'region', 'mode', 'scale_condition', 'status', 'reason']
            if [[r.get(k) for k in roster] for r in before['rows']] != [[r.get(k) for k in roster] for r in after['rows']]:
                raise ValueError('private evaluation roster/status differs')
            for key in ['iou', 'ap']:
                if current['native_generic_relevance'][key] != previous['native_generic_relevance'][key]:
                    raise ValueError('native readout mismatch')
            value = current['shared_membership']['selected']['miou']
            old = previous['shared_membership']['selected']['miou']
            private = {mode: dict(before=before['summary'][mode+'/default']['mean_actual_subset_iou'],
                                 after=after['summary'][mode+'/default']['mean_actual_subset_iou'])
                       for mode in ['point_2d', 'point_3d', 'region_image']}
            gate = dict(text_subset_improves=value > old,
                        **{mode+'_nonregression': values['after'] >= values['before'] for mode, values in private.items()})
            arms[job['arm']] = dict(actual_text_subset_before=old, actual_text_subset_after=value,
                actual_text_delta_pp=100*(value-old), private=private, gate=gate, local_gate_passed=all(gate.values()),
                v10_gap_before_pp=100*(baseline[scene]['selected_projection_iou']-old),
                v10_gap_after_pp=100*(baseline[scene]['selected_projection_iou']-value),
                native_semantic_iou=current['native_generic_relevance']['iou'],
                member_semantic_iou=current['shared_membership']['iou'],
                precision_recall={k: v for k, v in errors[job['name']]['controls']['shared_membership'].items() if k != 'rows'},
                spec=manifest['spec'], total_file_bytes=manifest['files']['field.pt']['bytes'],
                eligible_anchors=manifest['eligible_anchors'], field_sha256=prediction['field_sha256'])
        rows[scene] = dict(v10=baseline[scene], arms=arms)
    return rows


def run():
    target = ROOT/'docs/experiments/2026-10-07-spatial-metric-evidence.json'
    if target.exists():
        raise FileExistsError('do not overwrite a sealed round')
    config = read(ROOT/'configs/query_field/spatial_metric_study.json')
    scenes = collect(config, RUNS/'spatial_metric_lerf_v1', ['figurines', 'ramen'])
    passing = [arm for arm in ['frozen_metric', 'fitted_metric']
               if all(r['arms'][arm]['local_gate_passed'] for r in scenes.values())]
    nominated = max(passing, key=lambda arm: sum(r['arms'][arm]['actual_text_subset_after'] for r in scenes.values())) if passing else None
    artifacts = []
    third = ROOT/'configs/query_field/spatial_metric_teatime_study.json'
    if third.exists():
        third_config = read(third)
        if nominated is None or any(j['arm'] != nominated for j in third_config['jobs']):
            raise ValueError('third scene is not the predeclared nominated arm')
        scenes.update(collect(third_config, RUNS/'spatial_metric_teatime_lerf_v1', ['teatime']))
        artifacts.extend([third, RUNS/'spatial_metric_teatime_lerf_v1/subset_errors.json'])
    result = dict(schema='radio_gs.spatial_metric_evidence.v1', parent_commit='990d2c9', scenes=scenes,
        two_scene_passing_arms=passing, nominated_arm=nominated,
        confirmed_three_scene_gate=nominated is not None and 'teatime' in scenes and scenes['teatime']['arms'][nominated]['local_gate_passed'],
        independent_final_test=False, v10_contract_equivalent=False, benchmark_replacement=False,
        extra_teachers=False, additional_3d_labels=False, artifacts={}, code={})
    artifacts.extend([ROOT/'configs/query_field/spatial_metric_study.json',
        ROOT/'docs/experiments/2026-10-07-physical-scale-member-protocol.md',
        RUNS/'spatial_metric_lerf_v1/subset_errors.json', RUNS/'spatial_metric_tests.log',
        RUNS/'spatial_metric_compatibility_tests.log',
        RUNS/'spatial_metric_package.json', RUNS/'spatial_metric_integrity.json'])
    for scene, row in scenes.items():
        artifacts.extend([ROOT/f'configs/query_field/{scene}_spatial_metric.json',
                          RUNS/f'{scene}_18views_v1/observations/observations.json',
                          RUNS/f'{scene}_supported_anchor_v1/models/trained/manifest.json',
                          RUNS/f'{scene}_supported_anchor_v1/development_trained.json'])
        for arm in row['arms']:
            root = RUNS/f'{scene}_{arm}_v1'
            public = RUNS/('spatial_metric_teatime_lerf_v1' if scene == 'teatime' else 'spatial_metric_lerf_v1')/f'{scene}_{arm}'
            artifacts.extend([root/'models/trained/manifest.json', root/'development_trained.json',
                              public/'prediction.json', public/'evaluation/evaluation.json'])
            for optional in [root/'runtime_smoke.json', root/'models/training.json']:
                if optional.exists(): artifacts.append(optional)
    for name in ['spatial_metric_original_worktree_check.json', 'spatial_metric_teatime_integrity.json']:
        path = RUNS/name
        if path.exists(): artifacts.append(path)
    for path in artifacts:
        result['artifacts'][str(path.relative_to(ROOT))] = receipt(path)
    for path in sorted((ROOT/'radio_gs/query_field').glob('*.py')) + [Path(__file__), ROOT/'tests/test_query_field_spatial_metric.py']:
        result['code'][str(path.relative_to(ROOT))] = receipt(path)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(nominated=nominated, scenes=scenes), indent=2))


if __name__ == '__main__':
    run()
