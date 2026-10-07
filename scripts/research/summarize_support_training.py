"""Seal all fixed support-training arms, their failed gates and replay controls."""
import json
import os
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/support_training_v1')
BASE = Path('output/query_field_20261007')
read = lambda path: json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def main():
    study = read('configs/query_field/support_training_study.json')
    nvos_config = read('configs/query_field/support_training_nvos.json')
    evaluation = read(ROOT/'nvos/evaluation.json')
    nvos_parent = read(BASE/'nvos_fern_v1/public/evaluation.json')
    baseline = dict(figurines=BASE/'figurines_fitted_metric_v1', ramen=BASE/'ramen_fitted_metric_v1', fern=BASE/'nvos_fern_v1')
    cold = read(ROOT/'cold_replay.json')
    assert len(cold['comparisons']) == len(nvos_config['jobs']) == 4
    assert cold['original_open_rejected'] and all(all(row['exact'].values()) for row in cold['comparisons'])
    traces, training, development, lerf, nvos, gates = {}, {}, {}, {}, {}, {}
    artifacts = [ROOT/name for name in ['launch.json', 'protocol_at_launch.md', 'tests.log',
        'single_replay.json', 'original_replay_sources.json', 'training_repeatability.json',
        'cold_replay.json', 'nvos/evaluation.json', 'subset_errors.json']]
    artifacts += [ROOT/f'{scene}_concentration.json' for scene in baseline]
    for job in study['fits']:
        name, scene = job['name'], job['scene']
        root = Path(job['output'])
        fitted = read(root/'models/training.json')
        assert fitted['geometry_unchanged'] and fitted['native_coordinates_unchanged']
        assert sum(fitted['support_counts'].values()) == 2000
        traces[name] = [(r['frame'], r['region'], r['first_pixel']) for r in fitted['support_trace']]
        parent_training = read(baseline[scene]/'models/training.json')
        assert [(r['frame'], r['region']) for r in fitted['history']] == [(r['frame'], r['region']) for r in parent_training['history']]
        manifest = read(root/'models/trained/manifest.json')
        parent = read(baseline[scene]/'models/trained/manifest.json')
        # Pre-locality exports omit this field; the loader's default is none.
        assert {'local_metric':'none', **manifest['spec']} == {'local_metric':'none', **parent['spec']}
        assert manifest['tensor_bytes'] == parent['tensor_bytes']
        assert sha256(root/'models/trained/field.pt') == manifest['files']['field.pt']['sha256']
        training[name] = {k:fitted[k] for k in ['support_counts', 'member_forward_evaluations', 'training_episodes',
            'eligible_input_anchors', 'seconds', 'trained_deployment_bytes', 'native_coordinates_unchanged', 'geometry_unchanged']}
        current = read(root/'development_trained.json')['summary']
        previous = read(baseline[scene]/'development_trained.json')['summary']
        development[name] = {k:dict(cases=v['cases'], failed=v['failed'], actual_iou=v['mean_actual_subset_iou'],
            parent_actual_iou=previous[k]['mean_actual_subset_iou'],
            change_pp=100*(v['mean_actual_subset_iou']-previous[k]['mean_actual_subset_iou']))
            for k,v in current.items() if k.endswith('/default')}
        artifacts += [root/'models/training.json', root/'models/trained/manifest.json', root/'development_trained.json']
        if scene != 'fern':
            smoke = read(root/'runtime_smoke.json')
            assert all(smoke['exact_replay'].values())
            artifacts.append(root/'runtime_smoke.json')
    for scene in baseline:
        assert traces[f'{scene}_mixed_unsigned'] == traces[f'{scene}_mixed_signed']
    for job in study['jobs']:
        root = ROOT/'lerf'/job['name']
        pred = read(root/'prediction.json')
        assert pred['configuration'] == study and pred['native_response_bit_exact']
        assert sha256(root/'predictions.pt') == pred['predictions_sha256']
        metrics = read(root/'evaluation/evaluation.json')['metrics']
        previous = read(BASE/'spatial_metric_lerf_v1'/f"{job['scene']}_fitted_metric"/'evaluation/evaluation.json')['metrics']
        lerf[job['name']] = dict(native_semantic_iou=metrics['native_generic_relevance']['iou'],
            member_semantic_iou=metrics['shared_membership']['iou'],
            actual_iou=metrics['shared_membership']['selected']['miou'],
            parent_actual_iou=previous['shared_membership']['selected']['miou'],
            cases=metrics['shared_membership']['selected']['count'],
            missing=metrics['shared_membership']['selected']['missing'])
        artifacts += [root/'prediction.json', root/'evaluation/evaluation.json']
    for row in evaluation['results']:
        name = row['job']
        parent_name = 'trained_'+name.rsplit('_', 1)[1]+'_default'
        previous = next(r for r in nvos_parent['results'] if r['job'] == parent_name)
        pred = read(ROOT/'nvos'/name/'prediction.json')
        assert pred['configuration'] == nvos_config
        assert sha256(ROOT/'nvos'/name/'predictions.pt') == pred['predictions_sha256']
        nvos[name] = dict(actual_iou=row['foreground_iou'], response_iou=row['retained_response']['foreground_iou'],
                         response_ap=row['retained_response']['average_precision'],
                         parent_actual_iou=previous['foreground_iou'], parent_response_iou=previous['retained_response']['foreground_iou'],
                         parent_response_ap=previous['retained_response']['average_precision'])
        artifacts.append(ROOT/'nvos'/name/'prediction.json')
    for policy in ['mixed_unsigned', 'mixed_signed']:
        reasons = []
        for scene in ['figurines', 'ramen']:
            row = lerf[f'{scene}_{policy}']
            if row['actual_iou'] <= row['parent_actual_iou']:
                reasons.append(f'{scene} text actual subset did not improve')
        for scene in baseline:
            for mode, row in development[f'{scene}_{policy}'].items():
                if row['change_pp'] < 0:
                    reasons.append(f'{scene} {mode} actual subset regressed')
        for mode in ['positive', 'signed']:
            row = nvos[f'fern_{policy}_{mode}']
            for key in ['actual_iou', 'response_iou', 'response_ap']:
                if row[key] < row['parent_'+key]:
                    reasons.append(f'fern {mode} {key} regressed')
        gates[policy] = dict(passed=not reasons, failures=reasons)
    # Compare initializations and source diagnostics for original-code repeats.
    replay = {}
    initial = torch.load(baseline['fern']/'models/untrained/field.pt', map_location='cpu', weights_only=True)
    for name in ['fern_single_replay', 'fern_original_replay_1', 'fern_original_replay_2']:
        root = ROOT/name
        state = torch.load(root/'models/untrained/field.pt', map_location='cpu', weights_only=True)
        identical = all(torch.equal(state[k],v) for k,v in initial.items())
        assert identical
        replay[name] = dict(initialization_all_tensors_exact=identical,
                           development=read(root/'development_trained.json')['summary'])
        artifacts += [root/'models/training.json', root/'models/trained/manifest.json', root/'development_trained.json']
        del state
    del initial
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    git = lambda *args: subprocess.check_output(['git', '-C', '/root/RADIO-GS', *args], env=env).decode().strip()
    index = Path(git('rev-parse', '--git-path', 'index'))
    if not index.is_absolute(): index = Path('/root/RADIO-GS')/index
    original = dict(branch=git('branch', '--show-current'), head=git('rev-parse', 'HEAD'), index_sha256=sha256(index))
    original['index_matches_frozen'] = original['index_sha256'] == '917366f03337f78208f976ee26f4c58f198d2ce78ff64b29f9d7c94c06ad621c'
    code = list(Path('radio_gs/query_field').glob('*.py'))
    code += [Path(job['config']) for job in study['fits']]
    code += list(Path('configs/query_field').glob('support_training*.json'))
    code += list(Path('scripts/research').glob('*support_training*.py'))
    code += [Path('scripts/research/run_query_field_nvos.py'), Path('tests/test_query_field_support_training.py')]
    report = dict(schema='radio_gs.query_field_support_training_evidence.v1', parent_commit='b831d03',
        training=training, development=development, lerf=lerf, nvos=nvos, gates=gates,
        original_code_replays=replay, original_worktree=original,
        exact_episode_and_first_pixel_sequence_between_arms=True, unchanged_inference_spec_and_tensor_storage=True,
        all_public_predictions_before_new_target_scoring=True, source_denied_replays='4 real NVOS cases plus 4 four-input LERF models',
        concentration={scene:read(ROOT/f'{scene}_concentration.json')['summary'] for scene in baseline},
        promotion=False, independent_final_test=False,
        limitations=['2 of 4 LERF scenes and 1 of 8 NVOS cases; failed confirmation gate',
                     'training is not bitwise repeatable, unlike tested inference; small deltas not significance claims',
                     'support concentration is a post-study source-only diagnostic, not causal proof',
                     'no new ScanNet/UQIS or compactness evidence'],
        code=[receipt(p) for p in sorted(set(code))], artifacts=[receipt(p) for p in sorted(set(artifacts))])
    output = Path('docs/experiments/2026-10-07-support-training-evidence.json')
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(lerf=lerf, nvos=nvos, gates=gates, original=original), indent=2))


if __name__ == '__main__':
    main()
