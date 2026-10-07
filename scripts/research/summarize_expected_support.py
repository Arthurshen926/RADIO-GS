"""Seal fixed aggregation controls and all input/readout gates without selection."""
import json
import os
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/expected_support_v1')
BASE = ROOT.parent
read = lambda p: json.loads(Path(p).read_text())


def receipt(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def spec(value):
    return dict(local_metric='none', support_aggregation='normalized_prototype', **{
        k:v for k,v in value.items() if k not in {'local_metric', 'support_aggregation'}}) | {
            'local_metric':value.get('local_metric', 'none'),
            'support_aggregation':value.get('support_aggregation', 'normalized_prototype')}


def main():
    torch.set_num_threads(4)
    study = read('configs/query_field/expected_support_study.json')
    for row in read(ROOT/'launch.json')['code_config_protocol']:
        assert sha256(row['path']) == row['sha256'], row['path']
    artifacts = [ROOT/name for name in ['launch.json', 'tests.log', 'cold_replay.json',
        'nvos/evaluation.json', 'subset_errors.json']]
    artifacts += [ROOT/f'{scene}_prescore_barrier.json' for scene in ['figurines','ramen','fern']]
    artifacts += [ROOT/f'{scene}_jensen.json' for scene in ['figurines','ramen','fern']]
    artifacts += [ROOT/'cross_gpu_replay_difference.json', ROOT/'same_gpu_probe.json', ROOT/'cold.log']
    cold = read(ROOT/'cold_replay.json')
    assert len(cold['comparisons']) == 6 and cold['original_open_rejected']
    assert all(all(row['exact'].values()) for row in cold['comparisons'])
    training, development, cost, traces, frozen = {}, {}, {}, {}, {}
    for model in study['models']:
        name, arm = model['name'], model['arm']
        directory = Path(model['output'])
        field = Path(model['field'])
        manifest = read(field/'manifest.json')
        parent_path = Path(model['parent_field'])
        parent = read(parent_path/'manifest.json')
        assert sha256(field/'field.pt') == manifest['files']['field.pt']['sha256']
        expected = spec(parent['spec']); expected['support_aggregation'] = model['aggregation']
        assert spec(manifest['spec']) == expected
        assert manifest['tensor_bytes'] == parent['tensor_bytes']
        assert manifest['provenance']['observations_sha256'] == parent['provenance']['observations_sha256']
        current = torch.load(field/'field.pt', map_location='cpu', weights_only=True)
        previous = torch.load(parent_path/'field.pt', map_location='cpu', weights_only=True)
        protected = [k for k in previous if k not in ['group','scale_cutoffs','log_temperature','affinity_bias']]
        assert set(current) == set(previous)
        assert all(torch.equal(current[k], previous[k]) for k in protected)
        if arm == 'frozen_expected':
            frozen[name] = dict(all_parent_tensors_exact=all(torch.equal(current[k],previous[k]) for k in previous))
            assert frozen[name]['all_parent_tensors_exact']
        del current, previous
        current = read(directory/'development_trained.json')['summary']
        previous = read(parent_path.parent.parent/'development_trained.json')['summary']
        development[name] = {k:dict(cases=v['cases'], failed=v['failed'], actual_iou=v['mean_actual_subset_iou'],
            parent_actual_iou=previous[k]['mean_actual_subset_iou'],
            change_pp=100*(v['mean_actual_subset_iou']-previous[k]['mean_actual_subset_iou']))
            for k,v in current.items() if k.endswith('/default')}
        if arm == 'frozen_expected':
            for mode in ['region_image/default','point_3d/default']:
                assert development[name][mode]['change_pp'] == 0
        else:
            fitted = read(directory/'models/training.json')
            assert fitted['geometry_unchanged'] and fitted['native_coordinates_unchanged']
            assert fitted['rendered_losses'] == 4000 and fitted['paired_anchor_episodes'] == 2000
            assert sum(fitted['support_counts'].values()) == 2000
            traces[name] = fitted['support_trace']
            parent_fit = read(parent_path.parent/'training.json')
            assert [(r['frame'],r['region']) for r in fitted['history']] == [(r['frame'],r['region']) for r in parent_fit['history']]
            training[name] = {k:fitted[k] for k in ['support_counts','member_forward_evaluations','rendered_losses',
                'paired_anchor_episodes','training_episodes','seconds','geometry_unchanged','native_coordinates_unchanged']}
            artifacts.append(directory/'models/training.json')
        smoke = read(directory/'runtime_smoke.json')
        assert all(smoke['exact_replay'].values())
        cost[name] = dict(tensor_bytes=manifest['tensor_bytes'], deployment_bytes=manifest['files']['field.pt']['bytes'])
        artifacts += [field/'manifest.json', directory/'development_trained.json', directory/'runtime_smoke.json']
    for scene in ['figurines','ramen','fern']:
        assert traces[f'{scene}_paired_legacy'] == traces[f'{scene}_paired_expected']
        initial = []
        for arm in ['paired_legacy','paired_expected']:
            model = next(m for m in study['models'] if m['name'] == f'{scene}_{arm}')
            initial.append(torch.load(Path(model['output'])/'models/untrained/field.pt', map_location='cpu', weights_only=True))
        assert all(torch.equal(initial[0][k], initial[1][k]) for k in initial[0])
        parent_initial = torch.load(Path(model['parent_field']).parent/'untrained/field.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(initial[0][k], parent_initial[k]) for k in initial[0])
        del parent_initial
        del initial
    lerf = {}
    for job in study['jobs']:
        directory = ROOT/'lerf'/job['name']
        record = read(directory/'prediction.json')
        assert record['configuration'] == study and record['native_response_bit_exact']
        assert sha256(directory/'predictions.pt') == record['predictions_sha256']
        value = read(directory/'evaluation/evaluation.json')['metrics']
        parent = read(Path(job['reference_prediction'])/'evaluation/evaluation.json')['metrics']
        lerf[job['name']] = dict(actual_iou=value['shared_membership']['selected']['miou'],
            parent_actual_iou=parent['shared_membership']['selected']['miou'],
            member_semantic_iou=value['shared_membership']['iou'],
            native_semantic_iou=value['native_generic_relevance']['iou'],
            cases=value['shared_membership']['selected']['count'], missing=value['shared_membership']['selected']['missing'])
        if job['arm'] == 'frozen_expected':
            a = torch.load(directory/'predictions.pt', map_location='cpu', weights_only=True)
            b = torch.load(Path(job['reference_prediction'])/'predictions.pt', map_location='cpu', weights_only=True)
            exact = torch.equal(a['scores']['shared_membership'],b['scores']['shared_membership']) and torch.equal(a['selected'],b['selected'])
            assert exact
            frozen[job['name']]['public_text_membership_and_selection_exact'] = exact
            del a,b
        artifacts += [directory/'prediction.json', directory/'evaluation/evaluation.json']
    nvos, runtime = {}, {}
    configuration = read('configs/query_field/expected_support_nvos.json')
    parent = read(BASE/'nvos_fern_v1/public/evaluation.json')['results']
    for row in read(ROOT/'nvos/evaluation.json')['results']:
        name = row['job']
        previous = next(r for r in parent if r['job'] == 'trained_'+name.rsplit('_',1)[1]+'_default')
        record = read(ROOT/'nvos'/name/'prediction.json')
        assert record['configuration'] == configuration
        assert sha256(ROOT/'nvos'/name/'predictions.pt') == record['predictions_sha256']
        nvos[name] = dict(actual_iou=row['foreground_iou'], response_iou=row['retained_response']['foreground_iou'],
            response_ap=row['retained_response']['average_precision'], parent_actual_iou=previous['foreground_iou'],
            parent_response_iou=previous['retained_response']['foreground_iou'],
            parent_response_ap=previous['retained_response']['average_precision'])
        runtime[name] = {k:record[k] for k in ['timing_seconds','peak_cuda_allocated_bytes','registration','prompt_agreement']}
        artifacts.append(ROOT/'nvos'/name/'prediction.json')
    gates = {}
    for arm in ['paired_legacy','paired_expected']:
        reasons = []
        for scene in ['figurines','ramen']:
            v = lerf[f'{scene}_{arm}']
            if v['actual_iou'] <= v['parent_actual_iou']:
                reasons.append(f'{scene} text actual subset did not improve')
        for scene in ['figurines','ramen','fern']:
            for mode, v in development[f'{scene}_{arm}'].items():
                if v['change_pp'] < 0:
                    reasons.append(f'{scene} {mode} regressed')
        for mode in ['positive','signed']:
            v = nvos[f'fern_{arm}_{mode}']
            for key in ['actual_iou','response_iou','response_ap']:
                if v[key] < v['parent_'+key]:
                    reasons.append(f'fern {mode} {key} regressed')
        gates[arm] = dict(passed=not reasons, failures=reasons)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    git = lambda *args:subprocess.check_output(['git','-C','/root/RADIO-GS',*args],env=env).decode().strip()
    index = Path(git('rev-parse','--git-path','index'))
    if not index.is_absolute(): index = Path('/root/RADIO-GS')/index
    original = dict(branch=git('branch','--show-current'), head=git('rev-parse','HEAD'), index_sha256=sha256(index))
    original['index_matches_frozen'] = original['index_sha256'] == '917366f03337f78208f976ee26f4c58f198d2ce78ff64b29f9d7c94c06ad621c'
    code = [Path(r['path']) for r in read(ROOT/'launch.json')['code_config_protocol']]
    code += [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__)]
    code.append(Path('radio_gs/query_field/support_jensen_audit.py'))
    result = dict(schema='radio_gs.expected_support_evidence.v1', parent_commit='b3ea675',
        training=training, development=development, lerf=lerf, nvos=nvos, runtime=runtime,
        frozen_invariants=frozen, gates=gates, cost=cost,
        source_jensen_audit={scene:read(ROOT/f'{scene}_jensen.json')['summary'] for scene in ['figurines','ramen','fern']},
        paired_initializations_and_episode_support_traces_exact=True,
        preserved_native_geometry_and_no_tensor_growth=True, all_predictions_before_new_target_scoring=True,
        source_denied_replays='6 NVOS prompts on the original GPU and 9 four-input models', original_worktree=original,
        cross_gpu_renderer_difference=read(ROOT/'cross_gpu_replay_difference.json'),
        promotion=False, independent_final_test=False,
        limitations=['expected logit is not expected or calibrated probability',
                     'one cross-GPU alpha replay differs by <=5.9605e-8 at 22 pixels; members, selections and >0.5 masks exact',
                     'paired training doubles rendered losses; independent negative rays cost extra compute',
                     'training non-bitwise repeatability established previously; no significance claim from one fit',
                     '2 LERF scenes and 1/8 NVOS cases; no new ScanNet/UQIS or compression result'],
        code=[receipt(p) for p in sorted(set(code))], artifacts=[receipt(p) for p in sorted(set(artifacts))])
    output = Path('docs/experiments/2026-10-07-expected-support-evidence.json')
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(lerf=lerf,nvos=nvos,gates=gates),indent=2))


if __name__ == '__main__':
    main()
