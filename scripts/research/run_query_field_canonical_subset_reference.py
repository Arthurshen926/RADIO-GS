"""Canonical v10 source-assisted reference on the current field's fixed inputs.

Deliberately calls the retained production stages, not a rewritten optimizer.
This research runner must never be confused with a source-free field decoder.
"""
import argparse
import copy
from pathlib import Path
import subprocess

import torch
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261011/canonical_subset_reference_v1')
PARENT = Path('output/query_field_20261011/binary_region_scope_v2')
PROTOCOL = Path('docs/experiments/2026-10-11-canonical-subset-reference-protocol.md')
CONTROLS = ['semantic_initial', 'frozen_visibility', 'seed_addition']
ARMS = CONTROLS + ['raw_fusion', 'raw_joint', 'admitted_fusion', 'admitted_joint']
STAGE_CODE = ['radio_gs/pipeline/subset.py', 'scripts/research/filter_source_identity.py',
              'scripts/research/condition_source_identity.py', 'radio_gs/querying/source_identity.py',
              'radio_gs/querying/typed_evidence.py', 'radio_gs/querying/finite_evidence.py',
              'radio_gs/rendering/contribution_compositor.py', 'radio_gs/geometry/frozen_gaussian.py',
              'radio_gs/field/moge_geometry_prior.py']


def validate_cohort(data, queries, construction, manifest):
    texts = [x['text'] for x in queries['queries']]
    if data['texts'] != texts or len(set(texts)) != len(texts):
        raise ValueError('query order or uniqueness changed')
    frames = [r['frame'] for r in construction['source_records']]
    if len(frames) != 18 or len(set(frames)) != len(frames):
        raise ValueError('expected eighteen distinct construction views')
    train = set(manifest['splits']['train_frames']); dev = set(manifest['splits']['development_frames'])
    if not set(frames) <= train or set(frames) & dev:
        raise ValueError('construction view not exclusively training')
    initial = data['membership_by_control']['semantic_initial']
    if initial.shape != (len(data['observed']), len(texts)) or not torch.isfinite(initial).all():
        raise ValueError('invalid current prior')
    if bool(((initial < 0) | (initial > 1)).any()) or bool(initial[~data['observed']].any()):
        raise ValueError('prior outside probability or member domain')
    return frames


def initializer_from_state(prior, state, history):
    """No-positive queries retain the native prior exactly, not an empty set."""
    result = prior.clone()
    if len(history) != prior.shape[1]:
        raise ValueError('incomplete canonical query history')
    expected = {qi for qi, row in enumerate(history) if row['views'] > 0}
    if set(state) != expected:
        raise ValueError('canonical initializer/history mismatch')
    for qi, row in state.items():
        value = row['start_membership']
        if value.shape != prior[:, qi].shape or not torch.isfinite(value).all():
            raise ValueError('invalid sealed source initializer')
        result[:, qi] = value
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT / 'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, runtime_source_access=True,
               status='canonical source-assisted diagnostic; never deployment',
               member_readout='retained v10 subset stage, current native prior and eighteen construction views')
    records = []
    for j in cfg['jobs']:
        out = ROOT / 'inputs' / j['scene']; out.mkdir(parents=True)
        path = PARENT / 'predictions' / j['name'] / 'predictions.pt'
        old_record = read(path.with_name('prediction.json'))
        check(dict(path=str(path), sha256=old_record['predictions_sha256']))
        data = torch.load(path, weights_only=True)
        construction = read(j['construction']); manifest = read(Path(j['observations']) / 'observations.json')
        frames = validate_cohort(data, read(j['queries']), construction, manifest)
        # This scene file contains source camera/geometry authority, not labels.
        source = read(j['evaluation_scene']); binding = source['geometry']
        check(binding)
        geometry = torch.load(binding['path'], weights_only=False)['state_dict']
        for key, value in data['geometry'].items():
            assert torch.equal(value, geometry[key]), ('changed geometry', j['scene'], key)
        scene = {k: source[k] for k in ['scene', 'camera_report', 'camera_report_sha256', 'native_region_directories']}
        scene['source_frames'] = frames
        write(out / 'scene.json', scene)
        assert source['source_fit_hw'] == [192, 256]
        torch.save(dict(texts=data['texts'], encodings=data['native_query_vectors'],
            semantic_probability=data['membership_by_control']['semantic_initial'], observed=data['observed'],
            geometry=binding), out / 'response.pt')
        inputs = dict(scene=str(out / 'scene.json'), response=str(out / 'response.pt'),
                      detections=j['teacher_detections'], base_raster=[192, 256], tracks=[None] * len(data['texts']))
        write(out / 'raw.json', inputs)
        paths = [path, path.with_name('prediction.json'), Path(j['queries']), Path(j['construction']),
                 Path(j['field']) / 'manifest.json', Path(j['observations']) / 'observations.json',
                 Path(j['evaluation_scene']), Path(binding['path']), Path(scene['camera_report']),
                 out / 'scene.json', out / 'response.pt', out / 'raw.json']
        proposal_counts = []
        for frame in frames:
            raw_path = Path(j['observations']) / f'view_{frame:05d}.pt'
            expected = next(r for r in manifest['frames'] if r['frame'] == frame)['compiled']
            assert receipt(raw_path)['sha256'] == expected['sha256']
            paths.extend([raw_path, Path(j['teacher_detections']) / f'identity_{frame}.pt'])
            count = 0
            for directory in scene['native_region_directories']:
                p = Path(directory) / f'frame_{frame:05d}.pt'; paths.append(p)
                native = torch.load(p, weights_only=False)
                count += len(native['descriptors'])
                if len(native['descriptors']): paths.append(Path(native['sam_path']))
            proposal_counts.append(dict(frame=frame, original_native_candidates=count))
        write(out / 'binding.json', dict(geometry_exact=True, source_frames=frames,
            source_raster=[384, 512], sampled_pixels_per_view=192*256,
            source_world_units_per_metre=manifest['world_units_per_metre'],
            member_domain=int(data['observed'].sum()), queries=len(data['texts']),
            original_native_candidates=proposal_counts, target_labels_opened=False))
        paths.append(out / 'binding.json')
        records += [dict(scene=j['scene'], **receipt(p)) for p in dict.fromkeys(paths)]
        print('sealed', j['scene'], len(frames), sum(x['original_native_candidates'] for x in proposal_counts), flush=True)
    write(ROOT / 'prediction_config.json', cfg)
    write(ROOT / 'design.json', dict(configuration=receipt(ROOT / 'prediction_config.json'),
        protocol=receipt(PROTOCOL), inputs=records,
        code=[receipt(p) for p in [__file__, *STAGE_CODE, 'radio_gs/query_field/public_lerf.py']],
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        primary='admitted_joint', comparator='seed_addition', steps=240,
        source_assisted=True, transient_query_fitting=True, persistent_field_updated=False,
        default_promoted=False, independent_final_test=False, whole_five_task=False))


def design(scene=None, inputs=True):
    d = read(ROOT / 'design.json')
    rows = d['code'] + [d['configuration'], d['protocol']]
    if inputs: rows += [r for r in d['inputs'] if scene is None or r['scene'] == scene]
    for r in rows: check(r)
    return d


def predict(index):
    d = design(inputs=False); cfg = read(d['configuration']['path']); j = cfg['jobs'][index]
    design(scene=j['scene']); inp = ROOT / 'inputs' / j['scene']; out = ROOT / 'predictions' / j['name']
    out.mkdir(parents=True, exist_ok=False)
    def stage(module, args, name):
        execute(['-m', module, *map(str, args)], out / (name + '.log'), None)
    stage('scripts.research.filter_source_identity', ['--inputs', inp / 'raw.json', '--output', out / 'identity_record',
          '--heldout-stride', '0', '--record-only'], 'record_identity')
    stage('scripts.research.condition_source_identity', ['--inputs', inp / 'raw.json', '--identity-record',
          out / 'identity_record/identity.json', '--output', out / 'identity_evidence'], 'admit_identity')
    old = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
    members = {a: old['membership_by_control'][a] for a in CONTROLS}; details = {}
    for arm, input_path in [('raw', inp / 'raw.json'), ('admitted', out / 'identity_evidence/inputs.json')]:
        state_path = out / (arm + '_optimization_state.pt')
        stage('radio_gs.pipeline.subset', ['--inputs', input_path, '--key', j['scene'], '--output-base', out / arm,
              '--mode', 'admission', '--scope-initialization', 'source', '--pixel-loss', 'balanced',
              '--support-prior', 'footprint', '--raster-scale', '2', '--steps', '240',
              '--optimization-state-out', state_path], arm + '_fit')
        fit = read(out / arm / j['scene'] / 'fit.json')
        result = torch.load(out / arm / j['scene'] / 'typed_readout.pt', weights_only=True)
        state = torch.load(state_path, weights_only=True)
        assert result['texts'] == old['texts']
        assert torch.equal(result['selection_observed'], old['observed'])
        assert torch.equal(result['scores'], members['semantic_initial'])
        assert fit['source_frames'] == read(inp / 'binding.json')['source_frames']
        assert fit['steps'] == 240 and fit['source_raster'] == [384, 512]
        members[arm + '_fusion'] = initializer_from_state(members['semantic_initial'], state, fit['history'])
        members[arm + '_joint'] = result['selection_scores']
        assert not bool((members[arm + '_joint'][~old['observed']] > .5).any())
        details[arm] = dict(fit=receipt(out / arm / j['scene'] / 'fit.json'), state=receipt(state_path),
                            history=fit['history'], active_queries=len(state))
        print('complete', j['scene'], arm, len(state), flush=True)
    sets = {a: v > .5 for a, v in members.items()}
    for a in CONTROLS: assert torch.equal(sets[a], old['selected_by_control'][a])
    old.update(scores={a: old['scores']['semantic_initial'] for a in ARMS}, membership_by_control=members,
               selected_by_control=sets, selected=sets['admitted_joint'])
    torch.save(old, out / 'predictions.pt')
    write(out / 'prediction.json', dict(configuration=cfg, job=j, queries=len(old['texts']), diagnostics=details,
        predictions_sha256=receipt(out / 'predictions.pt')['sha256'], source_access=True,
        transient_query_fitting=True, canonical_subset_used=True, geometry_exact=True, controls_exact=True,
        target_labels_opened=False, persistent_field_updated=False, default_promoted=False))


def run(gpus):
    d = design(inputs=False); cfg = read(d['configuration']['path'])
    def pred(i, gpu):
        execute(['-m', 'scripts.research.run_query_field_canonical_subset_reference', 'predict', '--index', str(i)],
                ROOT / 'logs' / f'predict_{i}.log', gpu)
    lanes(list(range(4)), gpus, pred)
    records = []
    for j in cfg['jobs']:
        path = ROOT / 'predictions' / j['name'] / 'prediction.json'; r = read(path)
        assert r['queries'] == len(read(j['queries'])['queries']) and not r['target_labels_opened']
        check(dict(path=str(path.with_name('predictions.pt')), sha256=r['predictions_sha256']))
        records.append(receipt(path))
    write(ROOT / 'predictions_complete.json', dict(records=records, queries=68, controls=len(ARMS)))
    def score(j, gpu):
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT / 'prediction_config.json'),
                 '--job', j['name'], '--output', str(ROOT / 'predictions')], ROOT / 'logs' / ('score_' + j['scene'] + '.log'), gpu)
    lanes(cfg['jobs'], gpus, score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        old = read(PARENT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for key in ['iou', 'ap', 'localization', 'observations']: assert m[key] == old['semantic_initial'][key]
            if a in CONTROLS:
                for key in ['miou', 'count', 'missing']: assert m['selected'][key] == old[a]['selected'][key]
            per[a][j['scene']] = m['selected']['miou']
    write(ROOT / 'scoring_complete.json', dict(records=[receipt(ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json') for j in cfg['jobs']]))
    report = dict(complete=True, macro={a: sum(v.values()) / 4 for a, v in per.items()}, per_scene=per,
                  source_assisted=True, canonical_subset_used=True, predictions_before_scoring=True,
                  persistent_field_updated=False, default_promoted=False, whole_five_task=False)
    write(ROOT / 'summary.json', report); print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'run']); p.add_argument('--index', type=int); p.add_argument('--gpus', default='2,5')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'predict': predict(a.index)
    else: run(list(map(int, a.gpus.split(','))))
