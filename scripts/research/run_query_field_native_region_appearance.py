"""Matched pooling interventions; freeze all eight-arm predictions before scores."""
import argparse
import builtins
import copy
import gc
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.region_carrier import query_text_region_scope
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from radio_gs.querying.conditional_region_membership import FEATURE_NAMES
from scripts.research.compile_query_field_native_region_appearance import ROOT, PARENT, VARIANTS
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

CONTROLS = ['semantic_initial', 'frozen_visibility', 'seed_addition', 'conditional_addition']
ARMS = CONTROLS + [v + '_' + a for v in VARIANTS for a in ['addition', 'replace']]
PROTOCOL = Path('docs/experiments/2026-10-11-native-region-appearance-protocol.md')


def prepare():
    cfg = copy.deepcopy(read(PARENT / 'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='matched native region pooling; adaptive development',
               member_readout='frozen native seed / scope policy; region appearance only intervention')
    records = []
    for j in cfg['jobs']:
        for variant in VARIANTS:
            folder = ROOT / variant / j['scene']
            assert read(folder / 'compilation.json')['complete']
            records += [receipt(folder / 'field/manifest.json'), receipt(folder / 'compilation.json')]
        j['field'] = str(ROOT / 'native_448' / j['scene'] / 'field')
        records += [receipt(j['queries']), receipt(PARENT / 'predictions' / j['name'] / 'predictions.pt'),
                    receipt(PARENT / 'predictions' / j['name'] / 'prediction.json')]
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__),
        Path('scripts/research/compile_query_field_native_region_appearance.py'),
        Path('radio_gs/models/native_region_appearance.py'), Path('radio_gs/querying/conditional_region_membership.py')]
    write(ROOT / 'prediction_config.json', cfg)
    write(ROOT / 'design.json', dict(configuration=receipt(ROOT / 'prediction_config.json'), inputs=records,
        protocol=receipt(PROTOCOL), code=[receipt(p) for p in code],
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        primary='native_448_addition', main_comparator='conditional_addition', stronger_comparator='seed_addition',
        earlier_development_scores_seen=True, default_promoted=False, training=False))


def design():
    d = read(ROOT / 'design.json')
    for r in d['inputs'] + d['code'] + [d['configuration'], d['protocol']]:
        check(r)
    return d


def denied(*args, **kwargs):
    raise AssertionError('unexpected runtime file open')


@torch.no_grad()
def predict(index):
    d = design(); cfg = read(d['configuration']['path']); j = cfg['jobs'][index]
    out = ROOT / 'predictions' / j['name']; out.mkdir(parents=True, exist_ok=False)
    previous = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
    old_details = read(PARENT / 'predictions' / j['name'] / 'prediction.json')['diagnostics']
    assert previous['texts'] == [q['text'] for q in read(j['queries'])['queries']]
    members = {a: previous['membership_by_control'][a] for a in CONTROLS}; diagnostics = {}
    for variant in VARIANTS:
        field, _ = load_field(ROOT / variant / j['scene'] / 'field')
        for key, value in previous['geometry'].items():
            assert torch.equal(value, getattr(field, key))
        decoder = field.region_carrier.decoder(field.xyz)
        exclusion = JointExclusionDecoder(field); native_reader = NativeTextDevice(field, 'cuda')
        kwargs = dict(region_decoder=decoder, exclusion_decoder=exclusion, native_reader=native_reader)
        query_text_region_scope(field, previous['native_query_vectors'][0], previous['generic_negative_vectors'], **kwargs)
        values_by_arm = {a: [] for a in ['addition', 'replace']}; details = []; pairs = []
        for qi, text in enumerate(previous['texts']):
            start = time.monotonic()
            with patch.object(builtins, 'open', denied):
                values, detail = query_text_region_scope(field, previous['native_query_vectors'][qi], previous['generic_negative_vectors'], **kwargs)
                # Diagnostic only: a same-instance quality estimate and the
                # current text identity are different conditional functions.
                _, native_probability = native_reader(previous['native_query_vectors'][qi], previous['generic_negative_vectors'])
                detail['admitted_native_probabilities'] = native_probability[detail.get('admitted', [])].tolist()
            for a in CONTROLS[:3]:
                assert torch.equal(values[a] > .5, previous['selected_by_control'][a][:, qi]), (variant, text, a)
                assert float((values[a] - members[a][:, qi]).abs().max()) <= 2e-6
            assert detail.get('seed') == old_details[qi].get('seed')
            for a in values_by_arm:
                values_by_arm[a].append(values['conditional_' + a])
            if 'seed' in detail:
                ids, features, _ = decoder.features.pairs(detail['seed']); pairs.append(features)
            details.append(dict(text=text, seconds=time.monotonic() - start, **detail))
            print('predicted', j['scene'], variant, qi, text, 'applied', detail['applied'], flush=True)
        for a, values in values_by_arm.items():
            members[variant + '_' + a] = torch.stack(values, 1)
        features = np.concatenate(pairs)
        diagnostics[variant] = dict(queries=details, feature_pairs=len(features), feature_medians=dict(zip(FEATURE_NAMES, np.median(features, axis=0).tolist())),
            file_opens_denied_queries=len(details), os_sandbox=False, warmup='first actual query before Python open guard')
        del field, decoder, exclusion, native_reader, kwargs, values, pairs, features
        gc.collect(); torch.cuda.empty_cache()
    sets = {a: v > .5 for a, v in members.items()}
    previous.update(scores={a: previous['scores']['semantic_initial'] for a in ARMS}, membership_by_control=members,
                    selected_by_control=sets, selected=sets['native_448_addition'])
    torch.save(previous, out / 'predictions.pt')
    write(out / 'prediction.json', dict(configuration=cfg, job=j, diagnostics=diagnostics, queries=len(previous['texts']),
        predictions_sha256=receipt(out / 'predictions.pt')['sha256'], source_access=False, target_labels_opened=False,
        evaluation_scene_opened_by_predictor=False, first_three_controls_recomputed_exact=True, old_conditional_reused=True))


def run(gpus):
    d = design(); cfg = read(d['configuration']['path'])
    def pred(i, gpu):
        execute(['-m', 'scripts.research.run_query_field_native_region_appearance', 'predict', '--index', str(i)], ROOT / 'logs' / f'predict_{i}.log', gpu)
    lanes(list(range(4)), [2], pred)
    write(ROOT / 'predictions_complete.json', dict(records=[receipt(ROOT / 'predictions' / j['name'] / 'prediction.json') for j in cfg['jobs']], queries=68, controls=len(ARMS)))
    def score(j, gpu):
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT / 'prediction_config.json'), '--job', j['name'], '--output', str(ROOT / 'predictions')], ROOT / 'logs' / ('score_' + j['scene'] + '.log'), gpu)
    lanes(cfg['jobs'], gpus, score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        previous = read(PARENT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for key in ['iou', 'ap', 'localization', 'observations']:
                assert m[key] == previous['semantic_initial'][key]
            if a in CONTROLS:
                for key in ['miou', 'count', 'missing']:
                    assert m['selected'][key] == previous[a]['selected'][key]
            per[a][j['scene']] = m['selected']['miou']
    report = dict(complete=True, macro={a: sum(v.values()) / 4 for a, v in per.items()}, per_scene=per,
        predictions_before_scoring=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'summary.json', report); print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'run']); p.add_argument('--index', type=int); p.add_argument('--gpus', default='4')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'predict': predict(a.index)
    else: run(list(map(int, a.gpus.split(','))))
