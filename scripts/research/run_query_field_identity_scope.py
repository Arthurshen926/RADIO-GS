"""Two-factor native identity / seed preservation study on fixed native448 fields."""
import argparse
import builtins
import copy
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from radio_gs.query_field.region_carrier import query_text_region_scope
from radio_gs.query_field.identity_region_scope import IdentityRegionScopeDecoder, query_text_identity_scope
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261011/query_identity_scope_v2')
PARENT = Path('output/query_field_20261011/native_region_appearance_v1')
ARMS = ['frozen_visibility', 'seed_addition', 'unqualified_addition', 'unqualified_seeded', 'native_addition', 'native_seeded']
PROTOCOL = Path('docs/experiments/2026-10-11-query-identity-scope-protocol.md')


def prepare():
    # Parent prediction barrier, not parent scores, is the dependency.
    barrier = read(PARENT / 'predictions_complete.json')
    for r in barrier['records']: check(r)
    cfg = copy.deepcopy(read(PARENT / 'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='identity / seed preservation factorial; adaptive development',
               member_readout='fixed native448 regional functions; native identity prerequisite and seed floor')
    records = [receipt(PARENT / 'predictions_complete.json')]
    for j in cfg['jobs']:
        for path in [Path(j['field']) / 'manifest.json', Path(j['queries']), PARENT / 'predictions' / j['name'] / 'predictions.pt']:
            records.append(receipt(path))
    write(ROOT / 'prediction_config.json', cfg)
    write(ROOT / 'design.json', dict(configuration=receipt(ROOT / 'prediction_config.json'), inputs=records,
        protocol=receipt(PROTOCOL), code=[receipt(p) for p in sorted(Path('radio_gs/query_field').glob('*.py'))] +
            [receipt(__file__), receipt('radio_gs/querying/conditional_region_membership.py')],
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        primary='native_seeded', comparator='seed_addition', factors=['native_identity', 'seed_preservation'],
        representation_selected_before_parent_scores=True, default_promoted=False, training=False))


def design():
    d = read(ROOT / 'design.json')
    for r in d['inputs'] + d['code'] + [d['configuration'], d['protocol']]: check(r)
    return d


@torch.no_grad()
def predict(index):
    d = design(); cfg = read(d['configuration']['path']); j = cfg['jobs'][index]
    out = ROOT / 'predictions' / j['name']; out.mkdir(parents=True, exist_ok=False)
    previous = torch.load(PARENT / 'predictions' / j['name'] / 'predictions.pt', weights_only=True)
    field, _ = load_field(j['field']); decoder = IdentityRegionScopeDecoder(field.region_carrier, field.xyz)
    exclusion = JointExclusionDecoder(field); native_reader = NativeTextDevice(field, 'cuda')
    kwargs = dict(region_decoder=decoder, exclusion_decoder=exclusion, native_reader=native_reader)
    old = previous['membership_by_control']
    members = dict(frozen_visibility=old['frozen_visibility'], seed_addition=old['seed_addition'],
        unqualified_addition=old['native_448_addition'], unqualified_seeded=torch.maximum(old['seed_addition'], old['native_448_addition']))
    query_text_region_scope(field, previous['native_query_vectors'][0], previous['generic_negative_vectors'], **kwargs)
    extension = []; seeded = []; details = []; max_public_score_error = 0.
    def denied(*args, **kwargs): raise AssertionError('runtime file open')
    for qi, text in enumerate(previous['texts']):
        with patch.object(builtins, 'open', denied):
            values, detail = query_text_region_scope(field, previous['native_query_vectors'][qi], previous['generic_negative_vectors'], **kwargs)
            public = query_text_identity_scope(field, previous['native_query_vectors'][qi], previous['generic_negative_vectors'], **kwargs)
        for a in ['semantic_initial', 'frozen_visibility', 'seed_addition']:
            assert torch.equal(values[a] > .5, previous['selected_by_control'][a][:, qi]), (text, a)
            assert float((values[a] - old[a][:, qi]).abs().max()) <= 2e-6
        expected = torch.maximum(values['seed_addition'], values['conditional_addition'])
        # Repeated CUDA semantic reductions need not be bitwise equal. Use the
        # existing 2e-6 continuous-score contract while keeping exact set parity.
        assert torch.equal(public.selected, expected > .5)
        public_score_error = float((public.membership_score - expected).abs().max())
        assert public_score_error <= 2e-6
        max_public_score_error = max(max_public_score_error, public_score_error)
        extension.append(values['conditional_addition']); seeded.append(public.membership_score)
        details.append(dict(text=text, **detail)); print('predicted', j['scene'], qi, text, detail, flush=True)
    members.update(native_addition=torch.stack(extension, 1), native_seeded=torch.stack(seeded, 1))
    sets = {a: v > .5 for a, v in members.items()}
    previous.update(scores={a: previous['scores']['semantic_initial'] for a in ARMS}, membership_by_control=members,
                    selected_by_control=sets, selected=sets['native_seeded'])
    torch.save(previous, out / 'predictions.pt')
    write(out / 'prediction.json', dict(configuration=cfg, job=j, diagnostics=details, queries=len(details),
        predictions_sha256=receipt(out / 'predictions.pt')['sha256'], source_access=False, target_labels_opened=False,
        evaluation_scene_opened_by_predictor=False, controls_exact=True, public_entry_sets_exact=True,
        public_score_max_error=max_public_score_error, public_score_tolerance=2e-6,
        file_opens_denied_queries=len(details), os_sandbox=False, warmup='first actual query before Python open guard'))


def run(gpus):
    d = design(); cfg = read(d['configuration']['path'])
    def pred(i, gpu):
        execute(['-m', 'scripts.research.run_query_field_identity_scope', 'predict', '--index', str(i)], ROOT / 'logs' / f'predict_{i}.log', gpu)
    lanes(list(range(4)), [2], pred)
    write(ROOT / 'predictions_complete.json', dict(records=[receipt(ROOT / 'predictions' / j['name'] / 'prediction.json') for j in cfg['jobs']], queries=68, controls=len(ARMS)))
    def score(j, gpu):
        execute(['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(ROOT / 'prediction_config.json'), '--job', j['name'], '--output', str(ROOT / 'predictions')], ROOT / 'logs' / ('score_' + j['scene'] + '.log'), gpu)
    lanes(cfg['jobs'], gpus, score)
    per = {a: {} for a in ARMS}
    for j in cfg['jobs']:
        metrics = read(ROOT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        old = read(PARENT / 'predictions' / j['name'] / 'evaluation/evaluation.json')['metrics']
        for a, m in metrics.items():
            for key in ['iou', 'ap', 'localization', 'observations']: assert m[key] == old['semantic_initial'][key]
            if a in ARMS[:3]:
                reference = a if a != 'unqualified_addition' else 'native_448_addition'
                for key in ['miou', 'count', 'missing']: assert m['selected'][key] == old[reference]['selected'][key]
            per[a][j['scene']] = m['selected']['miou']
    result = dict(complete=True, macro={a: sum(v.values()) / 4 for a, v in per.items()}, per_scene=per,
                  predictions_before_scoring=True, default_promoted=False, whole_five_task=False)
    write(ROOT / 'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'run']); p.add_argument('--index', type=int); p.add_argument('--gpus', default='2')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.stage == 'prepare': prepare()
    elif a.stage == 'predict': predict(a.index)
    else: run(list(map(int, a.gpus.split(','))))
