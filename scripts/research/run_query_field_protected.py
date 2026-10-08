"""Three full-cohort interventions: rollback, protected fit, functional natives."""
import argparse
import copy
import gc
from pathlib import Path

import torch

from radio_gs.query_field.query_program import PROGRAM
from radio_gs.query_field.storage import load_field, save_field, sha256
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

PREVIOUS = common.ROOT
ROOT = Path('output/query_field_20261008/protected_relations_v1')
ARMS = ('decoder_rollback', 'protected_training', 'functional_native')
PROTOCOL = Path('docs/experiments/2026-10-08-protected-native-protocol.md')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    previous = read(PREVIOUS / 'design.json')
    fields = copy.deepcopy(previous['fields'])
    for job in fields:
        directory = PREVIOUS / 'fits/cross_view_anchor' / job['key']
        job['cross_view_field'] = receipt(directory / 'field/field.pt')
        job['cross_view_training'] = receipt(directory / 'training.json')
    template = read(PREVIOUS / 'cross_view_anchor/cohort.json')
    lookup = {j['parent']: j['key'] for j in fields}
    arms = {}
    for arm in ARMS:
        root = ROOT / arm
        cohort = copy.deepcopy(template); cohort['root'] = str(root)
        for family in ['lerf', 'nvos', 'scannet', 'uqis']:
            for job in cohort[family]:
                job['field'] = str(ROOT / 'fits' / arm / lookup[job['parent_field']] / 'field')
        lerf = read(PREVIOUS / 'cross_view_anchor/configs/lerf.json')
        for job in cohort['lerf']:
            next(j for j in lerf['jobs'] if j['name'] == job['scene'])['field'] = job['field']
            job['predictions'] = str(root / 'predictions/lerf' / job['scene'])
        lerf['status'] = 'fixed protected-native full cohort; no default promotion'
        write(root / 'configs/lerf.json', lerf)
        for job in cohort['nvos']:
            pred = read(job['prediction_config']); scoring = read(job['scoring_config'])
            pred['jobs'][0]['field'] = job['field']
            job['prediction_config'] = str(root / 'configs' / (job['name'] + '_prediction.json'))
            job['scoring_config'] = str(root / 'configs' / (job['name'] + '_scoring.json'))
            scoring['prediction_config'] = job['prediction_config']
            write(job['prediction_config'], pred); write(job['scoring_config'], scoring)
        write(root / 'cohort.json', cohort); arms[arm] = receipt(root / 'cohort.json')
    code = list(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(fields=fields, arms=arms, protocol=receipt(PROTOCOL),
        previous_design=receipt(PREVIOUS / 'design.json'), previous_summary=receipt(PREVIOUS / 'summary.json'),
        previous_code_commit='dfd107b', code=[receipt(p) for p in sorted(code)],
        configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')], query_program=PROGRAM,
        relation=dict(epochs=2, seed=20261008, learning_rate=.001, parameter_policy='group_only'),
        functional=dict(epochs=2, seed=20261009, learning_rate=.0001, pairs=64),
        source_only=True, benchmark_scoring_started=False, primary_arm='functional_native',
        model_promotion=False, per_task_model_selection=False))


def frozen_design():
    design = read(ROOT / 'design.json')
    for r in [design['protocol'], design['previous_design'], design['previous_summary']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(r)
    return design


@torch.no_grad()
def rollback(job, output):
    """Keep learned Gaussian relations, restore the original shared decoder."""
    names = {'scale_cutoffs', 'log_temperature', 'affinity_bias'}
    check(job['parent_manifest'])
    check(job['parent_field'])
    state = torch.load(job['parent_field']['path'], map_location='cpu', weights_only=True)
    shared = {k: state[k] for k in names}; del state; gc.collect()
    directory = Path(job['cross_view_field']['path']).parent
    field, manifest = load_field(directory, 'cpu')
    if manifest['files']['field.pt']['sha256'] != job['cross_view_field']['sha256']:
        raise ValueError('cross-view field differs from frozen input')
    changed = []
    for name, value in field.named_parameters():
        if name in names:
            changed.append(name); value.copy_(shared[name])
    if set(changed) != names:
        raise ValueError('shared member parameter schema changed')
    result = save_field(field, output / 'field', dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        shared_parameters_field_sha256=job['parent_field']['sha256'],
        observations_sha256=sha256(Path(job['observations']) / 'observations.json'),
        intervention='restore original shared acceptance parameters, retain cross-view Gaussian groups',
        additional_3d_labels=False, benchmark_queries_used=False))
    write(output / 'training.json', dict(arm='decoder_rollback', changed_parameters=sorted(names),
        export_sha256=result['files']['field.pt']['sha256'], source_access_at_query=False,
        construction_training_steps=0))


def fit(design, index, arm):
    job = design['fields'][index]; output = ROOT / 'fits' / arm / job['key']
    if arm == 'decoder_rollback':
        rollback(job, output)
    elif arm == 'protected_training':
        from radio_gs.query_field.relation_training import fit as relation_fit
        check(job['parent_manifest'])
        relation_fit(job['parent'], job['observations'], output, arm='cross_view_anchor', **design['relation'])
    else:
        from radio_gs.query_field.functional_training import fit as functional_fit
        parent = ROOT / 'fits/protected_training' / job['key'] / 'field'
        functional_fit(parent, job['observations'], output, **design['functional'])


def build(design, gpus):
    def action(index, gpu):
        job = design['fields'][index]
        for arm in ARMS:
            out = ROOT / 'fits' / arm / job['key']
            if (out / 'training.json').exists():
                if sha256(out / 'field/field.pt') != read(out / 'training.json')['export_sha256']:
                    raise ValueError('completed export changed')
                continue
            execute(['-m', 'scripts.research.run_query_field_protected', 'fit', '--arm', arm, '--index', str(index)],
                    ROOT / 'logs/build' / (arm + '_' + job['key'] + '.log'), gpu)
            print('fit complete', arm, job['key'], 'GPU', gpu, flush=True)
    lanes(list(range(len(design['fields']))), gpus, action)
    records = []
    for job in design['fields']:
        previous = read(job['cross_view_training']['path']); check(job['cross_view_training'])
        fitted = read(ROOT / 'fits/protected_training' / job['key'] / 'training.json')
        if fitted['trace'] != previous['trace'] or not fitted['protected_buffers_unchanged'] or not fitted['protected_member_parameters_unchanged']:
            raise ValueError('protected group fit changed matched source exposure or protected state')
        native = read(ROOT / 'fits/functional_native' / job['key'] / 'training.json')
        if not native['protected_state_unchanged']:
            raise ValueError('native fit changed protected state')
        records.append(dict(key=job['key'], exact_previous_relation_trace=True,
            reports=[receipt(ROOT / 'fits' / a / job['key'] / 'training.json') for a in ARMS],
            exports=[receipt(ROOT / 'fits' / a / job['key'] / 'field/manifest.json') for a in ARMS]))
    write(ROOT / 'construction_complete.json', dict(records=records, arms=list(ARMS), field_count=len(records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'build', 'predict', 'score', 'all'])
    parser.add_argument('--gpus', default='1,2,3,4')
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--index', type=int)
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare()
    else:
        design = frozen_design(); gpus = [int(g) for g in args.gpus.split(',')]
        common.ROOT = ROOT; common.ARMS = ARMS
        if args.stage == 'fit':
            fit(design, args.index, args.arm)
        else:
            for stage, function in [('build', build), ('predict', common.predict), ('score', common.score)]:
                if args.stage in {stage, 'all'} and not (ROOT / dict(build='construction_complete.json',
                        predict='predictions_complete.json', score='scoring_complete.json')[stage]).exists():
                    function(design, gpus)
