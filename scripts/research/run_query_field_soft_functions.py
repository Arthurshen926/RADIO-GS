"""Same-support, confidence-weighted relation reference on the full cohort."""
import argparse
import copy
from pathlib import Path

import torch

from radio_gs.query_field.query_program import PROGRAM
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes
from scripts.research.run_query_field_paired_functions import ROOT as HARD

ROOT = Path('output/query_field_20261009/soft_paired_functions_v1')
ARMS = ('soft_paired_function',)
PROTOCOL = Path('docs/experiments/2026-10-09-soft-paired-function-protocol.md')


def rewrite(value):
    if isinstance(value, dict): return {k: rewrite(v) for k, v in value.items()}
    if isinstance(value, list): return [rewrite(v) for v in value]
    if isinstance(value, str) and value.startswith(str(HARD) + '/'):
        return value.replace(str(HARD), str(ROOT), 1).replace('/paired_function', '/soft_paired_function', 1)
    return value


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    previous = read(HARD / 'design.json'); read(HARD / 'summary.json')
    decision = HARD / 'diagnostics/soft_coverage_decision.json'
    if read(decision)['benchmark_prediction_started']: raise ValueError('coverage follow-up was not fixed before prediction')
    fields = copy.deepcopy(previous['fields'])
    for job in fields:
        job['hard_reference'] = str(HARD / 'fits/paired_function' / job['key'] / 'field')
        job['hard_manifest'] = receipt(Path(job['hard_reference']) / 'manifest.json')
    cohort = rewrite(read(HARD / 'paired_function/cohort.json'))
    write(ROOT / ARMS[0] / 'cohort.json', cohort)
    for path in (HARD / 'paired_function/configs').glob('*.json'):
        write(ROOT / ARMS[0] / 'configs' / path.name, rewrite(read(path)))
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(fields=fields, arms={ARMS[0]: receipt(ROOT / ARMS[0] / 'cohort.json')},
        protocol=receipt(PROTOCOL), code=[receipt(p) for p in code], prior_summary=receipt(HARD / 'summary.json'),
        coverage_decision=receipt(decision), coverage_preflight=receipt(HARD / 'diagnostics/soft_coverage_preflight.json'),
        configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')], query_program=PROGRAM,
        source_only=True, primary_arm=ARMS[0], model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, training_steps=0, prior_evidence_mass=1., uncompressed_reference=True,
        hard_reference_root=str(HARD), implementation_commit='recorded by frozen code receipts')))


def frozen_design():
    design = read(ROOT / 'design.json')
    for entry in [design[k] for k in ['protocol', 'prior_summary', 'coverage_decision', 'coverage_preflight']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design


def build(design, workers):
    def action(index, lane):
        job = design['fields'][index]; out = ROOT / 'fits' / ARMS[0] / job['key']
        if (out / 'construction.json').exists():
            check(dict(path=str(out / 'field/field.pt'), sha256=read(out / 'construction.json')['export_sha256']))
            return
        execute(['-m', 'scripts.research.run_query_field_soft_functions', 'compile', '--index', str(index)],
            ROOT / 'logs/build' / (job['key'] + '.log'))
        print('compiled soft functional field', job['key'], flush=True)
    lanes(list(range(len(design['fields']))), list(range(workers)), action)
    records = []
    for job in design['fields']:
        out = ROOT / 'fits' / ARMS[0] / job['key']; report = read(out / 'construction.json')
        if not report['exact_hard_association_coordinates'] or not report['protected_state_unchanged']:
            raise ValueError('soft extension changed fixed conditions')
        if report['parent_sha256'] != job['parent_field']['sha256']: raise ValueError('parent changed')
        records.append(dict(key=job['key'], report=receipt(out / 'construction.json'), export=receipt(out / 'field/manifest.json')))
    if len(records) != 30: raise ValueError('incomplete construction')
    write(ROOT / 'construction_complete.json', dict(fields=30, records=records))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'compile', 'build', 'predict', 'score', 'validate', 'all'])
    p.add_argument('--gpus', default='0,2,3,4,5'); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--index', type=int); args = p.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare': prepare()
    else:
        design = frozen_design(); gpus = list(map(int, args.gpus.split(',')))
        common.ROOT = ROOT; common.ARMS = ARMS
        if args.stage == 'compile':
            from radio_gs.query_field.compile_soft_paired_evidence import compile_field
            job = design['fields'][args.index]; check(job['parent_manifest']); check(job['hard_manifest'])
            r = compile_field(job['parent'], job['observations'], ROOT / 'fits' / ARMS[0] / job['key'], job['hard_reference'])
            if r['parent_sha256'] != job['parent_field']['sha256']: raise ValueError('parent changed')
        else:
            if args.stage in {'build', 'all'}: build(design, args.workers)
            if args.stage in {'predict', 'all'}: common.predict(design, gpus)
            if args.stage in {'score', 'all'}: common.score(design, gpus)
            if args.stage in {'validate', 'all'}:
                from scripts.research import validate_query_field_protected as cold
                cold.ROOT = ROOT; cold.ARMS = ARMS; cold.main()
