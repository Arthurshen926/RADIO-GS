"""Compile one functional reference and evaluate the complete five-task cohort."""
import argparse
import copy
from pathlib import Path

import torch

from radio_gs.query_field.query_program import PROGRAM
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes
from scripts.research.run_query_field_protected import ROOT as PRIOR

ROOT = Path('output/query_field_20261009/paired_functions_v1')
ARMS = ('paired_function',)
PROTOCOL = Path('docs/experiments/2026-10-09-paired-function-protocol.md')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    previous = read(PRIOR / 'design.json'); fields = copy.deepcopy(previous['fields'])
    lookup = {j['parent']: j['key'] for j in fields}
    for job in fields:
        job['parent'] = str(PRIOR / 'fits/protected_training' / job['key'] / 'field')
        job['parent_manifest'] = receipt(Path(job['parent']) / 'manifest.json')
        job['parent_field'] = receipt(Path(job['parent']) / 'field.pt')
    arm = ARMS[0]; root = ROOT / arm
    cohort = read(PRIOR / 'protected_training/cohort.json'); cohort['root'] = str(root)
    for family in ['lerf', 'nvos', 'scannet', 'uqis']:
        for job in cohort[family]:
            job['field'] = str(ROOT / 'fits' / arm / lookup[job['parent_field']] / 'field')
    lerf = read(PRIOR / 'protected_training/configs/lerf.json')
    for job in cohort['lerf']:
        next(j for j in lerf['jobs'] if j['name'] == job['scene'])['field'] = job['field']
        job['predictions'] = str(root / 'predictions/lerf' / job['scene'])
    lerf['status'] = 'uncompressed same-source relation functional reference; no promotion'
    write(root / 'configs/lerf.json', lerf)
    for job in cohort['nvos']:
        pred = read(job['prediction_config']); score = read(job['scoring_config'])
        pred['jobs'][0]['field'] = job['field']
        job['prediction_config'] = str(root / 'configs' / (job['name'] + '_prediction.json'))
        job['scoring_config'] = str(root / 'configs' / (job['name'] + '_scoring.json'))
        score['prediction_config'] = job['prediction_config']
        write(job['prediction_config'], pred); write(job['scoring_config'], score)
    write(root / 'cohort.json', cohort)
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(fields=fields, arms={arm: receipt(root / 'cohort.json')},
        protocol=receipt(PROTOCOL), code=[receipt(p) for p in code],
        prior_summary=receipt('output/query_field_20261008/joint_element_relations_v1/summary.json'),
        configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')], query_program=PROGRAM,
        source_only=True, primary_arm=arm, model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, training_steps=0, prior_evidence_mass=1., uncompressed_reference=True))


def frozen_design():
    design = read(ROOT / 'design.json')
    for entry in [design['protocol'], design['prior_summary']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design


def build(design, workers):
    def action(index, lane):
        job = design['fields'][index]; out = ROOT / 'fits' / ARMS[0] / job['key']
        if (out / 'construction.json').exists():
            check(dict(path=str(out / 'field/field.pt'), sha256=read(out / 'construction.json')['export_sha256']))
            return
        execute(['-m', 'scripts.research.run_query_field_paired_functions', 'compile', '--index', str(index)],
            ROOT / 'logs/build' / (job['key'] + '.log'))
        print('compiled functional field', job['key'], flush=True)
    lanes(list(range(len(design['fields']))), list(range(workers)), action)
    records = []
    for job in design['fields']:
        out = ROOT / 'fits' / ARMS[0] / job['key']; report = read(out / 'construction.json')
        if not report['protected_state_unchanged'] or report['parent_sha256'] != job['parent_field']['sha256']:
            raise ValueError('functional reference changed its parent')
        if not report['replay'] or max(max(r['positive_error'], r['negative_error']) for r in report['replay']) > 1e-5:
            raise ValueError('partial function preservation failed')
        records.append(dict(key=job['key'], report=receipt(out / 'construction.json'), export=receipt(out / 'field/manifest.json')))
    if len(records) != 30: raise ValueError('incomplete construction cohort')
    write(ROOT / 'construction_complete.json', dict(fields=30, records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'compile', 'build', 'predict', 'score', 'validate', 'all'])
    parser.add_argument('--gpus', default='0,2,3,4,5'); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--index', type=int); args = parser.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare': prepare()
    else:
        design = frozen_design(); gpus = list(map(int, args.gpus.split(',')))
        common.ROOT = ROOT; common.ARMS = ARMS
        if args.stage == 'compile':
            from radio_gs.query_field.compile_paired_evidence import compile_field
            job = design['fields'][args.index]; check(job['parent_manifest'])
            report = compile_field(job['parent'], job['observations'], ROOT / 'fits' / ARMS[0] / job['key'])
            if report['parent_sha256'] != job['parent_field']['sha256']: raise ValueError('parent payload changed')
        else:
            if args.stage in {'build', 'all'}: build(design, args.workers)
            if args.stage in {'predict', 'all'}: common.predict(design, gpus)
            if args.stage in {'score', 'all'}: common.score(design, gpus)
            if args.stage in {'validate', 'all'}:
                from scripts.research import validate_query_field_protected as cold
                cold.ROOT = ROOT; cold.ARMS = ARMS; cold.main()
