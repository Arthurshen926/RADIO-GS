"""Matched support-function repair, complete construction/prediction barriers."""
import argparse
import copy
from pathlib import Path
import subprocess

import torch

from radio_gs.query_field.query_program import PROGRAM
from radio_gs.query_field.research_storage import (research_writer,storage_binding,verify_storage_binding,validate_research_package,method_binding,runner_storage_args)
from radio_gs.query_field.storage import field_identity
STORE_ROOT=None
LEGACY_V1=False
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes
from scripts.research.run_query_field_soft_functions import ROOT as PREVIOUS

ROOT = Path('output/query_field_20261009/confidence_mixture_functions_v1')
ARMS = ('confidence_mixture_function',)
PROTOCOL = Path('docs/experiments/2026-10-09-confidence-mixture-function-protocol.md')


def rewrite(value):
    if isinstance(value, dict): return {k: rewrite(v) for k, v in value.items()}
    if isinstance(value, list): return [rewrite(v) for v in value]
    if isinstance(value, str) and value.startswith(str(PREVIOUS) + '/'):
        return value.replace(str(PREVIOUS), str(ROOT), 1).replace('/soft_paired_function', '/confidence_mixture_function', 1)
    return value


def prepare():
    research_writer(store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    ROOT.mkdir(parents=True, exist_ok=False)
    stopped = Path('output/query_field_20261009/mixture_functions_v1/source_counterexample_stop.json')
    if read(stopped)['benchmark_scoring_started']: raise ValueError('repair must precede scoring')
    previous = read(PREVIOUS / 'design.json'); read(PREVIOUS / 'summary.json')
    fields = copy.deepcopy(previous['fields'])
    for job in fields:
        job['protected_parent'] = job['parent']
        job['parent'] = str(PREVIOUS / 'fits/soft_paired_function' / job['key'] / 'field')
        job['parent_manifest'] = receipt(Path(job['parent']) / 'manifest.json')
        manifest = read(Path(job['parent']) / 'manifest.json')
        job['parent_binding']=method_binding(job['parent'],manifest)
        job['parent_field'] = (dict(path=str(Path(job['parent'])/'field.pt'),sha256=field_identity(manifest)) if manifest['schema'].endswith('.v1') else dict(state_identity=field_identity(manifest)))
    cohort = rewrite(read(PREVIOUS / 'soft_paired_function/cohort.json'))
    write(ROOT / ARMS[0] / 'cohort.json', cohort)
    for path in (PREVIOUS / 'soft_paired_function/configs').glob('*.json'):
        write(ROOT / ARMS[0] / 'configs' / path.name, rewrite(read(path)))
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(storage=storage_binding(store_root=STORE_ROOT,legacy_v1=LEGACY_V1),fields=fields, arms={ARMS[0]: receipt(ROOT / ARMS[0] / 'cohort.json')},
        protocol=receipt(PROTOCOL), code=[receipt(p) for p in code], prior_summary=receipt(PREVIOUS / 'summary.json'),
        configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')], query_program=PROGRAM,
        source_only=True, primary_arm=ARMS[0], model_promotion=False, independent_final_test=False,
        per_task_model_selection=False, training_steps=0, prior_evidence_mass=1., uncompressed_reference=True,
        exact_parent_tensor_payload=True, additional_tensor_bytes=0, stopped_source_counterexample=receipt(stopped),
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))


def frozen_design():
    design = read(ROOT / 'design.json')
    verify_storage_binding(design,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    for entry in [design[k] for k in ['protocol', 'prior_summary', 'stopped_source_counterexample']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design



def validate_compile(job,out):
    report=read(out/'construction.json')
    if "storage" not in report and LEGACY_V1:
        check(dict(path=str(out/'field/field.pt'),sha256=report['export_sha256']))
        if report['export_sha256']!=job['parent_field']['sha256']:raise ValueError('legacy parent changed')
        return report
    verify_storage_binding(report,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    manifest=validate_research_package(out/'field',store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    if report['export_binding']!=method_binding(out/'field',manifest):raise ValueError('compiled method changed')
    if report['parent_binding']!=method_binding(job['parent']):raise ValueError('parent method changed')
    if job.get('parent_binding',report['parent_binding'])!=report['parent_binding']:raise ValueError('frozen parent changed')
    if not report['exact_tensor_payload']:raise ValueError('tensor intervention leaked into function-only control')
    return report


def build(design, workers):
    def action(index, lane):
        job = design['fields'][index]; out = ROOT / 'fits' / ARMS[0] / job['key']
        if (out / 'construction.json').exists():
            validate_compile(job,out)
            return
        execute(['-m', 'scripts.research.run_query_field_confidence_mixture_functions', 'compile', '--index', str(index)]+runner_storage_args(ROOT,store_root=STORE_ROOT,legacy_v1=LEGACY_V1),
                ROOT / 'logs/build' / (job['key'] + '.log'))
        print('compiled mixture functional field', job['key'], flush=True)
    lanes(list(range(len(design['fields']))), list(range(workers)), action)
    records = []
    for job in design['fields']:
        out = ROOT / 'fits' / ARMS[0] / job['key']; report = validate_compile(job,out)
        records.append(dict(key=job['key'], report=receipt(out / 'construction.json'), export=receipt(out / 'field/manifest.json')))
    if len(records) != 30: raise ValueError('incomplete construction')
    write(ROOT / 'construction_complete.json', dict(fields=30, records=records))


def main(argv=None):
    global ROOT,STORE_ROOT,LEGACY_V1
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'compile', 'build', 'predict', 'score', 'validate', 'all'])
    p.add_argument('--gpus', default='0,2,3'); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--index', type=int)
    p.add_argument('--root',help='Required fresh output root for new shared-v2 runs')
    p.add_argument('--store-root');p.add_argument('--legacy-v1',action='store_true')
    args = p.parse_args(argv)
    STORE_ROOT=args.store_root;LEGACY_V1=args.legacy_v1
    research_writer(store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
    if not LEGACY_V1 and args.root is None:raise ValueError('new shared runs require explicit --root')
    if args.root is not None:ROOT=Path(args.root)
    torch.set_num_threads(4)
    if args.stage == 'prepare': prepare()
    else:
        design = frozen_design(); gpus = list(map(int, args.gpus.split(',')))
        common.ROOT = ROOT; common.ARMS = ARMS
        if args.stage == 'compile':
            from radio_gs.query_field.compile_mixture_evidence import compile_field
            job = design['fields'][args.index]; check(job['parent_manifest'])
            report = compile_field(job['parent'], job['observations'], ROOT / 'fits' / ARMS[0] / job['key'], confidence_weighted=True,store_root=STORE_ROOT,legacy_v1=LEGACY_V1)
            validate_compile(job,ROOT/'fits'/ARMS[0]/job['key'])
        else:
            if args.stage in {'build', 'all'}: build(design, args.workers)
            if args.stage in {'predict', 'all'}: common.predict(design, gpus)
            if args.stage in {'score', 'all'}: common.score(design, gpus)
            if args.stage in {'validate', 'all'}:
                from scripts.research import validate_query_field_protected as cold
                cold.ROOT = ROOT; cold.ARMS = ARMS; cold.main(store_root=STORE_ROOT)

if __name__=='__main__':main()
