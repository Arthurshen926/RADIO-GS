"""One entry point for the TMM study; default commands never run inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .registry import REPO, REGISTRY, export, inspect, read, replay_plan, require, sha


def run(command, log=None, cwd=REPO):
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(REPO))
    if log is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)


def report(catalog):
    print(f"{catalog['method']} | reference: retained v10")
    print(f"{catalog['complete_studies']} complete studies; 31 jobs and 268 UQIS queries each")
    print(f"{len(catalog['metrics'])} candidate metric rows; {catalog['historical_sources_verified']} historical receipts verified")
    for item in catalog['experiments']:
        print(f"{item['id']} {item['run']}: {item['status']}; {item['prediction_storage']}")
    print(f"{len(catalog['audits'])} audits (separate from complete shared-method studies)")
    for item in catalog.get('targeted_studies', []):
        print(f"{item['id']} {item['run']}: {item['status']}; "
              f"{item['intervention_queries']} intervened, {item['freshly_scored_queries']} scored UQIS queries; "
              'other benchmarks retained references')
    for item in catalog.get('learning_studies', []):
        print(f"{item['id']} {item['run']}: {item['status']}; {item['stage']}; full-five transfer not yet closed")
    for item in catalog.get('reference_studies', []):
        print(f"{item['id']} {item['run']}: {item['status']}; targeted NVOS; other outputs unchanged references")
    for warning in catalog['warnings']:
        print('LIMITATION: ' + warning)
    print('Shared admission transfer remains open. Checks do not re-score dense predictions.')


def refresh(registry):
    scripts = [['scripts/research/build_tmm_evidence.py']]
    scripts += [['scripts/research/summarize_tmm_response_study.py', '--run', e['run']]
                for e in registry['experiments']]
    if registry.get('targeted_studies'):
        scripts += [['scripts/research/summarize_tmm_query_witness_anchor.py']]
    if any(e['id']=='Q02' for e in registry.get('targeted_studies',[])):
        scripts += [['scripts/research/summarize_tmm_complete_response_verification.py']]
    learning = registry.get('learning_studies', [])
    initial_learning = [e for e in learning if e['id'] in ('L01', 'L02')]
    if initial_learning and all((REPO/registry['study_root']/e['run']/'summary.json').exists() for e in initial_learning):
        scripts += [['scripts/research/summarize_tmm_response_replacement.py']]
    joint_learning = [e for e in learning if e['id'] in ('L03', 'L04')]
    if len(joint_learning) == 2 and all((REPO/registry['study_root']/e['run']/'summary.json').exists() for e in joint_learning):
        scripts += [['scripts/research/summarize_tmm_joint_replacement.py']]
    value_learning = [e for e in learning if e['id'] in ('L05', 'L06')]
    if len(value_learning) == 2 and all((REPO/registry['study_root']/e['run']/'summary.json').exists() for e in value_learning):
        scripts += [['scripts/research/summarize_tmm_joint_replacement.py', '--value']]
    if all((REPO/registry['study_root']/r/'summary.json').exists()
           for r in ('source_decision_audit_v1', 'cross_task_targets_v1')):
        scripts += [['scripts/research/summarize_tmm_cross_task.py']]
    if (REPO/registry['study_root']/'category_native_responses_v1/summary.json').exists():
        scripts += [['scripts/research/summarize_tmm_category_supervision.py']]
    if (REPO/registry['study_root']/'category_transfer_inputs_v2/summary.json').exists():
        scripts += [['scripts/research/summarize_tmm_multitask_transfer.py']]
    if (REPO/registry['study_root']/'response_value_multitask_v1/summary.json').exists():
        scripts += [['scripts/research/summarize_tmm_utility_diagnosis.py']]
    if (REPO/registry['study_root']/'residual_utility_audit_v1/summary.json').exists():
        scripts += [['scripts/research/summarize_tmm_residual_value.py']]
    for item in registry.get('reference_studies',[]):
        if (REPO/item.get('study_root',registry['study_root'])/item['run']/'summary.json').exists():
            builders={'N01':'coverage','N02':'source_readout','N03':'local_credit','N04':'image_response','N05':'partial_response','N06':'unboxed_response','N07':'memory_response','N08':'native_writeback','N09':'signed_affinity','N10':'direct_affinity','N11':'instance_extent','N12':'reference_completion','N13':'reference_delta','N14':'reference_admission','N15':'multiview_completion','N16':'target_redecode'}
            scripts += [[f"scripts/research/summarize_tmm_nvos_{builders[item['id']]}.py"]]
    for item in registry.get('audits',[]):
        if item['id']=='A24' and (REPO/item['study_root']/item['path']).exists():
            scripts += [['scripts/research/summarize_tmm_nvos_score_precision.py']]
    scripts += [['scripts/research/build_tmm_paper_experiments.py'],
                ['scripts/research/summarize_tmm_response_study.py', '--compare'],
                ['scripts/research/build_tmm_method.py']]
    logs = REPO / registry['paper_root'] / 'build/logs'
    for index, command in enumerate(scripts):
        print('Build:', ' '.join(command), flush=True)
        run([sys.executable, *command], logs / f'refresh_{index:02d}.log')
    report(export())


def build_paper(registry):
    from pypdf import PdfReader
    catalog = inspect()
    paper = REPO / registry['paper_root']
    (paper / 'build').mkdir(exist_ok=True)
    pdfs = {}
    for name in ['main', 'diagnostics', 'replacement']:
        for iteration in range(2):
            run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                 '-output-directory=build', name + '.tex'],
                paper / f'build/logs/{name}_{iteration}.log', cwd=paper)
        path = paper / f'build/{name}.pdf'
        log = (paper / f'build/{name}.log').read_text(errors='replace')
        require('undefined references' not in log and 'Rerun to get' not in log,
                f'{name}: unresolved references')
        pdfs[name] = dict(pages=len(PdfReader(path).pages), bytes=path.stat().st_size,
                          sha256=sha(path), unresolved_references=False,
                          latex_warnings='LaTeX Warning:' in log)
    manifest = read(paper / 'method_manifest.json')
    validation = dict(complete=True, method_manifest_sha256=sha(paper / 'method_manifest.json'),
        historical_files_verified=catalog['historical_sources_verified'],
        complete_studies_verified=catalog['complete_studies'],
        code_bindings_verified=len(manifest['code_bindings']), pdfs=pdfs,
        scientific_default_changed=False, build_runs_benchmark_inference=False, new_performance_claim=False,
        source_version_warnings=catalog['warnings'])
    (paper / 'method_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['status', 'verify', 'results', 'refresh', 'paper', 'test']:
        sub.add_parser(name)
    replay = sub.add_parser('reproduce', help='plan a bounded full-cohort historical replay')
    replay.add_argument('experiment')
    replay.add_argument('--run', required=True, help='new unique run name')
    replay.add_argument('--execute', action='store_true', help='run inference; otherwise only print plan')
    args = parser.parse_args()
    registry = read(REPO / REGISTRY)
    if args.command in ['status', 'verify']:
        report(inspect())
    elif args.command == 'results':
        report(export())
    elif args.command == 'refresh':
        refresh(registry)
    elif args.command == 'paper':
        build_paper(registry)
    elif args.command == 'test':
        run([sys.executable, '-m', 'pytest', '-q', *registry['tests']])
    elif args.command == 'reproduce':
        plan = replay_plan(args.experiment, args.run)
        print(json.dumps(plan, indent=2), flush=True)
        if args.execute:
            run(plan['command'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'Study command failed: {error}')
