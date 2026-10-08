"""Two matched source-training arms, complete five-family prediction and scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from radio_gs.query_field.storage import sha256

ARMS = ('local_anchor_control', 'cross_view_anchor')
PARENT = Path('configs/query_field/five_benchmark_v1/cohort.json')
PROTOCOL = Path('docs/experiments/2026-10-08-cross-view-relation-protocol.md')
ROOT = Path('output/query_field_20261008/cross_view_relations_v1')
CODE = [Path(__file__), Path('radio_gs/query_field/relation_training.py'),
        *[Path('radio_gs/query_field') / (x + '.py') for x in
          ['field', 'storage', 'observations', 'geometry', 'public_lerf', 'public_nvos', 'public_scannet', 'public_uqis']]]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def receipt(path):
    return dict(path=str(path), sha256=sha256(path))


def check(entry):
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('frozen input changed: ' + entry['path'])


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    parent = read(PARENT)
    fields, seen = [], {}
    for family in ['lerf', 'nvos', 'scannet', 'uqis']:
        for job in parent[family]:
            if job['field'] in seen:
                continue
            name = job.get('name', job.get('scene'))
            key = family + '_' + name
            seen[job['field']] = key
            fields.append(dict(key=key, family=family, scene=name, parent=job['field'],
                parent_field=receipt(Path(job['field']) / 'field.pt'),
                parent_manifest=receipt(Path(job['field']) / 'manifest.json'),
                observations=str(Path(job['field']).parent.parent / 'observations')))
    if len(fields) != 30:
        raise ValueError('distinct field cohort changed')
    arms = {}
    original_lerf = {j['name']: j for j in read('configs/pipeline/system_v10.json')['jobs'] if j['benchmark'] == 'lerf'}
    template = read(Path(parent['lerf'][-1]['predictions']) / 'prediction.json')['configuration']
    for arm in ARMS:
        arm_root = ROOT / arm
        config = copy.deepcopy(parent)
        config['root'] = str(arm_root)
        for family in ['lerf', 'nvos', 'scannet', 'uqis']:
            for job in config[family]:
                key = seen[job['field']]
                job['parent_field'] = job['field']
                job['field'] = str(ROOT / 'fits' / arm / key / 'field')
        lerf = copy.deepcopy(template)
        lerf['jobs'] = []
        lerf['status'] = 'fixed full-cohort relation training; no default promotion'
        for job in config['lerf']:
            name = job['scene']
            source = original_lerf[name]
            queries = read(source['queries'])['queries']
            old = read(Path(job['predictions']) / 'prediction.json')
            if [q['text'] for q in queries] != [q['text'] for q in old['diagnostics']]:
                raise ValueError('LERF query order changed: ' + name)
            lerf['jobs'].append(dict(name=name, scene=name, field=job['field'], queries=source['queries'],
                                     evaluation_scene=old['job']['evaluation_scene']))
            job['predictions'] = str(arm_root / 'predictions/lerf' / name)
        write(arm_root / 'configs/lerf.json', lerf)
        for job in config['nvos']:
            pred = read(job['prediction_config'])
            if len(pred['jobs']) != 1:
                raise ValueError('NVOS baseline arm count differs')
            pred['jobs'][0]['field'] = job['field']
            pred_path = arm_root / 'configs' / (job['name'] + '_prediction.json')
            score_path = arm_root / 'configs' / (job['name'] + '_scoring.json')
            scoring = read(job['scoring_config']); scoring['prediction_config'] = str(pred_path)
            write(pred_path, pred); write(score_path, scoring)
            job['prediction_config'] = str(pred_path); job['scoring_config'] = str(score_path)
        write(arm_root / 'cohort.json', config)
        arms[arm] = receipt(arm_root / 'cohort.json')
    design = dict(parent=receipt(PARENT), protocol=receipt(PROTOCOL), code=[receipt(p) for p in CODE],
                  fields=fields, arms=arms, epochs=2, seed=20261008, learning_rate=.001,
                  configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')],
                  benchmark_scoring_started=False, source_only=True)
    write(ROOT / 'design.json', design)


def frozen_design():
    design = read(ROOT / 'design.json')
    for entry in [design['parent'], design['protocol']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design


def execute(command, log, gpu=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    # Preserve failed attempts as well as successful retries.
    index = 0
    while log.exists():
        index += 1; log = log.with_name(log.stem.split('.attempt')[0] + f'.attempt{index}.log')
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log.open('x') as handle:
        subprocess.run([sys.executable, *command], env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def lanes(tasks, gpus, action):
    chunks = [tasks[i::len(gpus)] for i in range(len(gpus))]
    def lane(i):
        for task in chunks[i]:
            action(task, gpus[i])
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))


def build(design, gpus):
    tasks = [(arm, i) for i in range(len(design['fields'])) for arm in ARMS]
    def action(task, gpu):
        arm, index = task; job = design['fields'][index]
        out = ROOT / 'fits' / arm / job['key']
        if (out / 'training.json').exists():
            report = read(out / 'training.json')
            if sha256(out / 'field/field.pt') != report['export_sha256']:
                raise ValueError('completed fit changed')
            return
        execute(['-m', 'scripts.research.run_query_field_relations', 'fit', '--arm', arm, '--index', str(index)],
                ROOT / 'logs/build' / (arm + '_' + job['key'] + '.log'), gpu)
        print('fit complete', arm, job['key'], 'GPU', gpu, flush=True)
    lanes(tasks, gpus, action)
    records = []
    for job in design['fields']:
        reports = [read(ROOT / 'fits' / a / job['key'] / 'training.json') for a in ARMS]
        if reports[0]['trace'] != reports[1]['trace']:
            raise ValueError('paired source/target/anchor trace differs')
        for r in reports:
            if not r['protected_buffers_unchanged'] or r['steps'] != 2 * r['training_regions']:
                raise ValueError('training contract differs')
        records.append(dict(key=job['key'], steps=reports[0]['steps'],
            cross_view_steps=reports[0]['cross_view_steps'], distinct_anchor_steps=reports[0]['distinct_anchor_steps'],
            training=[receipt(ROOT / 'fits' / a / job['key'] / 'training.json') for a in ARMS],
            fields=[receipt(ROOT / 'fits' / a / job['key'] / 'field/field.pt') for a in ARMS]))
    write(ROOT / 'construction_complete.json', dict(records=records, paired_traces_exact=True))


def verify_predictions(design):
    records = []
    for arm in ARMS:
        cohort = read(design['arms'][arm]['path']); root = ROOT / arm / 'predictions'
        for family in ['lerf', 'nvos', 'scannet']:
            for job in cohort[family]:
                name = job.get('name', job.get('scene'))
                path = root / family / name
                if family == 'nvos': path = path / name
                report = read(path / 'prediction.json')
                if sha256(path / 'predictions.pt') != report['predictions_sha256']:
                    raise ValueError('prediction changed')
                records.append(receipt(path / 'prediction.json'))
        expected = set()
        for job in cohort['uqis']:
            path = root / 'uqis/receipts' / (job['name'] + '.json')
            report = read(path)
            for q in report['queries']:
                expected.add(q['query_id'])
                for folder, suffix, key in [('predictions', '.npy', 'prediction_sha256'),
                        ('selections', '.npy', 'selection_sha256'), ('gaussians', '.pt', 'gaussian_sha256')]:
                    if sha256(root / 'uqis' / folder / (q['query_id'] + suffix)) != q[key]:
                        raise ValueError('UQIS prediction changed')
            records.append(receipt(path))
        if len(expected) != 268 or {p.stem for p in (root / 'uqis/predictions').glob('*.npy')} != expected:
            raise ValueError('UQIS complete denominator differs')
    return records


def predict(design, gpus):
    read(ROOT / 'construction_complete.json')
    tasks = [(a, f, j) for a in ARMS for f in ['lerf', 'nvos', 'scannet', 'uqis']
             for j in read(design['arms'][a]['path'])[f]]
    def action(task, gpu):
        arm, family, job = task
        cohort = read(design['arms'][arm]['path']); root = ROOT / arm
        name = job.get('name', job.get('scene'))
        out = root / 'predictions' / family
        if family == 'lerf':
            complete = out / name / 'prediction.json'
            command = ['-m', 'radio_gs.query_field.public_lerf', 'predict', '--config', str(root / 'configs/lerf.json'),
                       '--job', name, '--output', str(out)]
        elif family == 'nvos':
            out = out / name; complete = out / name / 'prediction.json'
            command = ['-m', 'radio_gs.query_field.public_nvos', 'predict', '--config', job['prediction_config'],
                       '--job', name, '--output', str(out)]
        elif family == 'scannet':
            out = out / name; complete = out / 'prediction.json'
            command = ['-m', 'radio_gs.query_field.public_scannet', 'predict', '--field', job['field'],
                       '--queries', job['queries'], '--encoder', cohort['encoder'], '--output', str(out)]
        else:
            complete = out / 'receipts' / (name + '.json')
            command = ['-m', 'radio_gs.query_field.public_uqis', '--field', job['field'], '--scene', name,
                       '--public', str(Path(cohort['uqis_package']) / 'method'), '--encoder', cohort['encoder'], '--output', str(out)]
        if family in {'scannet', 'uqis'} and design.get('query_program', 'original') != 'original':
            command += ['--query-program', design['query_program']]
        if complete.exists(): return
        execute(command, ROOT / 'logs/predict' / (arm + '_' + family + '_' + name + '.log'), gpu)
        print('predicted', arm, family, name, flush=True)
    lanes(tasks, gpus, action)
    write(ROOT / 'predictions_complete.json', dict(records=verify_predictions(design),
          all_arms_all_families_before_scoring=True, design=receipt(ROOT / 'design.json')))


def score(design, gpus):
    barrier = read(ROOT / 'predictions_complete.json'); check(barrier['design'])
    verify_predictions(design)
    tasks = []
    for arm in ARMS:
        cohort = read(design['arms'][arm]['path']); root = ROOT / arm
        for family in ['lerf', 'nvos', 'scannet']:
            for j in cohort[family]: tasks.append((arm, family, j))
        tasks.append((arm, 'uqis', {}))
    def action(task, gpu):
        arm, family, job = task
        cohort = read(design['arms'][arm]['path']); root = ROOT / arm
        name = job.get('name', job.get('scene', 'all'))
        out = root / 'predictions' / family
        if family == 'lerf':
            complete = out / name / 'evaluation/evaluation.json'
            command = ['-m', 'radio_gs.query_field.public_lerf', 'score', '--config', str(root / 'configs/lerf.json'),
                       '--job', name, '--output', str(out)]
        elif family == 'nvos':
            out = out / name; complete = out / 'evaluation.json'
            command = ['-m', 'radio_gs.query_field.public_nvos', 'score', '--config', job['scoring_config'], '--output', str(out)]
        elif family == 'scannet':
            out = out / name; complete = out / 'evaluation.json'
            command = ['-m', 'radio_gs.query_field.public_scannet', 'score', '--predictions', str(out),
                       '--evaluation', job['evaluation'], '--output', str(complete)]
        else:
            complete = out / 'evaluation.json'
            command = ['-m', 'radio_gs.benchmarks.uqis.evaluate', '--package', cohort['uqis_package'],
                       '--predictions', str(out / 'predictions'), '--selections', str(out / 'selections'), '--output', str(complete)]
        if complete.exists(): return
        execute(command, ROOT / 'logs/score' / (arm + '_' + family + '_' + name + '.log'), gpu)
        print('scored', arm, family, name, flush=True)
    lanes(tasks, gpus, action)
    write(ROOT / 'scoring_complete.json', dict(arms=list(ARMS), families=['lerf2d', 'lerf3d', 'nvos', 'scannet', 'uqis']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'build', 'fit', 'predict', 'score'])
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    else:
        design = frozen_design()
        if args.stage == 'fit':
            import torch
            from radio_gs.query_field.relation_training import fit
            torch.set_num_threads(4)
            job = design['fields'][args.index]
            check(job['parent_field']); check(job['parent_manifest'])
            result = fit(job['parent'], job['observations'], ROOT / 'fits' / args.arm / job['key'],
                         arm=args.arm, epochs=design['epochs'], seed=design['seed'], learning_rate=design['learning_rate'])
            print(json.dumps({k: v for k, v in result.items() if k not in {'trace', 'history'}}, indent=2))
        else:
            {'build': build, 'predict': predict, 'score': score}[args.stage](design, args.gpus)
