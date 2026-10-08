"""Matched joint element/subset-alpha repair of conditional acceptance drift."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from radio_gs.query_field.query_program import PROGRAM
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute
from scripts.research.run_query_field_elements import ROOT as CONTROL

ROOT = Path('output/query_field_20261008/joint_element_relations_v1')
ARMS = ('joint_conditional_relation',)
PROTOCOL = Path('docs/experiments/2026-10-08-joint-relation-protocol.md')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    prior = read(CONTROL / 'design.json'); fields = copy.deepcopy(prior['fields'])
    read(CONTROL / 'summary.json')
    for job in fields:
        job['matched_element_training'] = receipt(CONTROL / 'fits/conditional_relation' / job['key'] / 'training.json')
    lookup = {j['parent']: j['key'] for j in fields}
    arms = {}
    for arm in ARMS:
        root = ROOT / arm; cohort = read(CONTROL / 'conditional_relation/cohort.json'); cohort['root'] = str(root)
        for family in ['lerf', 'nvos', 'scannet', 'uqis']:
            for job in cohort[family]:
                job['field'] = str(ROOT / 'fits' / arm / lookup[job['parent_field']] / 'field')
        lerf = read(CONTROL / 'conditional_relation/configs/lerf.json')
        for job in cohort['lerf']:
            next(j for j in lerf['jobs'] if j['name'] == job['scene'])['field'] = job['field']
            job['predictions'] = str(root / 'predictions/lerf' / job['scene'])
        lerf['status'] = 'matched joint element/rendered relation repair; no promotion'
        write(root / 'configs/lerf.json', lerf)
        for job in cohort['nvos']:
            pred = read(job['prediction_config']); score = read(job['scoring_config'])
            pred['jobs'][0]['field'] = job['field']
            job['prediction_config'] = str(root / 'configs' / (job['name'] + '_prediction.json'))
            job['scoring_config'] = str(root / 'configs' / (job['name'] + '_scoring.json'))
            score['prediction_config'] = job['prediction_config']
            write(job['prediction_config'], pred); write(job['scoring_config'], score)
        write(root / 'cohort.json', cohort); arms[arm] = receipt(root / 'cohort.json')
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(fields=fields, arms=arms, protocol=receipt(PROTOCOL),
        control_summary=receipt(CONTROL / 'summary.json'), code=[receipt(p) for p in code],
        configs=[receipt(p) for p in ROOT.glob('*/configs/*.json')], query_program=PROGRAM,
        epochs=2, seed=20261008, learning_rate=.001, per_class=128,
        objective_weights=dict(element=.5, primary_rendered=.25, secondary_rendered=.25),
        source_only=True, primary_arm=ARMS[0], model_promotion=False,
        independent_final_test=False, per_task_model_selection=False))


def frozen_design():
    design = read(ROOT / 'design.json')
    for entry in [design['protocol'], design['control_summary']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design


def fit(design, index):
    from radio_gs.query_field.element_relation_training import fit as element_fit
    job = design['fields'][index]; check(job['parent_manifest'])
    report = element_fit(job['parent'], job['observations'], ROOT / 'fits' / ARMS[0] / job['key'], arm=ARMS[0],
        **{k: design[k] for k in ['epochs', 'seed', 'learning_rate', 'per_class']})
    if report['parent_sha256'] != job['parent_field']['sha256'] or report['rendered_weight'] != .5:
        raise ValueError('joint relation parent/objective differs')


def build(design, gpus):
    large = [i for i, j in enumerate(design['fields']) if j['family'] == 'lerf']
    other = [i for i, j in enumerate(design['fields']) if j['family'] != 'lerf']
    chunks = [large + other] if len(gpus) == 1 else [large] + [other[i::len(gpus)-1] for i in range(len(gpus)-1)]
    def lane(i):
        for index in chunks[i]:
            job = design['fields'][index]; out = ROOT / 'fits' / ARMS[0] / job['key']
            if (out / 'training.json').exists():
                check(dict(path=str(out / 'field/field.pt'), sha256=read(out / 'training.json')['export_sha256']))
                continue
            execute(['-m', 'scripts.research.run_query_field_joint_elements', 'fit', '--index', str(index)],
                    ROOT / 'logs/build' / (job['key'] + '.log'), gpus[i])
            print('joint fit complete', job['key'], flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    records = []
    for job in design['fields']:
        check(job['matched_element_training']); control = read(job['matched_element_training']['path'])
        out = ROOT / 'fits' / ARMS[0] / job['key']; report = read(out / 'training.json')
        for key in ['trace', 'sample_trace_sha256', 'parent_sha256', 'source_frames', 'training_regions',
                    'steps', 'seed', 'epochs', 'learning_rate', 'per_class', 'row_exposure']:
            if report[key] != control[key]:
                raise ValueError('joint intervention changed matched source/element construction: ' + key)
        if not report['protected_state_unchanged'] or report['rendered_weight'] != .5:
            raise ValueError('joint protection/objective changed')
        records.append(dict(key=job['key'], exact_control_source_trace=True, exact_element_samples=True,
            training=receipt(out / 'training.json'), export=receipt(out / 'field/manifest.json')))
    write(ROOT / 'construction_complete.json', dict(records=records, fields=30, exports=30))


def validate():
    from scripts.research import validate_query_field_protected as cold
    cold.ROOT = ROOT; cold.ARMS = ARMS
    cold.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'build', 'predict', 'score', 'validate', 'all'])
    parser.add_argument('--gpus', default='5,0,2,3,4'); parser.add_argument('--index', type=int)
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare()
    else:
        design = frozen_design(); gpus = list(map(int, args.gpus.split(',')))
        common.ROOT = ROOT; common.ARMS = ARMS
        if args.stage == 'fit':
            fit(design, args.index)
        elif args.stage == 'validate':
            validate()
        else:
            for stage, function, marker in [('build', build, 'construction_complete.json'),
                    ('predict', common.predict, 'predictions_complete.json'), ('score', common.score, 'scoring_complete.json')]:
                if args.stage in {stage, 'all'} and not (ROOT / marker).exists():
                    function(design, gpus)
            if args.stage == 'all' and not (ROOT / 'validation/runtime/report.json').exists():
                validate()
