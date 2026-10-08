"""Complete direct-native-coordinate construction and five-family evaluation."""
import argparse
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

from radio_gs.query_field.query_program import PROGRAM
from scripts.research import run_query_field_relations as common
from scripts.research.run_query_field_relations import read, write, receipt, check, execute
from scripts.research.run_query_field_protected import ROOT as PARENT_ROOT

ROOT = Path('output/query_field_20261008/native_coordinates_v1')
ARMS = ('native_coordinates',)
PROTOCOL = Path('docs/experiments/2026-10-08-native-coordinate-protocol.md')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    fields = copy.deepcopy(read(PARENT_ROOT / 'design.json')['fields'])
    for job in fields:
        job['input'] = str(PARENT_ROOT / 'fits/protected_training' / job['key'] / 'field')
        job['input_manifest'] = receipt(Path(job['input']) / 'manifest.json')
        job['input_sha256'] = read(job['input_manifest']['path'])['files']['field.pt']['sha256']
    root = ROOT / ARMS[0]
    cohort = read(PARENT_ROOT / 'protected_training/cohort.json'); cohort['root'] = str(root)
    lookup = {j['input']: j['key'] for j in fields}
    for family in ['lerf', 'nvos', 'scannet', 'uqis']:
        for job in cohort[family]:
            job['field'] = str(ROOT / 'fits' / ARMS[0] / lookup[job['field']] / 'field')
    lerf = read(PARENT_ROOT / 'protected_training/configs/lerf.json')
    for job in cohort['lerf']:
        next(j for j in lerf['jobs'] if j['name'] == job['scene'])['field'] = job['field']
        job['predictions'] = str(root / 'predictions/lerf' / job['scene'])
    lerf['status'] = 'fixed full-native coordinate control, no promotion'
    write(root / 'configs/lerf.json', lerf)
    for job in cohort['nvos']:
        pred = read(job['prediction_config']); scoring = read(job['scoring_config'])
        pred['jobs'][0]['field'] = job['field']
        job['prediction_config'] = str(root / 'configs' / (job['name'] + '_prediction.json'))
        job['scoring_config'] = str(root / 'configs' / (job['name'] + '_scoring.json'))
        scoring['prediction_config'] = job['prediction_config']
        write(job['prediction_config'], pred); write(job['scoring_config'], scoring)
    write(root / 'cohort.json', cohort)
    code = sorted(Path('radio_gs/query_field').glob('*.py')) + [Path(__file__), Path(common.__file__)]
    write(ROOT / 'design.json', dict(fields=fields, arms={ARMS[0]: receipt(root / 'cohort.json')},
        parent_construction=receipt(PARENT_ROOT / 'construction_complete.json'), protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in code], configs=[receipt(p) for p in root.glob('configs/*.json')],
        query_program=PROGRAM, epochs=2, seed=20261009, learning_rate=.001, pairs=64,
        source_only=True, benchmark_scores_used_to_choose_recipe=False, model_promotion=False))


def frozen_design():
    design = read(ROOT / 'design.json')
    for entry in [design['parent_construction'], design['protocol']] + design['code'] + design['configs'] + list(design['arms'].values()):
        check(entry)
    return design


def fit(design, index):
    from radio_gs.query_field.coordinate_training import fit as coordinate_fit
    job = design['fields'][index]; check(job['input_manifest'])
    report = coordinate_fit(job['input'], job['observations'], ROOT / 'fits' / ARMS[0] / job['key'],
        **{k: design[k] for k in ['epochs', 'seed', 'learning_rate', 'pairs']})
    if report['parent_sha256'] != job['input_sha256']:
        raise ValueError('coordinate parent differs from frozen recipe')


def build(design, gpus):
    # At most one very large LERF field owns dense CPU moments at a time.
    # Remaining lanes process the smaller rooms; all fields use the same recipe.
    large = [i for i, j in enumerate(design['fields']) if j['family'] == 'lerf']
    remaining = [i for i, j in enumerate(design['fields']) if j['family'] != 'lerf']
    if len(gpus) == 1:
        chunks = [large + remaining]
    else:
        chunks = [large] + [remaining[i::len(gpus)-1] for i in range(len(gpus)-1)]
    def lane(lane_index):
        for index in chunks[lane_index]:
            job = design['fields'][index]; out = ROOT / 'fits' / ARMS[0] / job['key']
            if (out / 'training.json').exists():
                check(dict(path=str(out / 'field/field.pt'), sha256=read(out / 'training.json')['export_sha256']))
                continue
            execute(['-m', 'scripts.research.run_query_field_coordinates', 'fit', '--index', str(index)],
                    ROOT / 'logs/build' / (job['key'] + '.log'), gpus[lane_index])
            print('coordinate fit complete', job['key'], flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    records = []
    for job in design['fields']:
        out = ROOT / 'fits' / ARMS[0] / job['key']; report = read(out / 'training.json')
        if not report['protected_state_unchanged'] or report['parent_sha256'] != job['input_sha256']:
            raise ValueError('coordinate construction contract changed')
        records.append(dict(key=job['key'], report=receipt(out / 'training.json'), export=receipt(out / 'field/manifest.json')))
    write(ROOT / 'construction_complete.json', dict(records=records, field_count=len(records)))


def validate():
    from scripts.research import validate_query_field_protected as cold
    cold.ROOT = ROOT; cold.ARMS = ARMS
    cold.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'build', 'predict', 'score', 'validate', 'all'])
    parser.add_argument('--gpus', default='0,1,2'); parser.add_argument('--index', type=int)
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
