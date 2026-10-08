"""Copied-export replay for both fitted arms and the complete typed program."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from radio_gs.query_field.cold_check import outputs
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.storage import load_field, sha256
from scripts.research.run_query_field_relations import ROOT, ARMS, read, write, receipt, check


def results(field_path, fixture, mode):
    field, _ = load_field(field_path, 'cuda')
    runtime = QueryFieldRuntime(field) if mode == 'common_program' else field
    answer = outputs(runtime, fixture)
    if mode == 'common_program':
        scores, labels = runtime.category_partition(fixture['category_queries'])
        answer['category'] = dict(scores_sha256=hashlib.sha256(scores.cpu().numpy().tobytes()).hexdigest(),
                                  selected_sha256=hashlib.sha256(labels.cpu().numpy().tobytes()).hexdigest(),
                                  selected=int((labels >= 0).sum()))
    return answer


def child(args):
    forbidden = [Path(p).resolve() for p in args.deny]
    attempts = []
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(path == prefix or prefix in path.parents for prefix in forbidden):
            attempts.append(str(path))
            raise PermissionError('source access denied: ' + str(path))
    sys.addaudithook(guard)
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    print(json.dumps(dict(results=results(args.field, fixture, args.mode), forbidden_open_attempts=attempts)))


def run():
    root = ROOT / 'validation/runtime'
    root.mkdir(parents=True, exist_ok=False)
    design = read(ROOT / 'design.json')
    job = next(j for j in design['fields'] if j['key'] == 'uqis_scene0030_00')
    observations = Path(job['observations']); manifest = read(observations / 'observations.json')
    entry = next(e for e in manifest['frames'] if e['split'] == 'development')
    view = torch.load(observations / f"view_{entry['frame']:05d}.pt", map_location='cpu', weights_only=True)
    encoded = next(e for e in read('output/query_field_20261008/native_identity_transfer_v1/encoding.json')['queries']
                   if e['scene'] == 'scene0030_00')
    check(encoded['encoded']); image = torch.load(encoded['encoded']['path'], weights_only=True)
    parent = read(design['parent']['path'])
    text_path = Path(parent['lerf'][0]['predictions']) / 'predictions.pt'
    text_receipt = read(text_path.parent / 'prediction.json')
    if sha256(text_path) != text_receipt['predictions_sha256']: raise ValueError('text encoding payload changed')
    text = torch.load(text_path, map_location='cpu', weights_only=True)['native_query_vectors'][:3]
    field, _ = load_field(ROOT / 'fits' / ARMS[1] / job['key'] / 'field')
    height, width = manifest['raster_hw']
    operator = view['operator'].coalesce()
    mass = torch.sparse.mm(operator, field.observed.float()[:, None])[:, 0]
    candidates = torch.where((mass > 0) & (view['best_element'] >= 0))[0]
    candidates = candidates[field.observed[view['best_element'][candidates]]]
    distance = (candidates // width - height / 2)**2 + (candidates % width - width / 2)**2
    pixel = int(candidates[distance.argmin()])
    k = view['intrinsic'].clone()
    k[0] *= width / view['original_hw'][1]; k[1] *= height / view['original_hw'][0]
    fixture = dict(text=text[0], category_queries=text, image_semantic=image['semantic'], image_local=image['local'],
        pixel_xy=[pixel % width, pixel // width], xyz=field.xyz[int(view['best_element'][pixel])].cpu(),
        camera=dict(intrinsic=k, world_to_camera=view['world_to_camera'], hw=[height, width],
                    world_units_per_metre=manifest['world_units_per_metre']))
    torch.save(fixture, root / 'query.pt')
    del field, view; gc.collect()
    repo = Path.cwd().resolve()
    forbidden = [repo / 'output', Path('/root/RADIO-GS/output'), Path(entry['rgb']['path']).parents[1]]
    reports = []
    for arm, mode in [(a, 'original') for a in ARMS] + [(ARMS[1], 'common_program')]:
        original = ROOT / 'fits' / arm / job['key'] / 'field'
        expected = results(original, fixture, mode)
        gc.collect(); torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='query-field-program-cold-') as temporary:
            isolated = Path(temporary)
            shutil.copytree(original, isolated / 'scene')
            shutil.copy2(root / 'query.pt', isolated / 'query.pt')
            command = [sys.executable, '-m', 'scripts.research.validate_query_field_program', '--child',
                       '--mode', mode, '--field', str(isolated / 'scene'), '--fixture', str(isolated / 'query.pt')]
            for p in forbidden: command.extend(['--deny', str(p.resolve())])
            process = subprocess.run(command, cwd=isolated, env=dict(os.environ, PYTHONPATH=str(repo)),
                                     capture_output=True, text=True, check=True)
            cold = json.loads(process.stdout.strip().splitlines()[-1])
        parity = {name: all(expected[name][k] == cold['results'][name][k]
                           for k in ['scores_sha256', 'selected_sha256', 'selected']) for name in expected}
        if not all(parity.values()) or cold['forbidden_open_attempts']:
            raise ValueError('copied-export replay or source denial failed')
        reports.append(dict(arm=arm, mode=mode, field=receipt(original / 'field.pt'), exact_replay=parity, **cold))
        print('cold replay complete', arm, mode, flush=True)
    write(root / 'report.json', dict(records=reports, fixture=receipt(root / 'query.pt'),
        code=[receipt(Path(__file__)), receipt(Path('radio_gs/query_field/runtime.py')),
              receipt(Path('radio_gs/query_field/query_program.py'))],
        query_encoding=dict(image=encoded, text=receipt(text_path)), copied_export=True, fresh_process=True,
        enforcement='Python file-open audit; not an OS security sandbox', benchmark_result=False,
        coverage='one real scene, four input types, three-category current-query menu; not a full-cohort cold audit'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child', action='store_true'); p.add_argument('--field'); p.add_argument('--fixture')
    p.add_argument('--mode', choices=['original', 'common_program']); p.add_argument('--deny', action='append', default=[])
    args = p.parse_args(); torch.set_num_threads(4)
    child(args) if args.child else run()
