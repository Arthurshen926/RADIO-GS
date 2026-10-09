"""Measure CUDA replay error and selections instead of hiding hash differences."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from radio_gs.query_field.adapters import point_2d, point_3d
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.runtime import QueryFieldRuntime
from scripts.research.run_query_field_confidence_mixture_functions import ROOT, ARMS, read, write, receipt, check
from scripts.research.run_query_field_protected import PREVIOUS

CHILD_MODULE = 'scripts.research.validate_query_field_mixture_replay'
VALIDATION_DIRECTORY = 'runtime_numeric'

@torch.no_grad()
def results(directory, fixture):
    field = QueryFieldRuntime.load(directory, 'cuda'); camera = fixture['camera']
    packets = dict(text=QueryPacket('text', semantic=fixture['text']),
        image=QueryPacket('image', semantic=fixture['image_semantic'], local=fixture['image_local']),
        point_2d=point_2d(field, fixture['pixel_xy'], camera['intrinsic'], camera['world_to_camera'],
            camera['hw'], world_units_per_metre=camera['world_units_per_metre']),
        point_3d=point_3d(field, fixture['xyz']))
    outputs = {}
    for name, packet in packets.items():
        result = field.query(packet)
        outputs[name] = dict(scores=result.membership_scores.cpu(), selected=result.selected.cpu())
        if packet.positive is not None:
            ids = packet.positive.indices.to(field.xyz.device); mask = field.observed[ids]
            weights = packet.positive.weights.to(field.xyz.device)
            support = Support(ids[mask], weights[mask])
            outputs[name].update(support_indices=support.indices.cpu(), support_weights=support.weights.cpu(),
                relation_coefficients=field.evidence.coefficients(support, result.scale).cpu())
    scores, selected = field.category_partition(fixture['category_queries'])
    outputs['category'] = dict(scores=scores.cpu(), selected=selected.cpu())
    return outputs


def differences(reference, candidate):
    records = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        record = dict(exact_scores=torch.equal(a['scores'], b['scores']),
            maximum_absolute_error=float((a['scores'] - b['scores']).abs().max()),
            different_score_entries=int((a['scores'] != b['scores']).sum()),
            exact_selections=torch.equal(a['selected'], b['selected']),
            different_selections=int((a['selected'] != b['selected']).sum()))
        if 'support_indices' in a:
            record.update(exact_support_indices=torch.equal(a['support_indices'], b['support_indices']),
                exact_support_weights=torch.equal(a['support_weights'], b['support_weights']),
                coefficient_maximum_error=float((a['relation_coefficients'] - b['relation_coefficients']).abs().max()))
        records[name] = record
    return records


def child(args):
    forbidden = [Path(p).resolve() for p in args.deny]; attempts = []
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)): return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(path == p or p in path.parents for p in forbidden):
            attempts.append(str(path)); raise PermissionError('source access denied: ' + str(path))
    sys.addaudithook(guard)
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    torch.save(results(args.field, fixture), args.output)
    print(json.dumps(dict(forbidden_open_attempts=attempts)))


def main():
    read(ROOT / 'scoring_complete.json')
    out = ROOT / 'validation' / VALIDATION_DIRECTORY; out.mkdir(parents=True, exist_ok=False)
    prior = read(PREVIOUS / 'validation/runtime/report.json'); check(prior['fixture'])
    fixture = torch.load(prior['fixture']['path'], map_location='cpu', weights_only=True)
    directory = ROOT / 'fits' / ARMS[0] / 'uqis_scene0030_00/field'
    first = results(directory, fixture); torch.save(first, out / 'original.pt')
    gc.collect(); torch.cuda.empty_cache()
    repeated = results(directory, fixture); torch.save(repeated, out / 'repeated_original.pt')
    same = differences(first, repeated); gc.collect(); torch.cuda.empty_cache()
    repo = Path.cwd().resolve()
    forbidden = [repo / 'output', repo / 'data', Path('/root/RADIO-GS/output'), Path('/root/RADIO-GS/data')]
    records = []
    with tempfile.TemporaryDirectory(prefix='query-field-mixture-cold-') as temporary:
        isolated = Path(temporary); shutil.copytree(directory, isolated / 'scene')
        shutil.copy2(prior['fixture']['path'], isolated / 'query.pt')
        for repeat in range(2):
            command = [sys.executable, '-m', CHILD_MODULE, '--child',
                '--field', str(isolated / 'scene'), '--fixture', str(isolated / 'query.pt'), '--output', str(isolated / 'result.pt')]
            for path in forbidden: command.extend(['--deny', str(path.resolve())])
            process = subprocess.run(command, cwd=isolated, env=dict(os.environ, PYTHONPATH=str(repo)),
                                     capture_output=True, text=True, check=True)
            log = json.loads(process.stdout.strip().splitlines()[-1])
            cold = torch.load(isolated / 'result.pt', map_location='cpu', weights_only=True)
            shutil.copy2(isolated / 'result.pt', out / f'copied_{repeat}.pt')
            records.append(dict(repeat=repeat, comparison=differences(first, cold), **log))
    checks = [same] + [r['comparison'] for r in records]
    valid = all(r['exact_selections'] and r['maximum_absolute_error'] <= 1e-6 for c in checks for r in c.values())
    valid = valid and not any(r['forbidden_open_attempts'] for r in records)
    report = dict(complete=True, numerically_equivalent=valid, tolerance=1e-6,
        tolerance_basis='Existing CPU/CUDA functional-test absolute tolerance; not a change to model scores or thresholds',
        original_repeat=same, records=records, fixture=prior['fixture'], field=receipt(directory / 'field.pt'),
        code=receipt(Path(__file__)), copied_export=True, fresh_process=True,
        enforcement='Python file-open audit; not an OS security sandbox',
        strict_hash_failure_preserved=True, benchmark_predictions_changed=False,
        coverage='one real scene; four input modalities and category menu; two fresh copied processes',
        evidence=[receipt(p) for p in out.glob('*.pt')])
    write(out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not valid: raise ValueError('numeric replay or actual selections differ')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--child', action='store_true')
    p.add_argument('--field'); p.add_argument('--fixture'); p.add_argument('--output'); p.add_argument('--deny', action='append', default=[])
    args = p.parse_args(); torch.set_num_threads(4)
    child(args) if args.child else main()
