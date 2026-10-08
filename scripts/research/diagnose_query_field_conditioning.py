"""Source-held-out feature-binding interventions; no fitting or benchmark use.

Permutations preserve each feature group's within-query marginal distribution,
while breaking its link to Gaussian identity. This measures dependence on
source-query binding, not independent benchmark benefit or training causality.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from radio_gs.query_field.contracts import Support
from radio_gs.query_field.element_relation_training import ElementRelations, element_loss
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_elements import ROOT, read, write, receipt, execute, frozen_design


def scene(index):
    design = frozen_design(); job = design['fields'][index]
    path = ROOT / 'fits/conditional_relation' / job['key'] / 'field'
    field, _ = load_field(path, 'cuda')
    source = SourceRelations(job['observations'], field.observed, split='development')
    elements = ElementRelations(source, field.observed)
    pairs = torch.Generator().manual_seed(20261011); samples = torch.Generator().manual_seed(20261012)
    permutations = torch.Generator().manual_seed(20261013)
    records = []
    with torch.no_grad():
        for index in range(len(source.examples)):
            pair = source.pair(index, pairs); sample = elements.sample(pair, samples)
            if sample is None:
                continue
            rows, target, confidence, classes = [x.cuda() for x in sample]
            unique, inverse = torch.unique(rows, return_inverse=True)
            support = Support(torch.tensor([pair['anchor']], device='cuda'), torch.ones(1, device='cuda'))
            captured = []
            handle = field.relation.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
            try:
                original = field.member_logits(support, pair['scale'], unique)
            finally:
                handle.remove()
            features = torch.cat(captured)
            if len(features) != len(unique):
                raise ValueError('feature intervention row mismatch')
            correction = field.relation(features)
            logits = dict(original=original, residual_removed=original - correction)
            permutation = torch.randperm(len(unique), generator=permutations).cuda()
            for name, columns in [('native_binding_permuted', [1, 2]),
                                  ('geometry_binding_permuted', [3, 4, 5, 6, 7]),
                                  ('exposure_binding_permuted', [8])]:
                changed = features.clone(); changed[:, columns] = features[permutation][:, columns]
                logits[name] = original - correction + field.relation(changed)
            records.append(dict(example=index, unique_rows=len(unique),
                bce={k: float(element_loss(v[inverse], target, confidence, classes)) for k, v in logits.items()},
                residual_mean=float(correction.mean()), residual_std=float(correction.std(unbiased=False)),
                residual_mean_absolute=float(correction.abs().mean())))
    metrics = {k: float(np.mean([r['bce'][k] for r in records])) for k in records[0]['bce']}
    out = ROOT / 'diagnostics/conditioning' / (job['key'] + '.json')
    write(out, dict(key=job['key'], family=job['family'], metrics=metrics, records=records,
        field=receipt(path / 'field.pt'), observations=receipt(Path(job['observations']) / 'observations.json'),
        no_model_change=True, benchmark_queries_used=False, used_for_selection=False,
        interpretation=__doc__))


def main(gpus):
    design = frozen_design(); read(ROOT / 'construction_complete.json')
    large = [i for i, j in enumerate(design['fields']) if j['family'] == 'lerf']
    other = [i for i, j in enumerate(design['fields']) if j['family'] != 'lerf']
    chunks = [large + other] if len(gpus) == 1 else [large] + [other[i::len(gpus)-1] for i in range(len(gpus)-1)]
    def lane(i):
        for index in chunks[i]:
            key = design['fields'][index]['key']
            execute(['-m', 'scripts.research.diagnose_query_field_conditioning', '--index', str(index)],
                ROOT / 'logs/conditioning' / (key + '.log'), gpus[i])
            print('conditioning audit', key, flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(lane, range(len(gpus))))
    reports = [read(ROOT / 'diagnostics/conditioning' / (j['key'] + '.json')) for j in design['fields']]
    baseline = np.array([r['metrics']['original'] for r in reports])
    summary = {k: dict(mean_bce=float(np.mean([r['metrics'][k] for r in reports])),
        mean_delta=float(np.mean([r['metrics'][k] for r in reports] - baseline)),
        scenes_worse=sum(r['metrics'][k] > r['metrics']['original'] for r in reports))
        for k in reports[0]['metrics']}
    write(ROOT / 'diagnostics/conditioning/report.json', dict(fields=30, summary=summary, scenes=reports,
        no_model_change=True, benchmark_queries_used=False, used_for_selection=False, interpretation=__doc__,
        code=receipt(Path(__file__))))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int); parser.add_argument('--gpus', default='0,1,2')
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.index is None:
        main(list(map(int, args.gpus.split(','))))
    else:
        scene(args.index)
