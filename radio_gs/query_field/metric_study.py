"""Declared member-metric interventions; frozen queries precede label scoring."""
import argparse
import json
from pathlib import Path

import torch

from .contracts import QueryPacket
from .public_lerf import score
from .storage import load_field, save_field, sha256


@torch.no_grad()
def package(config, job, output, device='cuda'):
    """Inference-only control: add the same physical metric without refitting."""
    if job['arm'] != 'frozen_metric':
        raise ValueError('only the frozen control can be packaged without fitting')
    field, parent = load_field(job['parent_field'], device)
    observations = Path(job['observations'])/'observations.json'
    observation_hash = sha256(observations)
    if parent['provenance']['observations_sha256'] != observation_hash:
        raise ValueError('physical units must belong to the field construction')
    intervention = config.get('member_intervention', {'spatial_metric': 'scale_normalized'})
    if not intervention or not set(intervention) <= {'spatial_metric', 'local_metric'}:
        raise ValueError('undeclared member intervention')
    for key, value in intervention.items():
        allowed = {'spatial_metric': {'scale_normalized'}, 'local_metric': {'native_cosine'}}
        if value not in allowed[key]:
            raise ValueError('unsupported member intervention')
        setattr(field, key, value)
    field.world_units_per_metre = json.loads(observations.read_text())['world_units_per_metre']
    manifest = save_field(field, job['field'], dict(
        parent_field_sha256=parent['files']['field.pt']['sha256'],
        observations_sha256=observation_hash,
        operation='Apply declared member metric; all parent tensors retained.', member_intervention=intervention,
        extra_3d_labels=False, benchmark_queries_used=False))
    return dict(field_sha256=manifest['files']['field.pt']['sha256'], spec=manifest['spec'])


@torch.no_grad()
def predict(config, job, output, device='cuda'):
    previous = Path(job['reference_prediction'])
    reference = json.loads((previous/'prediction.json').read_text())
    if sha256(previous/'predictions.pt') != reference['predictions_sha256']:
        raise ValueError('reference query vectors changed')
    data = torch.load(previous/'predictions.pt', map_location='cpu', weights_only=True)
    field, manifest = load_field(job['field'], device)
    if manifest['spec']['spatial_metric'] != 'scale_normalized':
        raise ValueError('study requires the declared spatial member metric')
    for key, value in config.get('member_intervention', {}).items():
        if manifest['spec'].get(key) != value:
            raise ValueError('export differs from declared member intervention')
    for key, value in data['geometry'].items():
        if not torch.equal(getattr(field, key).cpu(), value):
            raise ValueError('reference geometry differs')
    if not torch.equal(field.observed.cpu(), data['observed']):
        raise ValueError('observed output carrier differs')
    background = field.semantic.score(data['generic_negative_vectors']).max(1).values
    scores, selected, diagnostics = [], [], []
    for column, (text, vector) in enumerate(zip(data['texts'], data['native_query_vectors'])):
        result = field.query(QueryPacket('text', task='union', semantic=vector.to(device)))
        native = torch.sigmoid(10*(result.semantic_scores-background)).cpu()
        if not torch.equal(native, data['scores']['native_generic_relevance'][:, column]):
            raise ValueError('native response changed')
        scores.append(result.membership_scores.cpu())
        selected.append(result.selected.cpu())
        diagnostics.append(dict(text=text, selected=int(result.selected.sum()), **result.diagnostics))
        print('metric prediction', job['name'], text, flush=True)
    payload = dict(texts=data['texts'], geometry=data['geometry'], observed=data['observed'],
        scores=dict(native_generic_relevance=data['scores']['native_generic_relevance'],
                    shared_membership=torch.stack(scores, 1)), selected=torch.stack(selected, 1),
        native_query_vectors=data['native_query_vectors'], generic_negative_vectors=data['generic_negative_vectors'])
    root = Path(output)/job['name']
    root.mkdir(parents=True, exist_ok=False)
    torch.save(payload, root/'predictions.pt')
    report = dict(schema='radio_gs.query_field_metric_prediction.v1', job=job, configuration=config,
        predictions_sha256=sha256(root/'predictions.pt'), field_sha256=manifest['files']['field.pt']['sha256'],
        reference_prediction_sha256=reference['predictions_sha256'], native_response_bit_exact=True,
        member_carrier_unchanged=True, runtime_source_access=False, legacy_calibration_reused=False,
        diagnostics=diagnostics)
    (root/'prediction.json').write_text(json.dumps(report, indent=2)+'\n')
    return {key: value for key, value in report.items() if key != 'diagnostics'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['package', 'predict', 'score'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    job = next(j for j in config['jobs'] if j['name'] == args.job)
    print(json.dumps({'package': package, 'predict': predict, 'score': score}[args.stage](
        config, job, args.output, args.device), indent=2))
