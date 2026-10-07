"""Fixed causal controls for actual selection, using already frozen text vectors.

Native-only selection is a diagnostic, not a replacement member architecture.
The original union output must replay exactly before its controls are exported.
No target labels, cameras or source observations are opened while predicting.
"""
import argparse
import json
from pathlib import Path

import torch

from .contracts import QueryPacket
from .public_lerf import score
from .storage import load_field, sha256


@torch.no_grad()
def predict(config, job, output, device='cuda'):
    out = Path(output)/job['name']
    out.mkdir(parents=True, exist_ok=False)
    original = Path(job['reference_prediction'])
    receipt = json.loads((original/'prediction.json').read_text())
    if sha256(original/'predictions.pt') != receipt['predictions_sha256']:
        raise ValueError('reference prediction changed')
    previous = torch.load(original/'predictions.pt', map_location='cpu', weights_only=True)
    field, manifest = load_field(job['field'], device)
    if manifest['files']['field.pt']['sha256'] != receipt['field_sha256']:
        raise ValueError('field changed since the reference prediction')
    for key, value in previous['geometry'].items():
        if not torch.equal(getattr(field,key).cpu(), value):
            raise ValueError('reference geometry differs')
    if not torch.equal(field.observed.cpu(), previous['observed']):
        raise ValueError('reference observation domain differs')
    controls = {'union_member': [], 'single_anchor_member': [], 'native_only': []}
    replay = []
    for column, vector in enumerate(previous['native_query_vectors']):
        query = vector.to(device)
        union = field.query(QueryPacket('text',task='union',semantic=query))
        if not torch.equal(union.membership_scores.cpu(), previous['scores']['shared_membership'][:,column]):
            raise ValueError('original union scores did not replay exactly')
        if not torch.equal(union.selected.cpu(), previous['selected'][:,column]):
            raise ValueError('original union set did not replay exactly')
        single = field.query(QueryPacket('text',task='instance',semantic=query))
        # Both conditions use the same default scale, identity and member model.
        # The native-only condition is the unchanged four-negative response.
        controls['union_member'].append(union.membership_scores.cpu())
        controls['single_anchor_member'].append(single.membership_scores.cpu())
        controls['native_only'].append(previous['scores']['native_generic_relevance'][:,column])
        replay.append(dict(text=previous['texts'][column], union_elements=int(union.selected.sum()),
                           single_elements=int(single.selected.sum())))
        print('readout controls',job['name'],previous['texts'][column],flush=True)
    scores = {key:torch.stack(values,1) for key,values in controls.items()}
    selected = {key:(values>.5)&previous['observed'][:,None] for key,values in scores.items()}
    payload = dict(texts=previous['texts'], geometry=previous['geometry'], observed=previous['observed'],
                   scores=scores, selected_by_control=selected)
    torch.save(payload,out/'predictions.pt')
    report = dict(schema='radio_gs.query_field_readout_prediction.v1', job=job, configuration=config,
                  predictions_sha256=sha256(out/'predictions.pt'),
                  reference_prediction_sha256=receipt['predictions_sha256'],
                  field_sha256=manifest['files']['field.pt']['sha256'],
                  union_replayed_bit_exact=True, selection_threshold=.5,
                  current_text_vectors_reused=True, runtime_source_access=False,
                  evaluation_scene_opened_by_predictor=False,
                  native_only_is_diagnostic=True, development_adaptive=True,
                  predictions_precede_this_round_scoring=True, diagnostics=replay, promotion=False)
    (out/'prediction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['predict','score'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    config=json.loads(Path(args.config).read_text())
    job=next(item for item in config['jobs'] if item['name']==args.job)
    report=(predict if args.stage=='predict' else score)(config,job,args.output,args.device)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
