"""Persist the complete prompt cohort and agreement gate, replay, then score."""
import json
from pathlib import Path
import subprocess
import sys

import torch

from radio_gs.query_field.storage import sha256
from scripts.research.prepare_competitive_evidence import ROOT, CONFIG, write
from scripts.research.run_competitive_evidence import verify_launch
from scripts.research.run_query_field_nvos import run_prediction, replay


def main():
    verify_launch()
    path = CONFIG/'competitive_evidence_nvos.json'
    config = json.loads(path.read_text())
    predictions = ROOT/'nvos'
    predictions.mkdir(exist_ok=False)
    agreements, rows, references = {}, [], {}
    for job in config['jobs']:
        run_prediction(path, job['name'], predictions, ROOT/f"predict_{job['name']}.log")
        directory = predictions/job['name']
        record = json.loads((directory/'prediction.json').read_text())
        payload = torch.load(directory/'predictions.pt', map_location='cpu', weights_only=True)
        agreements[job['name']] = record['prompt_agreement']
        exact = None
        if job['name'].startswith('product_'):
            old = ROOT.parent/'nvos_fern_v1/public'/f"trained_{job['mode']}_default"
            old_record = json.loads((old/'prediction.json').read_text())
            if sha256(old/'predictions.pt') != old_record['predictions_sha256']:
                raise ValueError('saved parent changed')
            parent = torch.load(old/'predictions.pt', map_location='cpu', weights_only=True)
            exact = {key: torch.equal(payload[key], parent[key]) for key in payload}
            references[job['mode']] = payload
        elif job['mode'] == 'positive':
            exact = {key: torch.equal(payload[key], references['positive'][key]) for key in payload}
        if exact is not None and not all(exact.values()):
            raise RuntimeError(f'parent/unsigned replay differs: {job["name"]}: {exact}')
        rows.append(dict(job=job['name'], predictions_sha256=record['predictions_sha256'],
                         parent_or_unsigned_exact=exact, prompt_agreement=record['prompt_agreement']))
        print('predicted', job['name'], record['prompt_agreement'], flush=True)
    parent = agreements['product_signed']
    gates = {}
    for rule in ['competitive_pooled', 'competitive_rays']:
        agreement = agreements[rule+'_signed']
        gates[rule] = dict(positive_kept=agreement['positive_covered_fraction'] >= parent['positive_covered_fraction']-.01,
                          negative_not_increased=agreement['negative_selected_fraction'] <= parent['negative_selected_fraction'])
    write(ROOT/'prompt_prescore.json', dict(rows=rows, gates=gates, target_labels_opened=False))
    replay(path, predictions, ROOT)
    verify_launch()
    with (ROOT/'score_nvos.log').open('x') as handle:
        subprocess.run([sys.executable, '-m', 'radio_gs.query_field.public_nvos', 'score', '--config',
                        str(CONFIG/'competitive_evidence_nvos_scoring.json'), '--output', str(predictions)],
                       check=True, stdout=handle, stderr=subprocess.STDOUT)
    print('all frozen prompt arms scored after complete source-denied replay', flush=True)


if __name__ == '__main__':
    main()
